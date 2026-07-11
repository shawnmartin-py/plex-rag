from pathlib import Path
from typing import Any

from nicegui import Client, app, run, ui
from nicegui.events import ValueChangeEventArguments

from app.config import NICEGUI_STORAGE_SECRET
from app.models.media_item import MediaItem, StreamingSource, VideoResolution
from app.repositories.qdrant_media_items import QdrantMediaItems
from app.services.recommendation import ConversationalRecommendationService
from nicegui_app.components import render_chat_row, render_recommendations
from nicegui_app.service_cache import get_service
from nicegui_app.styles import SIDEBAR_WIDTH_PX, apply_theme

# Streaming-platform logo badges (see app/formatting/media_badge.py) — served from
# /static rather than embedded as data URIs so the browser can cache them.
app.add_static_files("/static", Path(__file__).parent / "static")


# SF-Symbols-style sidebar glyph (rounded panel with divider) — Material's
# view_sidebar icon reads too heavy for the frosted chrome.
_SIDEBAR_GLYPH = (
    '<svg width="17" height="17" viewBox="0 0 24 24" fill="none" '
    'stroke="currentColor" stroke-width="1.8" stroke-linecap="round">'
    '<rect x="3" y="5" width="18" height="14" rx="3.5"/><path d="M9.5 5v14"/>'
    "</svg>"
)

# Static placeholder until conversation persistence lands — gives the sidebar
# its intended shape (per the approved design mock) ahead of the feature.
_PLACEHOLDER_RECENTS = [
    "Unsettling psychological horror",
    "Rainy-Sunday comfort films",
    "Heist thrillers with a twist",
    "Movie night with the kids",
]


def _item_to_dict(item: MediaItem) -> dict[str, Any]:
    return {
        "imdb_id": item.imdb_id,
        "type": item.type,
        "title": item.title,
        "year": item.year,
        "imdb_rating": item.imdb_rating,
        "content_rating": item.content_rating,
        "genres": item.genres,
        "thumb_url": item.thumb_url,
        # app.storage.tab persists via JSON, so enums are stored as their raw value
        # and rebuilt in _dict_to_item — MediaItem itself isn't JSON-serializable.
        "video_resolution": item.video_resolution.value
        if item.video_resolution
        else None,
        "source_platform": item.source_platform.value if item.source_platform else None,
    }


def _dict_to_item(data: dict[str, Any]) -> MediaItem:
    return MediaItem(
        **{
            **data,
            "video_resolution": VideoResolution(data["video_resolution"])
            if data.get("video_resolution")
            else None,
            "source_platform": StreamingSource(data["source_platform"])
            if data.get("source_platform")
            else None,
        }
    )


@ui.page("/")
async def index(client: Client) -> None:
    apply_theme()
    await client.connected()

    # Per-tab displayed transcript — survives page reloads within the tab,
    # unlike a plain closure var (NiceGUI re-runs this function from scratch
    # on every reload). Distinct from the LLM conversation history, which
    # lives inside the (intentionally shared, see service_cache.py)
    # ConversationalRecommendationService instance.
    messages: list[dict[str, Any]] = app.storage.tab.setdefault("messages", [])
    state: dict[str, Any] = {
        "spoiler_free": app.storage.tab.setdefault("spoiler_free", False)
    }

    async def current_service() -> tuple[
        ConversationalRecommendationService, QdrantMediaItems
    ]:
        return await get_service(state["spoiler_free"])

    with (
        ui.left_drawer(value=True, fixed=True, bordered=False)
        .classes("plex-sidebar")
        .props(f"width={SIDEBAR_WIDTH_PX}")
    ) as drawer:
        with ui.row().classes("plex-sb-head w-full items-center"):
            with ui.element("div").classes("plex-app-mark"):
                ui.icon("play_arrow")
            ui.label("Plex Assistant").classes("plex-sb-name")
            with (
                ui.button(color=None, on_click=drawer.toggle)
                .props("flat round dense")
                .classes("plex-icon-btn")
            ):
                ui.html(_SIDEBAR_GLYPH)
        new_conv_btn = (
            ui.button("New conversation", icon="add", color=None)
            .props("flat no-caps")
            .classes("plex-new-conv-btn w-full")
        )
        ui.label("Recent").classes("plex-sec-label")
        for i, conv_title in enumerate(_PLACEHOLDER_RECENTS):
            conv = ui.label(conv_title).classes("plex-conv w-full")
            if i == 0:
                conv.classes("plex-conv-active")
        with ui.row().classes("plex-sb-bottom w-full"):
            spoiler_switch = ui.switch("Spoiler-free mode", value=state["spoiler_free"])

    # Shown only while the drawer is hidden — pops the sidebar back out.
    with (
        ui.button(color=None, on_click=drawer.toggle)
        .props("flat round dense")
        .classes("plex-icon-btn plex-float-toggle")
        .bind_visibility_from(drawer, "value", backward=lambda v: not v)
    ):
        ui.html(_SIDEBAR_GLYPH)

    with ui.column().classes("plex-main w-full items-stretch"):
        transcript = ui.column().classes("plex-transcript w-full")
        with ui.row().classes("plex-input-row w-full items-center"):
            chat_input = (
                ui.input(placeholder="Ask for a movie recommendation...")
                .classes("plex-chat-input flex-grow")
                .props("outlined dense")
            )
            with chat_input.add_slot("append"):
                send_icon = ui.icon("arrow_upward", color=None).classes(
                    "plex-send-icon cursor-pointer"
                )
                send_icon.bind_visibility_from(
                    chat_input, "value", backward=lambda v: bool(v and v.strip())
                )

    def render_stored_messages() -> None:
        transcript.clear()
        for msg in messages:
            if msg["role"] == "assistant":
                body = render_chat_row(transcript, "assistant", "")
                items = [_dict_to_item(d) for d in msg.get("items", [])]
                render_recommendations(body, msg["content"], items)
            else:
                render_chat_row(transcript, "user", msg["content"])

    async def on_spoiler_toggle(e: ValueChangeEventArguments[bool | None]) -> None:
        state["spoiler_free"] = bool(e.value)
        app.storage.tab["spoiler_free"] = state["spoiler_free"]
        # Displayed transcript is intentionally NOT cleared here — this
        # matches the Streamlit app's existing behavior, where toggling the
        # switch swaps which cached service is used without touching
        # st.session_state.messages. Only "New conversation" clears it.
        await current_service()  # warm the cache for the new setting

    async def on_new_conversation() -> None:
        service, _ = await current_service()
        service.reset_history()
        messages.clear()
        app.storage.tab["messages"] = messages
        render_stored_messages()

    busy = {"value": False}

    async def on_send() -> None:
        if busy["value"]:
            return
        prompt = (chat_input.value or "").strip()
        if not prompt:
            return
        busy["value"] = True
        chat_input.set_value("")
        chat_input.disable()

        messages.append({"role": "user", "content": prompt})
        render_chat_row(transcript, "user", prompt)

        assistant_body = render_chat_row(transcript, "assistant", "")
        with assistant_body:
            spinner = ui.spinner()

        service, media_repo = await current_service()
        result = await run.io_bound(service.chat_with_items, prompt, media_repo)
        items: list[MediaItem]
        if result is None:
            answer, items = "Something went wrong. Please try again.", []
        else:
            answer, items = result

        spinner.delete()
        render_recommendations(assistant_body, answer, items)
        messages.append(
            {
                "role": "assistant",
                "content": answer,
                "items": [_item_to_dict(i) for i in items],
            }
        )
        app.storage.tab["messages"] = messages

        chat_input.enable()
        busy["value"] = False

    spoiler_switch.on_value_change(on_spoiler_toggle)
    new_conv_btn.on_click(on_new_conversation)
    send_icon.on("click", on_send)
    chat_input.on("keydown.enter", on_send)

    with transcript:
        loading = ui.spinner(size="lg")
    chat_input.disable()
    await current_service()  # warm the initial cache (spinner while it builds)
    loading.delete()
    render_stored_messages()
    chat_input.enable()


if __name__ in {"__main__", "__mp_main__"}:
    ui.run(
        title="Plex Movie Assistant",
        dark=True,
        storage_secret=NICEGUI_STORAGE_SECRET,
        reload=False,
        favicon="🎬",
    )
