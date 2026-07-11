import logging
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from nicegui import Client, app, run, ui
from nicegui.events import ValueChangeEventArguments

from app.config import CONVERSATIONS_DB_PATH, NICEGUI_STORAGE_SECRET
from app.domain.ports import ConversationTitler
from app.domain.recommender import TextDelta
from app.models.conversation import Conversation, ConversationMessage, MessageRole
from app.models.media_item import MediaItem, StreamingSource, VideoResolution
from app.repositories.conversation_store import ConversationStore
from app.repositories.qdrant_media_items import QdrantMediaItems
from app.services.recommendation import CardReady, ConversationalRecommendationService
from nicegui_app.components import (
    render_chat_row,
    render_movie_card,
    render_recommendations,
)
from nicegui_app.service_cache import get_service
from nicegui_app.styles import SIDEBAR_WIDTH_PX, apply_theme

_MAX_ANSWER_CHARS_FOR_TITLE = 1500

logger = logging.getLogger(__name__)

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

_store = ConversationStore(CONVERSATIONS_DB_PATH)


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
        "spoiler_free": app.storage.tab.setdefault("spoiler_free", False),
        "conversation_id": app.storage.tab.setdefault(
            "conversation_id", str(uuid.uuid4())
        ),
        # Non-None while displaying a read-only snapshot loaded from Recent —
        # persisted to app.storage.tab (not just a local var) so this stays
        # correct across a tab reload, since "conversation_id" also persists.
        "viewing_recent_id": app.storage.tab.setdefault("viewing_recent_id", None),
    }

    async def current_service() -> tuple[
        ConversationalRecommendationService, QdrantMediaItems, ConversationTitler
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
        recent_container = ui.column().classes("w-full gap-0")
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

    def render_recent_list() -> None:
        recent_container.clear()
        with recent_container:
            for conv in _store.list_recent():
                label = conv.title or "New conversation"
                row = ui.label(label).classes("plex-conv w-full")
                if conv.id == state["viewing_recent_id"]:
                    row.classes("plex-conv-active")
                row.on("click", lambda _, cid=conv.id: on_load_recent(cid))

    async def on_spoiler_toggle(e: ValueChangeEventArguments[bool | None]) -> None:
        state["spoiler_free"] = bool(e.value)
        app.storage.tab["spoiler_free"] = state["spoiler_free"]
        # Displayed transcript is intentionally NOT cleared here — this
        # matches the Streamlit app's existing behavior, where toggling the
        # switch swaps which cached service is used without touching
        # st.session_state.messages. Only "New conversation" clears it.
        await current_service()  # warm the cache for the new setting

    def _start_new_conversation() -> None:
        state["conversation_id"] = str(uuid.uuid4())
        app.storage.tab["conversation_id"] = state["conversation_id"]
        state["viewing_recent_id"] = None
        app.storage.tab["viewing_recent_id"] = None

    async def on_new_conversation() -> None:
        service, _, _ = await current_service()
        service.reset_history()
        messages.clear()
        app.storage.tab["messages"] = messages
        _start_new_conversation()
        render_stored_messages()
        render_recent_list()

    async def on_load_recent(conversation_id: str) -> None:
        conv = _store.get(conversation_id)
        if conv is None:  # stale row — e.g. pruned between render and click
            render_recent_list()
            return
        messages.clear()
        messages.extend(
            {"role": m.role.value, "content": m.content, "items": m.items}
            for m in conv.messages
        )
        app.storage.tab["messages"] = messages
        state["conversation_id"] = conversation_id
        app.storage.tab["conversation_id"] = conversation_id
        state["viewing_recent_id"] = conversation_id
        app.storage.tab["viewing_recent_id"] = conversation_id
        render_stored_messages()
        render_recent_list()

    async def _persist_current_conversation(
        latest_question: str, latest_answer: str, titler: ConversationTitler
    ) -> None:
        # This turn's own user+assistant pair, nothing before it — i.e. the
        # first exchange of a brand-new conversation, the one time a title
        # needs generating.
        is_first_exchange = len(messages) == 2
        now = datetime.now(UTC).isoformat()
        existing = None if is_first_exchange else _store.get(state["conversation_id"])
        title = existing.title if existing else None
        if is_first_exchange:
            try:
                title = await titler.title(
                    latest_question,
                    latest_answer[:_MAX_ANSWER_CHARS_FOR_TITLE],
                )
            except Exception:  # noqa: BLE001 — a titling hiccup must not lose the turn
                title = latest_question[:40]
        conversation = Conversation(
            id=state["conversation_id"],
            title=title,
            created_at=existing.created_at if existing else now,
            updated_at=now,
            messages=[
                ConversationMessage(
                    role=MessageRole(m["role"]),
                    content=m["content"],
                    items=m.get("items", []),
                )
                for m in messages
            ],
        )
        try:
            await run.io_bound(_store.save, conversation)
        except Exception:  # noqa: BLE001 — a persistence hiccup must not break the chat turn
            logger.exception("Failed to persist conversation %s", conversation.id)

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

        service, media_repo, titler = await current_service()

        # Sending while a past Recent conversation is loaded starts a
        # brand-new conversation rather than appending onto that snapshot's
        # transcript — resuming isn't supported (no LLM/RAG context was
        # restored for it), so continuing to type into a stale snapshot
        # would produce a transcript that looks continuous but whose earlier
        # turns the model never actually saw this session.
        if state["viewing_recent_id"] is not None:
            service.reset_history()
            messages.clear()
            _start_new_conversation()
            render_stored_messages()
            render_recent_list()

        messages.append({"role": "user", "content": prompt})
        render_chat_row(transcript, "user", prompt)

        assistant_body = render_chat_row(transcript, "assistant", "")
        with assistant_body:
            spinner = ui.spinner()

        streamed = await service.chat_with_items_stream(prompt, media_repo)

        spinner.delete()
        top_pick = True
        async for event in streamed.events:
            with assistant_body:
                if isinstance(event, TextDelta):
                    ui.markdown(event.text).classes("plex-msg-prose")
                elif isinstance(event, CardReady) and event.item is not None:
                    render_movie_card(event.item, event.body_md, top_pick=top_pick)
                    top_pick = False

        answer, items = streamed.answer, streamed.items
        messages.append(
            {
                "role": "assistant",
                "content": answer,
                "items": [_item_to_dict(i) for i in items],
            }
        )
        app.storage.tab["messages"] = messages

        await _persist_current_conversation(prompt, answer, titler)
        render_recent_list()

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
    render_recent_list()
    chat_input.enable()


def main() -> None:
    """Console-script entry point (`plex-rag-web`); also invoked when run as a script.

    Named `main`, not `run` — `run` would shadow the `nicegui.run` module
    imported above, which `_persist_current_conversation` calls as
    `run.io_bound(...)` (DuckDB has no async driver).
    """
    ui.run(
        title="Plex Movie Assistant",
        dark=True,
        storage_secret=NICEGUI_STORAGE_SECRET,
        reload=False,
        favicon="🎬",
    )


if __name__ in {"__main__", "__mp_main__"}:
    main()
