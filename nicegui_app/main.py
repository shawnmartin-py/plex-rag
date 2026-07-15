import asyncio
import logging
import sys
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from nicegui import Client, app, run, ui
from nicegui.events import ValueChangeEventArguments

from app.adapters.poster_accent import PosterAccents
from app.config import (
    CONVERSATIONS_DB_PATH,
    NICEGUI_STORAGE_SECRET,
    QDRANT_COLLECTION,
    QDRANT_URL,
)
from app.domain.diversity import NoWatchHistoryError
from app.domain.ports import ConversationTitler, TextDelta
from app.models.conversation import Conversation, ConversationMessage, MessageRole
from app.models.media_item import (
    HdrFormat,
    MediaItem,
    StreamingSource,
    VideoResolution,
)
from app.repositories.conversation_store import ConversationStore
from app.repositories.qdrant_media_items import QdrantMediaItems
from app.repositories.vector_store import (
    QdrantUnavailableError,
    ensure_qdrant_reachable,
)
from app.services.recommendation import CardReady, ConversationalRecommendationService
from nicegui_app.components import (
    render_chat_row,
    render_movie_card,
    render_recommendations,
    render_surprise_results,
)
from nicegui_app.service_cache import get_diversity_service, get_service
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

# Process-lifetime accent cache — poster colors don't change while running.
_poster_accents = PosterAccents()


async def _accent_for(item: MediaItem) -> tuple[str, ...] | None:
    """Key-light colors for a card, fetched/cached via the shared async client."""
    if not item.thumb_url:
        return None
    return await _poster_accents.accent_for(item.thumb_url)


# "Tonight" sidebar chips: label shown on the chip -> the chat message a
# click actually sends (through the exact same turn path as typed input).
# Deliberately curator-shaped rather than filter-shaped ("90s thriller",
# "under 2 hours") — these lean on the enrichment profiles the retrievers
# index, asking things a browse UI can't answer.
_TONIGHT_PROMPTS: dict[str, str] = {
    "Hidden gem": (
        "What's the most overlooked film in my library — something I "
        "probably don't realize is great? Make the case for it."
    ),
    "Mind-bender": (
        "Recommend a film that will mess with my head — unreliable "
        "narration, twisted structure, or a reality that doesn't hold."
    ),
    "Worth watching loud": (
        "Recommend a film whose score or soundtrack carries it — something "
        "worth turning up."
    ),
    "Defies genre": (
        "Recommend the film in my library that most resists genre labels — "
        "and tell me what it actually is."
    ),
    "A director's best": (
        "Pick a director represented in my library and make the case for "
        "their strongest film I own."
    ),
    "So bad it's good": (
        "Recommend the most enjoyably ridiculous film in my library — "
        "embrace the trash."
    ),
}


def _item_to_dict(item: MediaItem) -> dict[str, Any]:
    return {
        "tmdb_id": item.tmdb_id,
        "imdb_id": item.imdb_id,
        "type": item.type,
        "title": item.title,
        "year": item.year,
        "imdb_rating": item.imdb_rating,
        "content_rating": item.content_rating,
        "genres": item.genres,
        "description": item.description,
        "thumb_url": item.thumb_url,
        # app.storage.tab persists via JSON, so enums are stored as their raw value
        # and rebuilt in _dict_to_item — MediaItem itself isn't JSON-serializable.
        "video_resolution": item.video_resolution.value
        if item.video_resolution
        else None,
        "hdr_formats": [fmt.value for fmt in item.hdr_formats],
        "source_platform": item.source_platform.value if item.source_platform else None,
        "runtime_minutes": item.runtime_minutes,
    }


def _dict_to_item(data: dict[str, Any]) -> MediaItem:
    return MediaItem(
        **{
            **data,
            "video_resolution": VideoResolution(data["video_resolution"])
            if data.get("video_resolution")
            else None,
            # .get() with a default: tab storage written before hdr_formats
            # existed survives a reload after a redeploy
            "hdr_formats": [HdrFormat(v) for v in data.get("hdr_formats", [])],
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
        # behavior=desktop: below ~1024px Quasar otherwise switches the
        # drawer to mobile overlay mode, whose dimming backdrop over this
        # near-black theme makes the whole page (input bar included) read
        # as empty. Docked-always is right for a desktop-shaped app; the
        # toggle still collapses it on narrow windows.
        .props(f"width={SIDEBAR_WIDTH_PX} behavior=desktop")
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
        surprise_btn = (
            ui.button("Surprise me", icon="shuffle", color=None)
            .props("flat no-caps")
            .classes("plex-new-conv-btn w-full")
        )
        ui.label("Recent").classes("plex-sec-label")
        recent_container = ui.column().classes("w-full gap-0")
        ui.label("Tonight").classes("plex-sec-label")
        with ui.row().classes("plex-chip-row w-full"):
            for chip_label, chip_prompt in _TONIGHT_PROMPTS.items():
                # Same late-binding closure pattern as the Recent rows below:
                # run_turn is defined later in this page function but resolved
                # at click time.
                ui.label(chip_label).classes("plex-chip").on(
                    "click", lambda _, p=chip_prompt: run_turn(p)
                )
        # "Unwatched", not "Library": the media_items collection only holds
        # the unwatched catalog (plex-ingest's staging filters watched movies
        # out), so every stat below is a count of the recommendable shelf.
        ui.label("Unwatched").classes("plex-sec-label")
        stats_container = ui.column().classes("plex-stats w-full")
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
        # Real sibling element, not a ::before on the input row: a fixed-
        # position pseudo-element with a negative z-index nested inside a
        # sticky-positioned (stacking-context-forming) parent is exactly the
        # class of construct Safari has mis-rendered here before (see the
        # poster key-light history in app/adapters/poster_accent.py) — it
        # painted the fade over the input instead of behind it. A plain
        # element ordered before the row and given a lower, non-negative
        # z-index needs no cross-stacking-context assumptions.
        ui.element("div").classes("plex-input-fade")
        input_row = ui.row().classes("plex-input-row w-full items-center")
        with input_row:
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

    async def render_stored_messages() -> None:
        transcript.clear()
        for msg in messages:
            if msg["role"] == "assistant":
                body = render_chat_row(transcript, "assistant", "")
                items = [_dict_to_item(d) for d in msg.get("items", [])]
                accent_results = await asyncio.gather(*(_accent_for(i) for i in items))
                accents = dict(
                    zip((i.tmdb_id for i in items), accent_results, strict=True)
                )
                if msg.get("is_surprise"):
                    render_surprise_results(body, msg["content"], items, accents)
                else:
                    render_recommendations(body, msg["content"], items, accents)
            else:
                render_chat_row(transcript, "user", msg["content"])

    def _is_surprise_conversation() -> bool:
        return any(m.get("is_surprise") for m in messages)

    def _is_read_only_view() -> bool:
        # Two reasons the text input has no business being on screen:
        # - Viewing a Recent conversation: resuming it isn't implemented (no
        #   LLM/RAG history was restored for it), so it's a snapshot, not
        #   something you can continue.
        # - A "Surprise me" turn: it comes from the diversity recommender,
        #   which never joins the RAG chat history, so typing a follow-up
        #   into a conversation that already contains one would produce a
        #   reply the model has no memory of the surprise turn. The two
        #   features haven't been reasoned through together yet.
        return state["viewing_recent_id"] is not None or _is_surprise_conversation()

    def _apply_input_lock() -> None:
        locked = _is_read_only_view()
        input_row.set_visibility(not locked)
        if locked:
            chat_input.set_value("")
            chat_input.disable()
        else:
            chat_input.enable()

    def render_library_stats(items: list[MediaItem]) -> None:
        """Counts of the unwatched shelf, derived from what the contract
        actually stores (see docs/vector-store-contract.md) — no series or
        sync-age rows because neither exists in the collection's payload."""
        stats_container.clear()
        rows = [
            ("Movies", len(items)),
            (
                "In 4K",
                sum(1 for i in items if i.video_resolution is VideoResolution.R4K),
            ),
            ("Via streaming", sum(1 for i in items if i.source_platform is not None)),
        ]
        with stats_container:
            for name, count in rows:
                with ui.row().classes("plex-stat w-full justify-between items-center"):
                    ui.label(name).classes("plex-stat-name")
                    ui.label(f"{count:,}").classes("plex-stat-value")

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

    # Bumped by _start_new_conversation() below. A turn (run_turn/on_surprise)
    # snapshots this right before it starts touching shared state and checks
    # it again right before persisting/rendering its result — if "New
    # conversation"/a Recent click has bumped it in the meantime, the turn
    # discards its own outcome instead of appending onto (or re-locking) a
    # view the user has already left. Blocking those two handlers on `busy`
    # instead (an earlier attempt at this fix) made them silently do nothing
    # for as long as a turn's trailing persistence/poster-fetch work was
    # still running — up to a couple of seconds — with zero feedback, which
    # read as "New conversation is broken."
    state["turn_token"] = 0

    def _start_new_conversation() -> None:
        state["conversation_id"] = str(uuid.uuid4())
        app.storage.tab["conversation_id"] = state["conversation_id"]
        state["viewing_recent_id"] = None
        app.storage.tab["viewing_recent_id"] = None
        state["turn_token"] += 1

    async def on_new_conversation() -> None:
        service, _, _ = await current_service()
        service.reset_history()
        messages.clear()
        app.storage.tab["messages"] = messages
        _start_new_conversation()
        await render_stored_messages()
        render_recent_list()
        _apply_input_lock()
        # Whatever turn was in flight no longer has anything to finish for
        # this view — its result will be discarded via turn_token once it
        # does resolve (see run_turn/on_surprise), so don't leave the UI
        # looking busy on its behalf.
        busy["value"] = False
        surprise_btn.enable()

    async def on_load_recent(conversation_id: str) -> None:
        conv = _store.get(conversation_id)
        if conv is None:  # stale row — e.g. pruned between render and click
            render_recent_list()
            return
        messages.clear()
        messages.extend(
            {
                "role": m.role.value,
                "content": m.content,
                "items": m.items,
                "is_surprise": m.is_surprise,
            }
            for m in conv.messages
        )
        app.storage.tab["messages"] = messages
        state["conversation_id"] = conversation_id
        app.storage.tab["conversation_id"] = conversation_id
        state["viewing_recent_id"] = conversation_id
        app.storage.tab["viewing_recent_id"] = conversation_id
        state["turn_token"] += 1
        await render_stored_messages()
        render_recent_list()
        _apply_input_lock()
        # Same reasoning as on_new_conversation: don't leave the UI busy on
        # behalf of a turn this snapshot has already superseded.
        busy["value"] = False
        surprise_btn.enable()

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
                    is_surprise=m.get("is_surprise", False),
                )
                for m in messages
            ],
        )
        try:
            await run.io_bound(_store.save, conversation)
        except Exception:  # noqa: BLE001 — a persistence hiccup must not break the chat turn
            logger.exception("Failed to persist conversation %s", conversation.id)

    busy = {"value": False}

    async def run_turn(prompt: str) -> None:
        """One full chat turn — shared by typed input and the Tonight chips."""
        if busy["value"]:
            return
        busy["value"] = True
        chat_input.disable()

        service, media_repo, titler = await current_service()

        # Sending while a past Recent conversation is loaded, or while the
        # live conversation already has a Surprise-me turn in it, starts a
        # brand-new conversation rather than appending onto that transcript.
        # Neither the Recent snapshot (no LLM/RAG context was restored for
        # it) nor a Surprise-me turn (the diversity recommender never joins
        # the RAG chat history) has anything a continued chat could build
        # on — this is a defense-in-depth fallback for the Tonight chips,
        # which stay clickable even while the text input itself is hidden
        # (see _is_read_only_view).
        if _is_read_only_view():
            service.reset_history()
            messages.clear()
            _start_new_conversation()
            await render_stored_messages()
            render_recent_list()

        # Captured after the reset above (which is this turn's own, if it
        # happened) so it reflects the view this turn is actually building
        # on. Checked again below, after the awaits that let "New
        # conversation"/a Recent click run concurrently and move on without
        # this turn — see the comment on state["turn_token"].
        my_token = state["turn_token"]

        messages.append({"role": "user", "content": prompt})
        render_chat_row(transcript, "user", prompt)

        assistant_body = render_chat_row(transcript, "assistant", "")
        with assistant_body:
            spinner = ui.spinner()

        streamed = await service.chat_with_items_stream(prompt, media_repo)

        if state["turn_token"] != my_token:
            # Superseded while the LLM call was in flight — assistant_body
            # was already removed from the transcript by the reset that did
            # it, so there's nothing left to render into or lock.
            return

        spinner.delete()
        top_pick = True
        async for event in streamed.events:
            if state["turn_token"] != my_token:
                break
            if isinstance(event, TextDelta):
                with assistant_body:
                    ui.markdown(event.text).classes("plex-msg-prose")
            elif isinstance(event, CardReady) and event.item is not None:
                # Accent resolved before entering the container context — no
                # awaits inside a `with <element>:` block.
                accent = await _accent_for(event.item)
                if state["turn_token"] != my_token:
                    break
                with assistant_body:
                    render_movie_card(
                        event.item, event.body_md, top_pick=top_pick, accent=accent
                    )
                top_pick = False

        if state["turn_token"] != my_token:
            return

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
        _apply_input_lock()

        busy["value"] = False

    async def on_send() -> None:
        if busy["value"]:
            return
        prompt = (chat_input.value or "").strip()
        if not prompt:
            return
        chat_input.set_value("")
        await run_turn(prompt)

    async def on_surprise() -> None:
        if busy["value"]:
            return
        busy["value"] = True
        surprise_btn.disable()

        _, _, titler = await current_service()

        # Same reset as run_turn's read-only-view fallback (see
        # _is_read_only_view): a past Recent snapshot has nothing to build
        # on, and neither does a conversation that already holds a
        # Surprise-me turn — each pull from the diversity recommender is
        # independent, so without this a second click just kept appending
        # more picks onto the same turn instead of starting a new one.
        if _is_read_only_view():
            chat_service, _, _ = await current_service()
            chat_service.reset_history()
            messages.clear()
            _start_new_conversation()
            await render_stored_messages()
            render_recent_list()

        # See the comment on state["turn_token"] and the matching capture in
        # run_turn.
        my_token = state["turn_token"]

        prompt = "Surprise me"
        messages.append({"role": "user", "content": prompt})
        render_chat_row(transcript, "user", prompt)

        assistant_body = render_chat_row(transcript, "assistant", "")
        with assistant_body:
            spinner = ui.spinner()

        diversity_service = await get_diversity_service()
        items: list[MediaItem] = []

        if diversity_service is None:
            answer = (
                "Diversity mode isn't set up yet — the watch_history collection "
                "hasn't been populated. Run plex-ingest's watch_history pipeline "
                "first."
            )
        else:
            try:
                result = await run.io_bound(diversity_service.recommend)
            except NoWatchHistoryError:
                answer = (
                    "No recent watch history found — watch something on Plex first!"
                )
            else:
                # run.io_bound returns None only if the call was cancelled — never
                # a legitimate outcome of recommend(), which returns [] at worst.
                if result is None:
                    answer = "Something went wrong — please try again."
                else:
                    items = result
                    answer = (
                        "Something different, based on your recent watches:"
                        if items
                        else "Nothing left to recommend right now — try again later."
                    )

        if state["turn_token"] != my_token:
            # Superseded while the diversity lookup was in flight —
            # assistant_body was already removed from the transcript by the
            # reset that did it, so there's nothing left to render into or
            # lock. surprise_btn stays as the superseding reset left it.
            return

        spinner.delete()
        accent_results = await asyncio.gather(*(_accent_for(i) for i in items))
        accents = dict(zip((i.tmdb_id for i in items), accent_results, strict=True))
        if state["turn_token"] != my_token:
            return
        render_surprise_results(assistant_body, answer, items, accents)

        messages.append(
            {
                "role": "assistant",
                "content": answer,
                "items": [_item_to_dict(i) for i in items],
                "is_surprise": True,
            }
        )
        app.storage.tab["messages"] = messages

        await _persist_current_conversation(prompt, answer, titler)
        render_recent_list()
        _apply_input_lock()

        surprise_btn.enable()
        busy["value"] = False

    spoiler_switch.on_value_change(on_spoiler_toggle)
    new_conv_btn.on_click(on_new_conversation)
    surprise_btn.on_click(on_surprise)
    send_icon.on("click", on_send)
    chat_input.on("keydown.enter", on_send)

    with transcript:
        loading = ui.spinner(size="lg")
    chat_input.disable()
    # Warm the initial cache (spinner while it builds); the media repo also
    # feeds the sidebar's library snapshot.
    _, media_repo, _ = await current_service()
    loading.delete()
    render_library_stats(media_repo.all_items())
    await render_stored_messages()
    render_recent_list()
    _apply_input_lock()


def main() -> None:
    """Console-script entry point (`plex-rag-web`); also invoked when run as a script.

    Named `main`, not `run` — `run` would shadow the `nicegui.run` module
    imported above, which `_persist_current_conversation` calls as
    `run.io_bound(...)` (DuckDB has no async driver).
    """
    # Fail fast, before the server ever starts accepting connections, rather
    # than letting a browser tab's first page load be where a dead Qdrant
    # container is discovered (see app/repositories/vector_store.py).
    try:
        ensure_qdrant_reachable(QDRANT_URL, QDRANT_COLLECTION)
    except QdrantUnavailableError as e:
        print(f"\n{e}\n", file=sys.stderr)
        raise SystemExit(1) from None

    ui.run(
        title="Plex Movie Assistant",
        dark=True,
        storage_secret=NICEGUI_STORAGE_SECRET,
        reload=False,
        favicon="🎬",
    )


if __name__ in {"__main__", "__mp_main__"}:
    main()
