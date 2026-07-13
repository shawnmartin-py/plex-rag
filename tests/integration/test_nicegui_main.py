import asyncio
import time
from collections.abc import AsyncIterator
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from nicegui import ui
from nicegui.testing import user_simulation

import app.config as config
import nicegui_app.service_cache as service_cache
from app.domain.ports import ConversationTitler
from app.models.media_item import MediaItem, StreamingSource, VideoResolution
from app.repositories.conversation_store import ConversationStore
from app.repositories.qdrant_media_items import QdrantMediaItems
from app.services.diversity_recommendation import DiversityRecommendationService
from app.services.recommendation import (
    CardReady,
    ChatStreamEvent,
    ConversationalRecommendationService,
    StreamedChatAnswer,
)

# `user_simulation(main_file=...)` runs nicegui_app/main.py itself (via
# runpy, as if executed directly) inside an isolated in-process ASGI app, so
# the real `@ui.page("/")` / event-handler wiring is exercised end to end —
# only `build_recommender_service` (Qdrant + Gemini) is mocked out.
_MAIN_FILE = Path(__file__).resolve().parents[2] / "nicegui_app" / "main.py"


def make_item(imdb_id: str, title: str) -> MediaItem:
    return MediaItem(
        imdb_id=imdb_id,
        type="movie",
        title=title,
        year=2020,
        imdb_rating=8.0,
        content_rating="R",
        genres=["Drama"],
    )


async def _aevents(items: list[MediaItem]) -> AsyncIterator[ChatStreamEvent]:
    for item in items:
        yield CardReady(item=item, body_md=f"Reasoning for {item.title}.")


def make_answer(
    answer: str, items: list[MediaItem] | None = None
) -> StreamedChatAnswer:
    resolved_items = items if items is not None else [make_item("tt1", "Heat")]
    return StreamedChatAnswer(
        events=_aevents(resolved_items), answer=answer, items=resolved_items
    )


def make_service(answer: str = "1. **Heat** (1995)\nA tense heist.") -> MagicMock:
    service = MagicMock(spec=ConversationalRecommendationService)
    service.chat_with_items_stream.return_value = make_answer(answer)
    return service


def make_titler(title: str = "Heist thrillers with a twist") -> MagicMock:
    titler = MagicMock(spec=ConversationTitler)
    titler.title.return_value = title
    return titler


def make_diversity_service(items: list[MediaItem] | None = None) -> MagicMock:
    service = MagicMock(spec=DiversityRecommendationService)
    service.recommend.return_value = (
        items if items is not None else [make_item("tt9", "Paprika")]
    )
    return service


@pytest.fixture(autouse=True)
def _isolated_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    """`service_cache._cache`/`_lock` are process-lifetime globals shared by
    every page load — reset them per test so tests can't leak cached
    (mocked) services into one another. The diversity ("Surprise me")
    globals are the same kind of process-lifetime cache, reset for the same
    reason."""
    monkeypatch.setattr(service_cache, "_cache", {})
    monkeypatch.setattr(service_cache, "_lock", asyncio.Lock())
    monkeypatch.setattr(service_cache, "_diversity_service", None)
    monkeypatch.setattr(service_cache, "_diversity_loaded", False)
    monkeypatch.setattr(service_cache, "_diversity_lock", asyncio.Lock())


@pytest.fixture(autouse=True)
def _isolated_conversation_store(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Path:
    """`nicegui_app/main.py` is re-executed via `runpy` on every
    `user_simulation(main_file=...)` call, so its module-level `_store`
    can't be reached/monkeypatched directly from here (each run gets a fresh
    module namespace, not the cached `nicegui_app.main`). Patching
    `app.config.CONVERSATIONS_DB_PATH` before that runpy execution works
    instead, since `main.py`'s `from app.config import CONVERSATIONS_DB_PATH`
    reads the (already-patched) attribute at that point — giving every test
    its own on-disk DuckDB file rather than the real `data/` one."""
    db_path = tmp_path / "conversations.duckdb"
    monkeypatch.setattr(config, "CONVERSATIONS_DB_PATH", str(db_path))
    return db_path


@pytest.mark.anyio
async def test_page_loads_and_enables_chat_input() -> None:
    media_repo = MagicMock(spec=QdrantMediaItems)
    built = (make_service(), media_repo, make_titler())
    with patch.object(
        service_cache, "build_recommender_service", return_value=built
    ) as mock_build:
        async with user_simulation(main_file=str(_MAIN_FILE)) as user:
            await user.open("/")
            await user.should_see(kind=ui.input)
            input_el = next(iter(user.find(kind=ui.input).elements))
            assert input_el.enabled

    mock_build.assert_called_once_with(spoiler_free=False)


@pytest.mark.anyio
async def test_send_message_renders_user_and_assistant_turns() -> None:
    service = make_service()
    media_repo = MagicMock(spec=QdrantMediaItems)
    with patch.object(
        service_cache,
        "build_recommender_service",
        return_value=(service, media_repo, make_titler()),
    ):
        async with user_simulation(main_file=str(_MAIN_FILE)) as user:
            await user.open("/")
            await user.should_see(kind=ui.input)
            user.find(kind=ui.input).type("Recommend a heist movie")
            user.find(kind=ui.input).trigger("keydown.enter")

            await user.should_see(content="Recommend a heist movie")
            await user.should_see(content="Heat")

    service.chat_with_items_stream.assert_called_once_with(
        "Recommend a heist movie", media_repo
    )


@pytest.mark.anyio
async def test_new_conversation_clears_transcript_and_resets_history() -> None:
    service = make_service()
    media_repo = MagicMock(spec=QdrantMediaItems)
    with patch.object(
        service_cache,
        "build_recommender_service",
        return_value=(service, media_repo, make_titler()),
    ):
        async with user_simulation(main_file=str(_MAIN_FILE)) as user:
            await user.open("/")
            await user.should_see(kind=ui.input)
            user.find(kind=ui.input).type("Recommend a heist movie")
            user.find(kind=ui.input).trigger("keydown.enter")
            await user.should_see(content="Heat")

            user.find(content="New conversation").click()
            await user.should_not_see(content="Recommend a heist movie")
            await user.should_not_see(content="Heat")

    service.reset_history.assert_called_once()


@pytest.mark.anyio
async def test_spoiler_toggle_warms_new_cache_without_clearing_transcript() -> None:
    normal_service = make_service()
    spoiler_free_service = make_service()
    media_repo = MagicMock(spec=QdrantMediaItems)

    def fake_build(spoiler_free: bool) -> tuple[MagicMock, MagicMock, MagicMock]:
        service = spoiler_free_service if spoiler_free else normal_service
        return (service, media_repo, make_titler())

    with patch.object(
        service_cache, "build_recommender_service", side_effect=fake_build
    ) as mock_build:
        async with user_simulation(main_file=str(_MAIN_FILE)) as user:
            await user.open("/")
            await user.should_see(kind=ui.input)
            user.find(kind=ui.input).type("Recommend a heist movie")
            user.find(kind=ui.input).trigger("keydown.enter")
            await user.should_see(content="Heat")

            user.find(kind=ui.switch).click()
            await user.should_see(content="Heat")  # transcript untouched
            # The switch's on_value_change handler runs as a scheduled
            # background task, not synchronously with the click — poll for
            # it to complete rather than asserting immediately.
            for _ in range(20):
                if mock_build.call_count >= 2:
                    break
                await asyncio.sleep(0.05)

    mock_build.assert_any_call(spoiler_free=False)
    mock_build.assert_any_call(spoiler_free=True)


@pytest.mark.anyio
async def test_tonight_chip_sends_its_canned_prompt() -> None:
    service = make_service()
    media_repo = MagicMock(spec=QdrantMediaItems)
    with patch.object(
        service_cache,
        "build_recommender_service",
        return_value=(service, media_repo, make_titler()),
    ):
        async with user_simulation(main_file=str(_MAIN_FILE)) as user:
            await user.open("/")
            await user.should_see(kind=ui.input)

            user.find(content="Mind-bender").click()
            await user.should_see(content="Heat")
            # The chip's full prompt renders as the user turn, not the label.
            await user.should_see(content="Recommend a film that will mess")

    (call,) = service.chat_with_items_stream.call_args_list
    assert call.args[0].startswith("Recommend a film that will mess with my head")
    assert call.args[1] is media_repo


@pytest.mark.anyio
async def test_library_stats_render_counts_from_media_repo() -> None:
    media_repo = MagicMock(spec=QdrantMediaItems)
    items = [make_item("tt1", "A"), make_item("tt2", "B"), make_item("tt3", "C")]
    items[0].video_resolution = VideoResolution.R4K
    items[1].source_platform = StreamingSource.NETFLIX
    media_repo.all_items.return_value = items
    built = (make_service(), media_repo, make_titler())
    with patch.object(service_cache, "build_recommender_service", return_value=built):
        async with user_simulation(main_file=str(_MAIN_FILE)) as user:
            await user.open("/")
            await user.should_see(kind=ui.input)
            # The section header scopes all three rows to the unwatched
            # catalog — media_items only holds unwatched movies.
            await user.should_see(content="Unwatched")
            await user.should_see(content="Movies")
            await user.should_see(content="In 4K")
            await user.should_see(content="Via streaming")
            await user.should_see(content="3")


# --- Recent conversations ---


@pytest.mark.anyio
async def test_recent_list_shows_conversation_after_first_exchange(
    _isolated_conversation_store: Path,
) -> None:
    service = make_service()
    media_repo = MagicMock(spec=QdrantMediaItems)
    titler = make_titler("Heist thrillers with a twist")
    with patch.object(
        service_cache,
        "build_recommender_service",
        return_value=(service, media_repo, titler),
    ):
        async with user_simulation(main_file=str(_MAIN_FILE)) as user:
            await user.open("/")
            await user.should_see(kind=ui.input)
            user.find(kind=ui.input).type("Recommend a heist movie")
            user.find(kind=ui.input).trigger("keydown.enter")
            await user.should_see(content="Heat")

            await user.should_see(content="Heist thrillers with a twist")

    store = ConversationStore(str(_isolated_conversation_store))
    recent = store.list_recent()
    assert len(recent) == 1
    assert recent[0].title == "Heist thrillers with a twist"


@pytest.mark.anyio
async def test_clicking_recent_conversation_replays_its_transcript() -> None:
    service = make_service()
    media_repo = MagicMock(spec=QdrantMediaItems)
    titler = make_titler()
    titler.title.side_effect = ["Heist thrillers with a twist", "Comfort films"]
    service.chat_with_items_stream.side_effect = [
        make_answer("1. **Heat** (1995)\nA tense heist.", [make_item("tt1", "Heat")]),
        make_answer(
            "1. **Amelie** (2001)\nWarm and whimsical.", [make_item("tt2", "Amelie")]
        ),
    ]
    with patch.object(
        service_cache,
        "build_recommender_service",
        return_value=(service, media_repo, titler),
    ):
        async with user_simulation(main_file=str(_MAIN_FILE)) as user:
            await user.open("/")
            await user.should_see(kind=ui.input)

            user.find(kind=ui.input).type("Recommend a heist movie")
            user.find(kind=ui.input).trigger("keydown.enter")
            await user.should_see(content="Heat")

            user.find(content="New conversation").click()
            await user.should_not_see(content="Heat")

            user.find(kind=ui.input).type("Something cozy")
            user.find(kind=ui.input).trigger("keydown.enter")
            await user.should_see(content="Amelie")

            user.find(content="Heist thrillers with a twist").click()
            await user.should_see(content="Recommend a heist movie")
            await user.should_see(content="Heat")
            await user.should_not_see(content="Amelie")


@pytest.mark.anyio
async def test_viewing_recent_conversation_hides_the_text_input() -> None:
    """Resuming a Recent conversation isn't implemented (no LLM/RAG history
    is restored for it), so it's a read-only snapshot — the text input must
    disappear rather than silently starting a conversation the user didn't
    ask to start."""
    service = make_service()
    media_repo = MagicMock(spec=QdrantMediaItems)
    titler = make_titler("Heist thrillers with a twist")
    with patch.object(
        service_cache,
        "build_recommender_service",
        return_value=(service, media_repo, titler),
    ):
        async with user_simulation(main_file=str(_MAIN_FILE)) as user:
            await user.open("/")
            await user.should_see(kind=ui.input)

            user.find(kind=ui.input).type("Recommend a heist movie")
            user.find(kind=ui.input).trigger("keydown.enter")
            await user.should_see(content="Heat")
            await user.should_see(kind=ui.input)

            user.find(content="Heist thrillers with a twist").click()
            await user.should_see(content="Heat")
            await user.should_not_see(kind=ui.input)

            # "New conversation" gets the input back.
            user.find(content="New conversation").click()
            await user.should_not_see(content="Heat")
            await user.should_see(kind=ui.input)


@pytest.mark.anyio
async def test_new_conversation_does_not_persist_empty_conversation(
    _isolated_conversation_store: Path,
) -> None:
    service = make_service()
    built = (service, MagicMock(spec=QdrantMediaItems), make_titler())
    with patch.object(service_cache, "build_recommender_service", return_value=built):
        async with user_simulation(main_file=str(_MAIN_FILE)) as user:
            await user.open("/")
            await user.should_see(kind=ui.input)
            user.find(content="New conversation").click()
            # on_new_conversation runs as a scheduled background task, not
            # synchronously with the click — poll for it to finish before
            # tearing down the session, or the click may not have run yet.
            for _ in range(20):
                if service.reset_history.call_count >= 1:
                    break
                await asyncio.sleep(0.05)

    store = ConversationStore(str(_isolated_conversation_store))
    assert store.list_recent() == []


# --- Surprise me ---


@pytest.mark.anyio
async def test_surprise_me_replayed_from_recent_still_shows_its_picks(
    _isolated_conversation_store: Path,
) -> None:
    """Regression test: the diversity recommender's answer text is plain
    prose with no numbered sections, so replaying it through the same
    renderer used for regular chat turns silently dropped the items,
    leaving the Recent entry looking like it recommended nothing."""
    media_repo = MagicMock(spec=QdrantMediaItems)
    built = (make_service(), media_repo, make_titler("A change of pace"))
    diversity_service = make_diversity_service([make_item("tt9", "Paprika")])
    with (
        patch.object(service_cache, "build_recommender_service", return_value=built),
        patch.object(
            service_cache, "build_diversity_service", return_value=diversity_service
        ),
    ):
        async with user_simulation(main_file=str(_MAIN_FILE)) as user:
            await user.open("/")
            await user.should_see(kind=ui.input)

            user.find(content="Surprise me").click()
            await user.should_see(content="Paprika")

            user.find(content="New conversation").click()
            await user.should_not_see(content="Paprika")

            user.find(content="A change of pace").click()
            await user.should_see(content="Paprika")


@pytest.mark.anyio
async def test_surprise_me_hides_the_text_input_until_new_conversation() -> None:
    """The diversity recommender never joins the RAG chat history, so
    continuing to type after a Surprise-me turn hasn't been reasoned
    through — the text input is removed rather than silently mixing the
    two."""
    media_repo = MagicMock(spec=QdrantMediaItems)
    built = (make_service(), media_repo, make_titler())
    diversity_service = make_diversity_service([make_item("tt9", "Paprika")])
    with (
        patch.object(service_cache, "build_recommender_service", return_value=built),
        patch.object(
            service_cache, "build_diversity_service", return_value=diversity_service
        ),
    ):
        async with user_simulation(main_file=str(_MAIN_FILE)) as user:
            await user.open("/")
            await user.should_see(kind=ui.input)

            user.find(content="Surprise me").click()
            await user.should_see(content="Paprika")
            await user.should_not_see(kind=ui.input)

            user.find(content="New conversation").click()
            await user.should_not_see(content="Paprika")
            await user.should_see(kind=ui.input)


@pytest.mark.anyio
async def test_surprise_me_clicked_twice_starts_a_fresh_conversation_each_time(
    _isolated_conversation_store: Path,
) -> None:
    """Regression test: clicking "Surprise me" again while its own previous
    turn is still on screen used to append onto that same turn instead of
    starting over, so the transcript grew without bound and the Recent
    sidebar entry never got a sibling for the second pull."""
    media_repo = MagicMock(spec=QdrantMediaItems)
    built = (make_service(), media_repo, make_titler())
    diversity_service = make_diversity_service([make_item("tt9", "Paprika")])
    with (
        patch.object(service_cache, "build_recommender_service", return_value=built),
        patch.object(
            service_cache, "build_diversity_service", return_value=diversity_service
        ),
    ):
        async with user_simulation(main_file=str(_MAIN_FILE)) as user:
            await user.open("/")
            await user.should_see(kind=ui.input)

            user.find(kind=ui.button, content="Surprise me").click()
            await user.should_see(content="Paprika")
            await asyncio.sleep(0.2)

            user.find(kind=ui.button, content="Surprise me").click()
            await asyncio.sleep(0.2)
            await user.should_see(content="Paprika")

    store = ConversationStore(str(_isolated_conversation_store))
    recent = store.list_recent()
    assert len(recent) == 2
    assert all(len(conv.messages) == 2 for conv in recent)


@pytest.mark.anyio
async def test_new_conversation_clicked_mid_surprise_turn_leaves_input_usable(
    _isolated_conversation_store: Path,
) -> None:
    """Regression test: clicking "New conversation" while a still-running
    Surprise-me turn hasn't returned yet must reset the view immediately —
    an earlier fix instead blocked the click until the stale turn finished
    (silently doing nothing for as long as that took, which read as "New
    conversation is broken"), and before that, an even earlier version let
    the stale turn's completion race the reset and re-lock/hide the input
    right after it had been shown. Neither the input lock nor a Recent
    entry should end up reflecting a turn the user already left."""
    media_repo = MagicMock(spec=QdrantMediaItems)
    built = (make_service(), media_repo, make_titler())
    diversity_service = MagicMock(spec=DiversityRecommendationService)

    def slow_recommend() -> list[MediaItem]:
        time.sleep(0.3)
        return [make_item("tt9", "Paprika")]

    diversity_service.recommend.side_effect = slow_recommend
    with (
        patch.object(service_cache, "build_recommender_service", return_value=built),
        patch.object(
            service_cache, "build_diversity_service", return_value=diversity_service
        ),
    ):
        async with user_simulation(main_file=str(_MAIN_FILE)) as user:
            await user.open("/")
            await user.should_see(kind=ui.input)

            # Fire the slow Surprise-me turn but don't wait for it — click
            # "New conversation" while it's still running its io_bound call.
            user.find(kind=ui.button, content="Surprise me").click()
            await asyncio.sleep(0.05)
            user.find(content="New conversation").click()

            # The reset must take effect right away, not after the stale
            # turn eventually finishes.
            await asyncio.sleep(0.1)
            await user.should_see(kind=ui.input)
            inputs = user.find(kind=ui.input).elements
            assert all(el.enabled for el in inputs)

            # Let the stale Surprise-me turn resolve in the background and
            # confirm it had no lingering effect on the now-current view.
            await asyncio.sleep(0.5)
            await user.should_not_see(content="Paprika")
            await user.should_see(kind=ui.input)
            inputs = user.find(kind=ui.input).elements
            assert all(el.enabled for el in inputs)

    store = ConversationStore(str(_isolated_conversation_store))
    assert store.list_recent() == []
