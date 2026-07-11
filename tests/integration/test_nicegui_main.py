import asyncio
from collections.abc import AsyncIterator
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from nicegui import ui
from nicegui.testing import user_simulation

import app.config as config
import nicegui_app.service_cache as service_cache
from app.domain.ports import ConversationTitler
from app.models.media_item import MediaItem
from app.repositories.conversation_store import ConversationStore
from app.repositories.qdrant_media_items import QdrantMediaItems
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


@pytest.fixture(autouse=True)
def _isolated_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    """`service_cache._cache`/`_lock` are process-lifetime globals shared by
    every page load — reset them per test so tests can't leak cached
    (mocked) services into one another."""
    monkeypatch.setattr(service_cache, "_cache", {})
    monkeypatch.setattr(service_cache, "_lock", asyncio.Lock())


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
async def test_sending_message_while_viewing_recent_starts_new_conversation() -> None:
    service = make_service()
    media_repo = MagicMock(spec=QdrantMediaItems)
    titler = make_titler("Heist thrillers with a twist")
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

            # View the just-created conversation from Recent (rather than
            # continuing the live one), then send a new message — this
            # should start a brand-new conversation, not append to the
            # snapshot.
            user.find(content="Heist thrillers with a twist").click()
            await user.should_see(content="Heat")

            user.find(kind=ui.input).type("Something cozy")
            user.find(kind=ui.input).trigger("keydown.enter")
            await user.should_see(content="Amelie")
            await user.should_not_see(content="Recommend a heist movie")
            await user.should_not_see(content="Heat")

    service.reset_history.assert_called_once()


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
