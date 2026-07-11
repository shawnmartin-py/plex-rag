import asyncio
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from nicegui import ui
from nicegui.testing import user_simulation

import nicegui_app.service_cache as service_cache
from app.models.media_item import MediaItem
from app.repositories.qdrant_media_items import QdrantMediaItems
from app.services.recommendation import ConversationalRecommendationService

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


def make_service(answer: str = "1. **Heat** (1995)\nA tense heist.") -> MagicMock:
    service = MagicMock(spec=ConversationalRecommendationService)
    service.chat_with_items.return_value = (answer, [make_item("tt1", "Heat")])
    return service


@pytest.fixture(autouse=True)
def _isolated_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    """`service_cache._cache`/`_lock` are process-lifetime globals shared by
    every page load — reset them per test so tests can't leak cached
    (mocked) services into one another."""
    monkeypatch.setattr(service_cache, "_cache", {})
    monkeypatch.setattr(service_cache, "_lock", asyncio.Lock())


@pytest.mark.anyio
async def test_page_loads_and_enables_chat_input() -> None:
    media_repo = MagicMock(spec=QdrantMediaItems)
    built = (make_service(), media_repo)
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
        service_cache, "build_recommender_service", return_value=(service, media_repo)
    ):
        async with user_simulation(main_file=str(_MAIN_FILE)) as user:
            await user.open("/")
            await user.should_see(kind=ui.input)
            user.find(kind=ui.input).type("Recommend a heist movie")
            user.find(kind=ui.input).trigger("keydown.enter")

            await user.should_see(content="Recommend a heist movie")
            await user.should_see(content="Heat")

    service.chat_with_items.assert_called_once_with(
        "Recommend a heist movie", media_repo
    )


@pytest.mark.anyio
async def test_new_conversation_clears_transcript_and_resets_history() -> None:
    service = make_service()
    media_repo = MagicMock(spec=QdrantMediaItems)
    with patch.object(
        service_cache, "build_recommender_service", return_value=(service, media_repo)
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

    def fake_build(spoiler_free: bool) -> tuple[MagicMock, MagicMock]:
        return (spoiler_free_service if spoiler_free else normal_service, media_repo)

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
