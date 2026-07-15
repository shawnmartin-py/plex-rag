import asyncio
import time
from typing import cast
from unittest.mock import MagicMock, patch

import pytest

import api_app.service_cache as service_cache
from app.domain.ports import ConversationTitler
from app.repositories.qdrant_media_items import QdrantMediaItems
from app.services.recommendation import ConversationalRecommendationService

_BuiltService = tuple[
    ConversationalRecommendationService, QdrantMediaItems, ConversationTitler
]


@pytest.fixture(autouse=True)
def _isolated_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    """The module-level `_cache`/`_lock` are process-lifetime globals — give
    each test a fresh copy so calls in one test can't leak into another."""
    monkeypatch.setattr(service_cache, "_cache", {})
    monkeypatch.setattr(service_cache, "_lock", asyncio.Lock())


def make_built_service() -> _BuiltService:
    service = MagicMock(spec=ConversationalRecommendationService)
    media_repo = MagicMock(spec=QdrantMediaItems)
    titler = MagicMock(spec=ConversationTitler)
    return (service, media_repo, titler)


@pytest.mark.anyio
async def test_get_service_returns_same_instance_for_same_key() -> None:
    built = make_built_service()
    with patch.object(
        service_cache, "build_recommender_service", return_value=built
    ) as mock_build:
        first = await service_cache.get_service("session-a", True)
        second = await service_cache.get_service("session-a", True)

    assert first is second
    mock_build.assert_called_once_with(spoiler_free=True)


@pytest.mark.anyio
async def test_get_service_different_sessions_get_independent_cache_entries() -> None:
    built_a = make_built_service()
    built_b = make_built_service()

    calls: list[bool] = []

    def side_effect(spoiler_free: bool) -> _BuiltService:
        calls.append(spoiler_free)
        return built_a if len(calls) == 1 else built_b

    with patch.object(
        service_cache, "build_recommender_service", side_effect=side_effect
    ) as mock_build:
        result_a = await service_cache.get_service("session-a", False)
        result_b = await service_cache.get_service("session-b", False)

    assert result_a is built_a
    assert result_b is built_b
    assert result_a is not result_b
    assert mock_build.call_count == 2


@pytest.mark.anyio
async def test_get_service_same_session_different_spoiler_flag_gets_own_entry() -> None:
    built_normal = make_built_service()
    built_spoiler_free = make_built_service()

    def fake_build(spoiler_free: bool) -> _BuiltService:
        return built_spoiler_free if spoiler_free else built_normal

    with patch.object(
        service_cache, "build_recommender_service", side_effect=fake_build
    ):
        normal = await service_cache.get_service("session-a", False)
        spoiler_free = await service_cache.get_service("session-a", True)

    assert normal is built_normal
    assert spoiler_free is built_spoiler_free
    assert normal is not spoiler_free


@pytest.mark.anyio
async def test_get_service_does_not_rebuild_after_first_call() -> None:
    with patch.object(
        service_cache, "build_recommender_service", return_value=make_built_service()
    ) as mock_build:
        await service_cache.get_service("session-a", False)
        await service_cache.get_service("session-a", False)
        await service_cache.get_service("session-a", False)

    assert mock_build.call_count == 1


@pytest.mark.anyio
async def test_get_service_concurrent_calls_build_only_once() -> None:
    call_count = 0

    def slow_build(spoiler_free: bool) -> _BuiltService:
        nonlocal call_count
        call_count += 1
        time.sleep(0.05)
        return make_built_service()

    with patch.object(
        service_cache, "build_recommender_service", side_effect=slow_build
    ):
        first, second = await asyncio.gather(
            service_cache.get_service("session-a", False),
            service_cache.get_service("session-a", False),
        )

    assert call_count == 1
    assert first is second


# --- reset_session ---


@pytest.mark.anyio
async def test_reset_session_resets_history_on_matching_cached_services() -> None:
    with patch.object(
        service_cache,
        "build_recommender_service",
        side_effect=lambda **_: make_built_service(),
    ):
        service_normal, _, _ = await service_cache.get_service("session-a", False)
        service_spoiler_free, _, _ = await service_cache.get_service("session-a", True)
        other_session_service, _, _ = await service_cache.get_service(
            "session-b", False
        )

    found = service_cache.reset_session("session-a")

    assert found is True
    # `get_service` is typed to return the abstract ConversationalRecommendationService,
    # so mypy sees a plain `reset_history() -> None` method here despite these actually
    # being MagicMocks — cast to make the assert_* calls type-check.
    cast(MagicMock, service_normal.reset_history).assert_called_once()
    cast(MagicMock, service_spoiler_free.reset_history).assert_called_once()
    cast(MagicMock, other_session_service.reset_history).assert_not_called()


@pytest.mark.anyio
async def test_reset_session_returns_false_when_nothing_cached() -> None:
    assert service_cache.reset_session("unknown-session") is False
