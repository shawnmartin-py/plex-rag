import asyncio
import time
from unittest.mock import MagicMock, patch

import pytest
from nicegui import run

import nicegui_app.service_cache as service_cache
from app.domain.ports import ConversationTitler
from app.repositories.qdrant_media_items import QdrantMediaItems
from app.services.diversity_recommendation import DiversityRecommendationService
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
    monkeypatch.setattr(service_cache, "_diversity_service", None)
    monkeypatch.setattr(service_cache, "_diversity_loaded", False)
    monkeypatch.setattr(service_cache, "_diversity_lock", asyncio.Lock())


def make_built_service() -> _BuiltService:
    service = MagicMock(spec=ConversationalRecommendationService)
    media_repo = MagicMock(spec=QdrantMediaItems)
    titler = MagicMock(spec=ConversationTitler)
    return (service, media_repo, titler)


@pytest.mark.anyio
async def test_get_service_returns_same_instance_on_repeated_calls() -> None:
    built = make_built_service()
    with patch.object(
        service_cache, "build_recommender_service", return_value=built
    ) as mock_build:
        first = await service_cache.get_service(True)
        second = await service_cache.get_service(True)

    assert first is second
    mock_build.assert_called_once_with(spoiler_free=True)


@pytest.mark.anyio
async def test_get_service_does_not_rebuild_after_first_call() -> None:
    with patch.object(
        service_cache, "build_recommender_service", return_value=make_built_service()
    ) as mock_build:
        await service_cache.get_service(False)
        await service_cache.get_service(False)
        await service_cache.get_service(False)

    assert mock_build.call_count == 1


@pytest.mark.anyio
async def test_get_service_different_keys_get_independent_cache_entries() -> None:
    built_spoiler_free = make_built_service()
    built_normal = make_built_service()

    def fake_build(spoiler_free: bool) -> _BuiltService:
        return built_spoiler_free if spoiler_free else built_normal

    with patch.object(
        service_cache, "build_recommender_service", side_effect=fake_build
    ) as mock_build:
        result_true = await service_cache.get_service(True)
        result_false = await service_cache.get_service(False)

    assert result_true is built_spoiler_free
    assert result_false is built_normal
    assert result_true is not result_false
    assert mock_build.call_count == 2


@pytest.mark.anyio
async def test_get_service_calls_build_via_io_bound() -> None:
    built = make_built_service()
    with (
        patch.object(
            service_cache, "build_recommender_service", return_value=built
        ) as mock_build,
        patch.object(run, "io_bound", wraps=run.io_bound) as mock_io_bound,
    ):
        result = await service_cache.get_service(True)

    assert result is built
    mock_io_bound.assert_called_once_with(mock_build, spoiler_free=True)


@pytest.mark.anyio
async def test_get_service_raises_when_io_bound_returns_none() -> None:
    async def fake_io_bound(callback: object, *args: object, **kwargs: object) -> None:
        return None

    with patch.object(run, "io_bound", fake_io_bound):
        with pytest.raises(RuntimeError, match="cancelled"):
            await service_cache.get_service(True)


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
            service_cache.get_service(True), service_cache.get_service(True)
        )

    assert call_count == 1
    assert first is second


# --- get_diversity_service ---


@pytest.mark.anyio
async def test_get_diversity_service_returns_same_instance_on_repeated_calls() -> None:
    built = MagicMock(spec=DiversityRecommendationService)
    with patch.object(service_cache, "build_diversity_service", return_value=built):
        first = await service_cache.get_diversity_service()
        second = await service_cache.get_diversity_service()

    assert first is second is built


@pytest.mark.anyio
async def test_get_diversity_service_caches_a_legitimate_none_result() -> None:
    """The whole reason for the boxed-tuple trick: a build that legitimately
    returns None (watch_history collection missing) must be cached as None, not
    treated as a cancelled io_bound call and re-attempted every call."""
    with patch.object(
        service_cache, "build_diversity_service", return_value=None
    ) as mock_build:
        first = await service_cache.get_diversity_service()
        second = await service_cache.get_diversity_service()

    assert first is None
    assert second is None
    mock_build.assert_called_once()


@pytest.mark.anyio
async def test_get_diversity_service_does_not_rebuild_after_first_call() -> None:
    with patch.object(
        service_cache,
        "build_diversity_service",
        return_value=MagicMock(spec=DiversityRecommendationService),
    ) as mock_build:
        await service_cache.get_diversity_service()
        await service_cache.get_diversity_service()
        await service_cache.get_diversity_service()

    assert mock_build.call_count == 1
