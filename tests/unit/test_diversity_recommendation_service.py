from datetime import datetime

import pytest

from app.domain.diversity import DiversityRecommender, NoWatchHistoryError
from app.domain.ports import CandidateEmbedding, WatchedEmbedding
from app.models.media_item import MediaItem
from app.services.diversity_recommendation import DiversityRecommendationService


class _FakeWatchHistory:
    def __init__(self, items: list[WatchedEmbedding]) -> None:
        self._items = items

    def recent(self) -> list[WatchedEmbedding]:
        return self._items


class _FakeCandidatePool:
    def __init__(self, items: list[CandidateEmbedding]) -> None:
        self._items = items

    def all(self) -> list[CandidateEmbedding]:
        return self._items


class _FakeMediaRepo:
    def __init__(self, items: dict[str, MediaItem]) -> None:
        self._items = items

    def get_by_id(self, imdb_id: str) -> MediaItem | None:
        return self._items.get(imdb_id)


def _media_item(imdb_id: str) -> MediaItem:
    return MediaItem(
        imdb_id=imdb_id,
        type="movie",
        title=imdb_id,
        year=2020,
        imdb_rating=7.0,
        content_rating="PG-13",
        genres=["Drama"],
    )


def _watched(imdb_id: str) -> WatchedEmbedding:
    return WatchedEmbedding(
        imdb_id=imdb_id, vector=[1.0, 0.0], last_viewed_at=datetime(2026, 7, 1)
    )


def _candidate(imdb_id: str, vector: list[float]) -> CandidateEmbedding:
    return CandidateEmbedding(imdb_id=imdb_id, vector=vector, imdb_rating=7.0)


def test_recommend_resolves_imdb_ids_to_media_items() -> None:
    recommender = DiversityRecommender(
        _FakeWatchHistory([_watched("watched1")]),
        _FakeCandidatePool([_candidate("tt1", [0.0, 1.0])]),
        k=1,
        band_low_percentile=0.0,
        band_high_percentile=1.0,
    )
    service = DiversityRecommendationService(
        recommender, _FakeMediaRepo({"tt1": _media_item("tt1")})
    )

    items = service.recommend()

    assert len(items) == 1
    assert items[0].imdb_id == "tt1"


def test_recommend_drops_ids_with_no_media_item() -> None:
    recommender = DiversityRecommender(
        _FakeWatchHistory([_watched("watched1")]),
        _FakeCandidatePool([_candidate("tt1", [0.0, 1.0])]),
        k=1,
        band_low_percentile=0.0,
        band_high_percentile=1.0,
    )
    service = DiversityRecommendationService(recommender, _FakeMediaRepo({}))

    assert service.recommend() == []


def test_recommend_does_not_repeat_within_a_session() -> None:
    recommender = DiversityRecommender(
        _FakeWatchHistory([_watched("watched1")]),
        _FakeCandidatePool(
            [_candidate("tt1", [0.0, 1.0]), _candidate("tt2", [0.0, -1.0])]
        ),
        k=1,
        band_low_percentile=0.0,
        band_high_percentile=1.0,
    )
    media_repo = _FakeMediaRepo({"tt1": _media_item("tt1"), "tt2": _media_item("tt2")})
    service = DiversityRecommendationService(recommender, media_repo)

    first = service.recommend()
    second = service.recommend()

    assert {i.imdb_id for i in first} != {i.imdb_id for i in second}


def test_reset_clears_the_recently_shown_set() -> None:
    recommender = DiversityRecommender(
        _FakeWatchHistory([_watched("watched1")]),
        _FakeCandidatePool([_candidate("tt1", [0.0, 1.0])]),
        k=1,
        band_low_percentile=0.0,
        band_high_percentile=1.0,
    )
    media_repo = _FakeMediaRepo({"tt1": _media_item("tt1")})
    service = DiversityRecommendationService(recommender, media_repo)

    service.recommend()
    assert service.recommend() == []  # only candidate already shown
    service.reset()
    assert len(service.recommend()) == 1


def test_no_watch_history_error_propagates() -> None:
    recommender = DiversityRecommender(_FakeWatchHistory([]), _FakeCandidatePool([]))
    service = DiversityRecommendationService(recommender, _FakeMediaRepo({}))

    with pytest.raises(NoWatchHistoryError):
        service.recommend()
