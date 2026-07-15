from datetime import datetime

from app.domain.ports import CandidateEmbedding, WatchedEmbedding
from app.repositories.candidate_pool import QdrantCandidatePool
from app.repositories.watch_history import QdrantWatchHistory


def test_qdrant_watch_history_returns_stored_points() -> None:
    points = [
        WatchedEmbedding(
            tmdb_id="tt1", vector=[0.1], last_viewed_at=datetime(2026, 1, 1)
        )
    ]
    repo = QdrantWatchHistory(points)
    assert repo.recent() == points


def test_qdrant_candidate_pool_returns_stored_candidates() -> None:
    candidates = [CandidateEmbedding(tmdb_id="tt1", vector=[0.1], imdb_rating=7.5)]
    repo = QdrantCandidatePool(candidates)
    assert repo.all() == candidates
