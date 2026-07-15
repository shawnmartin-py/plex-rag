from app.domain.diversity import DiversityRecommender
from app.domain.ports import MediaItemLookup
from app.models.media_item import MediaItem


class DiversityRecommendationService:
    """Session-level wrapper around the stateless `DiversityRecommender`, the
    diversity-mode counterpart to `ConversationalRecommendationService` — tracks
    which tmdb_ids have already been shown this session (so repeated "show me
    another" calls don't repeat a pick) and resolves tmdb_ids to full `MediaItem`s
    for rendering. `NoWatchHistoryError` is not caught here; it propagates to the
    caller (CLI/web UI), same as `QdrantUnavailableError` does elsewhere — this
    layer doesn't decide how to present errors."""

    def __init__(
        self, recommender: DiversityRecommender, media_repo: MediaItemLookup
    ) -> None:
        self._recommender = recommender
        self._media_repo = media_repo
        self._recently_shown: set[str] = set()

    def recommend(self) -> list[MediaItem]:
        tmdb_ids = self._recommender.recommend(exclude=frozenset(self._recently_shown))
        self._recently_shown.update(tmdb_ids)
        items = [self._media_repo.get_by_id(tmdb_id) for tmdb_id in tmdb_ids]
        return [item for item in items if item is not None]

    def reset(self) -> None:
        self._recently_shown = set()
