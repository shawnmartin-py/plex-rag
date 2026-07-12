from app.domain.ports import WatchedEmbedding


class QdrantWatchHistory:
    """Implements `WatchHistoryLookup`. Thin in-memory wrapper over points already
    loaded once at startup (`load_watch_history_points`) — mirrors QdrantMediaItems'
    shape, same reasoning: the corpus is small enough to hold in memory, and this
    avoids a live Qdrant round trip on every recommend() call."""

    def __init__(self, points: list[WatchedEmbedding]) -> None:
        self._points = points

    def recent(self) -> list[WatchedEmbedding]:
        return self._points
