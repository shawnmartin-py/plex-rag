from app.domain.ports import CandidateEmbedding


class QdrantCandidatePool:
    """Implements `CandidatePool`. Thin in-memory wrapper over vectors already
    loaded once at startup (`load_synopsis_vectors`) — mirrors QdrantMediaItems'
    shape, same reasoning: the corpus is small enough to hold in memory, and this
    avoids a live Qdrant round trip on every recommend() call."""

    def __init__(self, candidates: list[CandidateEmbedding]) -> None:
        self._candidates = candidates

    def all(self) -> list[CandidateEmbedding]:
        return self._candidates
