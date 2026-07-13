from datetime import datetime

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue, VectorParams

from app.domain.ports import CandidateEmbedding, WatchedEmbedding

VECTOR_SIZE = 3072  # gemini-embedding-001 — see docs/vector-store-contract.md


class QdrantUnavailableError(RuntimeError):
    pass


def _connect_and_validate(url: str, collection_name: str) -> QdrantClient:
    """Connect to the Qdrant server plex-ingest owns and confirm it's usable. Per
    docs/vector-store-contract.md, the recommender never starts/manages this
    container or writes to it — only plex-ingest does — so this fails fast with a
    clear message instead of trying to build/repair anything itself."""
    try:
        client = QdrantClient(url=url)
        exists = client.collection_exists(collection_name)
    except Exception as e:
        raise QdrantUnavailableError(
            f"Could not reach Qdrant at {url}. Is the plex-ingest Docker "
            "container running?"
        ) from e
    if not exists:
        raise QdrantUnavailableError(
            f"Qdrant collection '{collection_name}' does not exist at {url}. "
            "Run the plex-ingest pipeline first."
        )
    vectors_config = client.get_collection(collection_name).config.params.vectors
    if not isinstance(vectors_config, VectorParams):
        raise QdrantUnavailableError(
            f"Qdrant collection '{collection_name}' uses named/multi-vector "
            "config, expected a single unnamed vector (gemini-embedding-001) "
            "— schema mismatch between plex-rag and plex-ingest."
        )
    actual_size = vectors_config.size
    if actual_size != VECTOR_SIZE:
        raise QdrantUnavailableError(
            f"Qdrant collection '{collection_name}' has vector size "
            f"{actual_size}, expected {VECTOR_SIZE} (gemini-embedding-001) — "
            "embedding model mismatch between plex-rag and plex-ingest."
        )
    return client


def ensure_qdrant_reachable(url: str, collection_name: str) -> None:
    """Same fail-fast validation as `connect_vector_store`, minus the embeddings
    wiring — for callers (the NiceGUI server's startup, before `ui.run()`) that
    just want to surface a clear `QdrantUnavailableError` immediately rather than
    waiting for the first page load to build the full recommender service."""
    _connect_and_validate(url, collection_name)


def connect_vector_store(
    url: str, collection_name: str, embeddings: Embeddings
) -> QdrantVectorStore:
    client = _connect_and_validate(url, collection_name)
    return QdrantVectorStore(
        client=client, collection_name=collection_name, embedding=embeddings
    )


def load_synopsis_documents(
    vector_store: QdrantVectorStore, collection_name: str
) -> list[Document]:
    """Every embedding_type=synopsis point, used to build the LLMKnowledgeRetriever's
    movie-title list and per-film MediaItem lookups without any direct DB dependency
    on plex-ingest's internal storage (see docs/vector-store-contract.md)."""
    points, _ = vector_store.client.scroll(
        collection_name=collection_name,
        scroll_filter=Filter(
            must=[
                FieldCondition(
                    key="metadata.embedding_type", match=MatchValue(value="synopsis")
                )
            ]
        ),
        limit=10_000,
        with_payload=True,
        with_vectors=False,
    )
    return [
        Document(page_content=p.payload["page_content"], metadata=p.payload["metadata"])
        for p in points
        if p.payload is not None
    ]


def _as_flat_vector(vector: object, imdb_id: str) -> list[float]:
    """Both `media_items` and `watch_history` use a single unnamed vector per point
    (see docs/vector-store-contract.md) — never Qdrant's named/multi-vector shape.
    Narrows qdrant-client's `VectorStruct` union to `list[float]` for mypy and fails
    fast on an actual schema mismatch, rather than silently dropping the point."""
    if not isinstance(vector, list) or not all(
        isinstance(v, float | int) for v in vector
    ):
        msg = (
            f"Expected a flat vector for imdb_id={imdb_id!r}, got {type(vector)!r} — "
            "named/multi-vector collection? See docs/vector-store-contract.md."
        )
        raise TypeError(msg)
    return [float(v) for v in vector]


def load_synopsis_vectors(
    vector_store: QdrantVectorStore, collection_name: str
) -> list[CandidateEmbedding]:
    """Every embedding_type=synopsis point's vector, keyed by imdb_id, for the
    diversity recommender's candidate pool. `media_items` already only contains
    unwatched movies (plex-ingest's own scope), so no separate "exclude watched"
    filter belongs here — see docs/pipeline-design.md."""
    points, _ = vector_store.client.scroll(
        collection_name=collection_name,
        scroll_filter=Filter(
            must=[
                FieldCondition(
                    key="metadata.embedding_type", match=MatchValue(value="synopsis")
                )
            ]
        ),
        limit=10_000,
        with_payload=True,
        with_vectors=True,
    )
    result = []
    for p in points:
        if p.payload is None:
            continue
        imdb_id = p.payload["metadata"]["imdb_id"]
        result.append(
            CandidateEmbedding(
                imdb_id=imdb_id,
                vector=_as_flat_vector(p.vector, imdb_id),
                imdb_rating=p.payload["metadata"].get("imdb_rating"),
            )
        )
    return result


def load_watch_history_points(
    vector_store: QdrantVectorStore, collection_name: str
) -> list[WatchedEmbedding]:
    """Every point in the `watch_history` collection — see
    docs/vector-store-contract.md's `watch_history` section. Already windowed and
    deduped by plex-ingest's `watch_history_qdrant_collection` asset; this just
    reads whatever's there."""
    points, _ = vector_store.client.scroll(
        collection_name=collection_name,
        limit=10_000,
        with_payload=True,
        with_vectors=True,
    )
    result = []
    for p in points:
        if p.payload is None:
            continue
        imdb_id = p.payload["metadata"]["imdb_id"]
        result.append(
            WatchedEmbedding(
                imdb_id=imdb_id,
                vector=_as_flat_vector(p.vector, imdb_id),
                last_viewed_at=datetime.fromisoformat(
                    p.payload["metadata"]["last_viewed_at"]
                ),
            )
        )
    return result
