from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue, VectorParams

VECTOR_SIZE = 3072  # gemini-embedding-001 — see docs/vector-store-contract.md


class QdrantUnavailableError(RuntimeError):
    pass


def connect_vector_store(
    url: str, collection_name: str, embeddings: Embeddings
) -> QdrantVectorStore:
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
