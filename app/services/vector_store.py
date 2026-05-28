import time

from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, FieldCondition, Filter, MatchAny, MatchValue, VectorParams

VECTOR_SIZE = 3072
BATCH_SIZE = 5
BASE_RETRY_DELAY = 10
MAX_RETRY_DELAY = 120
INTER_BATCH_DELAY = 4


class VectorStoreService:
    def __init__(
        self,
        path: str,
        collection_name: str,
        embeddings: GoogleGenerativeAIEmbeddings | None = None,
    ) -> None:
        self._embeddings = embeddings
        self._collection_name = collection_name
        self._client = QdrantClient(path=path)

    def load_or_build(self, documents: list[Document]) -> QdrantVectorStore:
        if self._client.collection_exists(self._collection_name):
            print("Loading existing vector store...")
            self._store = QdrantVectorStore(
                client=self._client,
                collection_name=self._collection_name,
                embedding=self._embeddings,
            )
            self._add_missing_synopsis_docs(documents)
        else:
            self._store = self._build(documents)
        return self._store

    def _get_synopsis_imdb_ids(self) -> set[str]:
        results, _ = self._client.scroll(
            collection_name=self._collection_name,
            scroll_filter=Filter(
                must=[FieldCondition(key="metadata.embedding_type", match=MatchValue(value="synopsis"))]
            ),
            limit=10_000,
            with_payload=True,
            with_vectors=False,
        )
        return {pt.payload["metadata"]["imdb_id"] for pt in results if pt.payload}

    def _add_missing_synopsis_docs(self, documents: list[Document]) -> None:
        existing_ids = self._get_synopsis_imdb_ids()
        missing = [doc for doc in documents if doc.metadata.get("imdb_id") not in existing_ids]
        if not missing:
            return
        print(f"Adding {len(missing)} new synopsis embedding(s)...")
        for i in range(0, len(missing), BATCH_SIZE):
            batch = missing[i : i + BATCH_SIZE]
            self._add_with_retry(self._store, batch)
            if i + BATCH_SIZE < len(missing):
                time.sleep(INTER_BATCH_DELAY)

    def delete_by_imdb_ids(self, imdb_ids: set[str]) -> None:
        if not imdb_ids:
            return
        self._client.delete(
            collection_name=self._collection_name,
            points_selector=Filter(must=[FieldCondition(key="metadata.imdb_id", match=MatchAny(any=list(imdb_ids)))]),
        )

    def _build(self, documents: list[Document]) -> QdrantVectorStore:
        print(f"Building vector store from {len(documents)} items...")
        self._client.create_collection(
            collection_name=self._collection_name,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        )
        store = QdrantVectorStore(
            client=self._client,
            collection_name=self._collection_name,
            embedding=self._embeddings,
        )
        for i in range(0, len(documents), BATCH_SIZE):
            batch = documents[i : i + BATCH_SIZE]
            self._add_with_retry(store, batch)
            print(f"  Embedded {min(i + BATCH_SIZE, len(documents))}/{len(documents)}")
            if i + BATCH_SIZE < len(documents):
                time.sleep(INTER_BATCH_DELAY)
        print("Done.")
        return store

    @property
    def client(self) -> QdrantClient:
        return self._client

    @property
    def store(self) -> QdrantVectorStore:
        return self._store

    def add_documents_with_retry(self, documents: list[Document]) -> None:
        self._add_with_retry(self._store, documents)

    def close(self) -> None:
        self._client.close()

    def _add_with_retry(self, store: QdrantVectorStore, batch: list[Document]) -> None:
        delay = BASE_RETRY_DELAY
        while True:
            try:
                store.add_documents(batch)
                return
            except Exception as e:
                if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                    print(f"  Rate limited, retrying in {delay}s...")
                    time.sleep(delay)
                    delay = min(delay * 2, MAX_RETRY_DELAY)
                else:
                    raise
