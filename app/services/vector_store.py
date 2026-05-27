import time

from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

VECTOR_SIZE = 3072
BATCH_SIZE = 5
BASE_RETRY_DELAY = 10
MAX_RETRY_DELAY = 120
INTER_BATCH_DELAY = 4


class VectorStoreService:
    def __init__(
        self,
        embeddings: GoogleGenerativeAIEmbeddings,
        path: str,
        collection_name: str,
    ) -> None:
        self._embeddings = embeddings
        self._collection_name = collection_name
        self._client = QdrantClient(path=path)

    def load_or_build(self, documents: list[Document]) -> QdrantVectorStore:
        if self._client.collection_exists(self._collection_name):
            print("Loading existing vector store...")
            return QdrantVectorStore(
                client=self._client,
                collection_name=self._collection_name,
                embedding=self._embeddings,
            )
        return self._build(documents)

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
