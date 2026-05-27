import tempfile
from collections.abc import Iterator

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from app.services.vector_store import VectorStoreService


class StubLLM(BaseChatModel):
    """Real BaseChatModel subclass so LCEL chains compose and invoke correctly."""

    responses: list[str] = []
    _index: int = 0

    @property
    def _llm_type(self) -> str:
        return "stub"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        response = self.responses[self._index % len(self.responses)]
        object.__setattr__(self, "_index", self._index + 1)
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=response))])


class StubEmbeddings(Embeddings):
    """Real Embeddings subclass returning fixed-size vectors."""

    dims: int = 3072

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[0.1] * self.dims for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        return [0.1] * self.dims


TEST_DOCS = [
    Document(
        page_content=(
            "Title: Parasite\nYear: 2019\nIMDb Rating: 8.5\n"
            "Genres: Drama, Thriller\n"
            "Synopsis: A poor Korean family schemes their way into the lives of a wealthy family, "
            "leading to an explosive confrontation about class and greed."
        ),
        metadata={"imdb_id": "tt6751668", "title": "Parasite", "year": 2019},
    ),
    Document(
        page_content=(
            "Title: Oldboy\nYear: 2003\nIMDb Rating: 8.1\n"
            "Genres: Action, Drama, Mystery\n"
            "Synopsis: A man is imprisoned for 15 years without explanation, then released and given "
            "five days to find out why."
        ),
        metadata={"imdb_id": "tt0364569", "title": "Oldboy", "year": 2003},
    ),
    Document(
        page_content=(
            "Title: The Handmaiden\nYear: 2016\nIMDb Rating: 8.1\n"
            "Genres: Drama, Mystery, Romance\n"
            "Synopsis: A woman is hired as a handmaiden to a Japanese heiress, but is secretly "
            "involved in a plot to defraud her."
        ),
        metadata={"imdb_id": "tt4016934", "title": "The Handmaiden", "year": 2016},
    ),
]


@pytest.fixture(scope="module")
def stub_embeddings() -> StubEmbeddings:
    return StubEmbeddings()


@pytest.fixture(scope="module")
def qdrant_store(stub_embeddings) -> Iterator:
    with tempfile.TemporaryDirectory() as tmpdir:
        service = VectorStoreService(
            embeddings=stub_embeddings,
            path=tmpdir,
            collection_name="test_movies",
        )
        store = service.load_or_build(TEST_DOCS)
        yield store
