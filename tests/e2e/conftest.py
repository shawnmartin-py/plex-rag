from typing import Any

import pytest
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable, RunnableLambda
from langchain_qdrant import QdrantVectorStore
from pydantic import BaseModel


class StubLLM(BaseChatModel):
    """Real BaseChatModel subclass so LCEL chains compose and invoke correctly."""

    responses: list[str] = []
    _index: int = 0

    @property
    def _llm_type(self) -> str:
        return "stub"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        response = self.responses[self._index % len(self.responses)]
        object.__setattr__(self, "_index", self._index + 1)
        return ChatResult(
            generations=[ChatGeneration(message=AIMessage(content=response))]
        )

    def with_structured_output(
        self, schema: dict[str, Any] | type, *, include_raw: bool = False, **kwargs: Any
    ) -> Runnable[Any, dict[str, Any] | BaseModel]:
        """Stand-in for provider structured-output support: parses the stubbed
        response's raw text as JSON matching `schema`, so responses configured on
        this stub must be JSON matching the target schema rather than free text.
        Only the Pydantic-class form of `schema` is supported — sufficient for
        this codebase's usage."""
        assert isinstance(schema, type) and issubclass(schema, BaseModel)
        pydantic_schema = schema

        def _parse(message: AIMessage) -> BaseModel:
            return pydantic_schema.model_validate_json(message.text)

        return self | RunnableLambda(_parse)


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
            "Synopsis: A poor Korean family schemes their way into the lives of a "
            "wealthy family, "
            "leading to an explosive confrontation about class and greed."
        ),
        metadata={
            "tmdb_id": "496243",
            "imdb_id": "tt6751668",
            "title": "Parasite",
            "year": 2019,
            "embedding_type": "synopsis",
        },
    ),
    Document(
        page_content=(
            "Title: Oldboy\nYear: 2003\nIMDb Rating: 8.1\n"
            "Genres: Action, Drama, Mystery\n"
            "Synopsis: A man is imprisoned for 15 years without explanation, then "
            "released and given "
            "five days to find out why."
        ),
        metadata={
            "tmdb_id": "670",
            "imdb_id": "tt0364569",
            "title": "Oldboy",
            "year": 2003,
            "embedding_type": "synopsis",
        },
    ),
    Document(
        page_content=(
            "Title: The Handmaiden\nYear: 2016\nIMDb Rating: 8.1\n"
            "Genres: Drama, Mystery, Romance\n"
            "Synopsis: A woman is hired as a handmaiden to a Japanese heiress, but is "
            "secretly "
            "involved in a plot to defraud her."
        ),
        metadata={
            "tmdb_id": "290098",
            "imdb_id": "tt4016934",
            "title": "The Handmaiden",
            "year": 2016,
            "embedding_type": "synopsis",
        },
    ),
]


@pytest.fixture(scope="module")
def stub_embeddings() -> StubEmbeddings:
    return StubEmbeddings()


@pytest.fixture(scope="module")
def qdrant_store(stub_embeddings: StubEmbeddings) -> QdrantVectorStore:
    return QdrantVectorStore.from_documents(
        TEST_DOCS,
        embedding=stub_embeddings,
        location=":memory:",
        collection_name="test_movies",
    )
