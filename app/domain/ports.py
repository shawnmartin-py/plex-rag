from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import Protocol

from langchain_core.documents import Document
from langchain_core.messages import BaseMessage

from app.models.media_item import MediaItem


class CandidateRetriever(ABC):
    name: str

    @abstractmethod
    async def retrieve(self, query: str) -> list[Document]: ...


class QueryRewriter(ABC):
    @abstractmethod
    async def rewrite(self, question: str, history: list[BaseMessage]) -> str: ...


class ConversationTitler(ABC):
    """Produces a short topic title for a conversation's first exchange — used to
    label its entry in the web UI's Recent-conversations sidebar list."""

    @abstractmethod
    async def title(self, first_question: str, first_answer: str) -> str: ...


class RecommendationGenerator(ABC):
    @abstractmethod
    async def generate(
        self, question: str, context: str, history: list[BaseMessage]
    ) -> str: ...

    @abstractmethod
    def stream(
        self, question: str, context: str, history: list[BaseMessage]
    ) -> AsyncIterator[str]:
        """Yield the answer as incremental text deltas rather than waiting for
        the full response — the streaming counterpart to `generate`."""


class MediaItemLookup(Protocol):
    """Resolves an imdb_id to a full MediaItem (poster, rating, etc.) for rendering
    recommended films — satisfied by QdrantMediaItems."""

    def get_by_id(self, imdb_id: str) -> MediaItem | None: ...
