from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Protocol

from langchain_core.documents import Document
from langchain_core.messages import BaseMessage
from pydantic import BaseModel

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


class RecommendationCard(BaseModel):
    """One recommended film. `imdb_id` is declared before `body_md` so it
    resolves before the prose does when streamed via `with_structured_output`
    (see docs/plan-structured-recommendation-output.md §3.3)."""

    imdb_id: str
    body_md: str


class RecommendationResponse(BaseModel):
    """The generator's full structured answer. `cards` carries no title/year
    heading — the UI already renders those from the matched `MediaItem`, and
    the CLI/persisted-history text synthesizes a heading from `grouped`
    (`app/domain/recommender.py`) rather than having the model restate it."""

    intro: str = ""
    cards: list[RecommendationCard] = []
    closing_note: str = ""


@dataclass(frozen=True)
class TextDelta:
    """A finished block of plain prose — intro text or a trailing note."""

    text: str


@dataclass(frozen=True)
class SectionReady:
    """One finished recommendation card, ready to render as a UI card."""

    imdb_id: str
    body_md: str


StreamEvent = TextDelta | SectionReady


class RecommendationGenerator(ABC):
    @abstractmethod
    async def generate(
        self, question: str, context: str, history: list[BaseMessage]
    ) -> RecommendationResponse: ...

    @abstractmethod
    def stream(
        self, question: str, context: str, history: list[BaseMessage]
    ) -> AsyncIterator[StreamEvent]:
        """Yield discrete events as the answer completes, rather than raw text
        deltas — a `TextDelta` for a finished block of prose, a
        `SectionReady` the moment one recommendation card's fields are fully
        written. The streaming counterpart to `generate`."""


class MediaItemLookup(Protocol):
    """Resolves an imdb_id to a full MediaItem (poster, rating, etc.) for rendering
    recommended films — satisfied by QdrantMediaItems."""

    def get_by_id(self, imdb_id: str) -> MediaItem | None: ...
