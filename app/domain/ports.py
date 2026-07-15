from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal, Protocol

from pydantic import BaseModel

from app.models.media_item import MediaItem


@dataclass(frozen=True)
class RetrievedChunk:
    """One retrieved passage and its metadata — the domain-native stand-in for
    LangChain's `Document`, so the domain layer isn't coupled to LangChain
    types. Adapters that actually talk to LangChain (`app/adapters/retrievers.py`)
    convert to/from `Document` right at that boundary."""

    page_content: str
    metadata: dict[str, Any]


@dataclass(frozen=True)
class ChatMessage:
    """One turn of conversation history — the domain-native stand-in for
    LangChain's `BaseMessage`. Adapters that call into LangChain
    (`app/adapters/generators.py`) convert this at that boundary."""

    role: Literal["human", "ai"]
    content: str


class CandidateRetriever(ABC):
    name: str

    @abstractmethod
    async def retrieve(self, query: str) -> list[RetrievedChunk]: ...


class QueryRewriter(ABC):
    @abstractmethod
    async def rewrite(self, question: str, history: list[ChatMessage]) -> str: ...


class ConversationTitler(ABC):
    """Produces a short topic title for a conversation's first exchange — used to
    label its entry in the web UI's Recent-conversations sidebar list."""

    @abstractmethod
    async def title(self, first_question: str, first_answer: str) -> str: ...


class RecommendationCard(BaseModel):
    """One recommended film. `tmdb_id` is declared before `body_md` so it
    resolves before the prose does when streamed via `with_structured_output`
    (see docs/plan-structured-recommendation-output.md §3.3)."""

    tmdb_id: str
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

    tmdb_id: str
    body_md: str


StreamEvent = TextDelta | SectionReady


class RecommendationGenerator(ABC):
    @abstractmethod
    async def generate(
        self, question: str, context: str, history: list[ChatMessage]
    ) -> RecommendationResponse: ...

    @abstractmethod
    def stream(
        self, question: str, context: str, history: list[ChatMessage]
    ) -> AsyncIterator[StreamEvent]:
        """Yield discrete events as the answer completes, rather than raw text
        deltas — a `TextDelta` for a finished block of prose, a
        `SectionReady` the moment one recommendation card's fields are fully
        written. The streaming counterpart to `generate`."""


class MediaItemLookup(Protocol):
    """Resolves a tmdb_id to a full MediaItem (poster, rating, etc.) for rendering
    recommended films — satisfied by QdrantMediaItems."""

    def get_by_id(self, tmdb_id: str) -> MediaItem | None: ...


@dataclass(frozen=True)
class WatchedEmbedding:
    """One point from plex-ingest's `watch_history` collection — see
    docs/vector-store-contract.md. `last_viewed_at` is naive (no tzinfo), matching
    how plex-ingest stores it (Plex's own local-server timestamps carry no tzinfo
    either) — callers compare it against a naive `now`, not an aware one."""

    tmdb_id: str
    vector: list[float]
    last_viewed_at: datetime


class WatchHistoryLookup(Protocol):
    """Recent watched-title embeddings for the diversity recommender's aversion
    vector — satisfied by QdrantWatchHistory. Returns an empty list, not an error,
    when the watch_history collection has nothing in the current window."""

    def recent(self) -> list[WatchedEmbedding]: ...


@dataclass(frozen=True)
class CandidateEmbedding:
    """One `media_items` synopsis-point embedding, the diversity recommender's
    candidate pool — satisfied by QdrantCandidatePool. `media_items` already only
    contains unwatched movies (plex-ingest's own scope — see
    docs/pipeline-design.md), so no separate "exclude watched" filtering is needed
    here."""

    tmdb_id: str
    vector: list[float]
    imdb_rating: float | None


class CandidatePool(Protocol):
    def all(self) -> list[CandidateEmbedding]: ...
