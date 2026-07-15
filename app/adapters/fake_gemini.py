"""Deterministic stand-ins for `ChatGoogleGenerativeAI`/`GoogleGenerativeAIEmbeddings`,
swapped in by `app/bootstrap.py` when `FAKE_GEMINI=true` (see `app/config.py`). Makes
zero network calls and needs no `GOOGLE_API_KEY` — lets the CLI, NiceGUI, and API
entry points be driven end to end (including real Qdrant retrieval) for local/manual
testing without spending Gemini quota. Never used by `evals/`, which builds its own
real Gemini clients independently of `app/bootstrap.py` — faking the model under test
would make those evals meaningless.
"""

import hashlib
import math
import random
import re
from typing import Any, cast

from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable, RunnableLambda
from pydantic import BaseModel

from app.adapters.retrievers import TitleSelection
from app.domain.ports import RecommendationCard, RecommendationResponse
from app.repositories.vector_store import VECTOR_SIZE

_TITLER_MARKER = "Summarize the topic of this movie-recommendation exchange"

_FAKE_TITLE_POOL = [
    "Fake-mode picks",
    "Deterministic double feature",
    "Offline test screening",
    "Local dev movie night",
    "Quota-free recommendations",
    "Sample conversation topic",
]

_CONTEXT_BLOCK_RE = re.compile(r"=== (.+?) \((\d*)\) \[tmdb_id: ([^\]]+)\] ===")
_AVAILABLE_MOVIES_MARKER = "Available movies:\n"


def _joined_text(messages: list[BaseMessage]) -> str:
    return "\n".join(m.text for m in messages)


def _last_human_text(messages: list[BaseMessage]) -> str:
    """Falls back to echoing the final human turn — a reasonable deterministic
    no-op for both the query rewriter (whose input is already a standalone-ish
    question most of the time) and the HyDE retriever's hypothetical-document
    prompt (whose output is only ever embedded, never shown to a user)."""
    humans = [m.text for m in messages if m.type == "human"]
    return humans[-1] if humans else _joined_text(messages)


def _fake_title(messages: list[BaseMessage]) -> str:
    digest = hashlib.sha256(_joined_text(messages).encode()).hexdigest()
    return _FAKE_TITLE_POOL[int(digest, 16) % len(_FAKE_TITLE_POOL)]


def _fake_plain_text(messages: list[BaseMessage]) -> str:
    if _TITLER_MARKER in _joined_text(messages):
        return _fake_title(messages)
    return _last_human_text(messages)


def _fake_recommendation_response(
    messages: list[BaseMessage],
) -> RecommendationResponse:
    """Builds cards from real `[tmdb_id: ...]` markers already present in the
    retrieved context (see `_format_grouped` in `app/domain/recommender.py`),
    so fake mode still exercises real card rendering (posters, ratings) for
    real library titles — sorted by tmdb_id for a selection that's stable
    regardless of `_format_grouped`'s own randomized ordering."""
    matches = sorted(
        _CONTEXT_BLOCK_RE.findall(_joined_text(messages)), key=lambda m: m[2]
    )
    if not matches:
        return RecommendationResponse(
            closing_note="FAKE_GEMINI=true and no candidates were retrieved for "
            "this request."
        )
    cards = [
        RecommendationCard(
            tmdb_id=tmdb_id,
            body_md=(
                "**Why it fits:** Deterministic fake-mode pick surfaced by real "
                f"retrieval for this request — no Gemini call was made.\n"
                f"- Matched candidate: {title} ({year})."
            ),
        )
        for title, year, tmdb_id in matches[:3]
    ]
    return RecommendationResponse(
        intro="Fake-mode recommendations (FAKE_GEMINI=true — no Gemini call was made).",
        cards=cards,
    )


def _fake_title_selection(messages: list[BaseMessage]) -> TitleSelection:
    joined = _joined_text(messages)
    idx = joined.find(_AVAILABLE_MOVIES_MARKER)
    if idx == -1:
        return TitleSelection(titles=[])
    titles = [
        line.strip()
        for line in joined[idx + len(_AVAILABLE_MOVIES_MARKER) :].splitlines()
        if line.strip()
    ]
    return TitleSelection(titles=sorted(titles)[:5])


class FakeChatModel(BaseChatModel):
    """Real `BaseChatModel` subclass (so LCEL chains compose and invoke exactly
    like they do with `ChatGoogleGenerativeAI`) that recognizes this codebase's
    plain-text chains (query rewrite, conversation title, HyDE profile) by their
    system-prompt wording, and its two `with_structured_output` schemas
    (`RecommendationResponse`, `TitleSelection`) by name — anything else raises
    rather than silently returning something structurally wrong."""

    @property
    def _llm_type(self) -> str:
        return "fake-gemini"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        content = _fake_plain_text(messages)
        return ChatResult(
            generations=[ChatGeneration(message=AIMessage(content=content))]
        )

    def with_structured_output(
        self, schema: dict[str, Any] | type, *, include_raw: bool = False, **kwargs: Any
    ) -> Runnable[Any, dict[str, Any] | BaseModel]:
        if not (isinstance(schema, type) and issubclass(schema, BaseModel)):
            raise TypeError(
                "FakeChatModel.with_structured_output only supports Pydantic "
                f"schemas, got {schema!r}"
            )

        def _invoke(prompt_value: Any) -> BaseModel:
            messages = (
                prompt_value.to_messages()
                if hasattr(prompt_value, "to_messages")
                else prompt_value
            )
            if issubclass(schema, RecommendationResponse):
                return cast(BaseModel, _fake_recommendation_response(messages))
            if issubclass(schema, TitleSelection):
                return cast(BaseModel, _fake_title_selection(messages))
            raise NotImplementedError(
                f"FakeChatModel.with_structured_output has no fake for {schema!r} "
                "— add one in app/adapters/fake_gemini.py."
            )

        return RunnableLambda(_invoke)


def _deterministic_vector(text: str, dims: int) -> list[float]:
    rng = random.Random(hashlib.sha256(text.encode()).digest())  # noqa: S311 — deterministic fake vector, not security
    vector = [rng.uniform(-1.0, 1.0) for _ in range(dims)]
    norm = math.sqrt(sum(v * v for v in vector)) or 1.0
    return [v / norm for v in vector]


class DeterministicEmbeddings(Embeddings):
    """Hash-derived unit vectors, not semantically meaningful — Qdrant similarity
    search still runs for real (so retrieval mechanics, filters, and dedup are all
    exercised), just without real relevance ranking. Matches
    `gemini-embedding-001`'s dimensionality by default so it satisfies
    `connect_vector_store`'s vector-size check against the real collection."""

    def __init__(self, dims: int = VECTOR_SIZE) -> None:
        self._dims = dims

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_query(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return _deterministic_vector(text, self._dims)

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.embed_documents(texts)

    async def aembed_query(self, text: str) -> list[float]:
        return self.embed_query(text)
