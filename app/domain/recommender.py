import asyncio
import random
from collections.abc import AsyncIterator
from dataclasses import dataclass, field

from app.domain.ports import (
    CandidateRetriever,
    ChatMessage,
    QueryRewriter,
    RecommendationGenerator,
    RecommendationResponse,
    RetrievedChunk,
    SectionReady,
    StreamEvent,
)

_SECTION_ORDER = {"": 0, "craft": 1, "meaning": 2, "context": 3}


@dataclass(frozen=True)
class CoverageEntry:
    title: str
    year: str
    sources: frozenset[str]


@dataclass(frozen=True)
class CoverageReport:
    """Which retriever(s) surfaced each film, and whether it made it into the
    final response — pure data. Rendering it (CLI table, log line, etc.) is a
    presentation concern that belongs to whichever entry point asked for it,
    not to the domain layer."""

    retriever_names: list[str]
    recommended: list[CoverageEntry]
    dropped: list[CoverageEntry]


def _group_docs(
    named_sets: list[tuple[str, list[RetrievedChunk]]],
) -> tuple[dict[str, list[RetrievedChunk]], dict[str, set[str]]]:
    grouped: dict[str, list[RetrievedChunk]] = {}
    sources: dict[str, set[str]] = {}
    seen: set[tuple[str, str | None, str | None]] = set()
    for retriever_name, docs in named_sets:
        for doc in docs:
            tmdb_id = str(doc.metadata.get("tmdb_id"))
            key = (
                tmdb_id,
                doc.metadata.get("embedding_type"),
                doc.metadata.get("section"),
            )
            if key not in seen:
                seen.add(key)
                grouped.setdefault(tmdb_id, []).append(doc)
            sources.setdefault(tmdb_id, set()).add(retriever_name)
    return grouped, sources


def _format_grouped(grouped: dict[str, list[RetrievedChunk]]) -> str:
    def sort_key(doc: RetrievedChunk) -> tuple[int, int]:
        is_enriched = 1 if doc.metadata.get("embedding_type") == "enriched" else 0
        return (is_enriched, _SECTION_ORDER.get(doc.metadata.get("section", ""), 99))

    entries = list(grouped.items())
    random.shuffle(entries)
    blocks = []
    for tmdb_id, docs in entries:
        ordered = sorted(docs, key=sort_key)
        title = ordered[0].metadata.get("title", "Unknown")
        year = ordered[0].metadata.get("year", "")
        chunks = "\n\n".join(doc.page_content for doc in ordered)
        blocks.append(f"=== {title} ({year}) [tmdb_id: {tmdb_id}] ===\n{chunks}")
    return "\n\n---\n\n".join(blocks)


def _format_card_heading(
    index: int, tmdb_id: str, grouped: dict[str, list[RetrievedChunk]]
) -> str:
    """Synthesize the numbered heading the model no longer writes itself
    (title/year live on the structured schema's context, not its output) —
    used to build the plain-text `answer` string for the CLI and persisted
    conversation history. The UI never sees this: it renders title/year from
    the matched `MediaItem` and only displays a card's `body_md`."""
    docs = grouped[tmdb_id]
    title = docs[0].metadata.get("title", tmdb_id)
    year = docs[0].metadata.get("year", "")
    return f"{index}. **{title}** ({year})"


def _build_answer(
    result: RecommendationResponse, grouped: dict[str, list[RetrievedChunk]]
) -> tuple[str, list[str]]:
    """Turns a generator's structured response into the plain-text `answer`
    string plus the tmdb_ids it actually recommended — dropping any card
    whose tmdb_id the model invented rather than copying from context (same
    shape as LLMKnowledgeRetriever's title filter in
    app/adapters/retrievers.py). Shared by `recommend` and
    `recommend_with_context`."""
    cards = [c for c in result.cards if c.tmdb_id in grouped]
    mentioned_ids = [c.tmdb_id for c in cards]

    parts: list[str] = []
    if result.intro:
        parts.append(result.intro)
    for i, card in enumerate(cards, start=1):
        heading = _format_card_heading(i, card.tmdb_id, grouped)
        parts.append(f"{heading}\n{card.body_md}")
    if result.closing_note:
        parts.append(result.closing_note)
    return "\n\n".join(parts), mentioned_ids


def _build_coverage_report(
    grouped: dict[str, list[RetrievedChunk]],
    sources: dict[str, set[str]],
    mentioned_ids: list[str],
    retriever_names: list[str],
) -> CoverageReport:
    mentioned = set(mentioned_ids)
    recommended: list[CoverageEntry] = []
    dropped: list[CoverageEntry] = []

    for tmdb_id, docs in grouped.items():
        title = str(docs[0].metadata.get("title", tmdb_id))
        year = str(docs[0].metadata.get("year", ""))
        entry = CoverageEntry(
            title=title, year=year, sources=frozenset(sources.get(tmdb_id, set()))
        )
        if tmdb_id in mentioned:
            recommended.append(entry)
        else:
            dropped.append(entry)

    return CoverageReport(
        retriever_names=retriever_names, recommended=recommended, dropped=dropped
    )


@dataclass
class StreamedAnswer:
    """`events` yields one event per completed section, in generation order —
    a `SectionReady` the moment the generator finishes writing a
    recommendation card, rather than waiting for the whole response. `answer`
    and `tmdb_ids` are only meaningful once `events` has been fully
    consumed."""

    events: AsyncIterator[StreamEvent]
    answer: str = ""
    tmdb_ids: list[str] = field(default_factory=list)


class MovieRecommender:
    def __init__(
        self,
        retrievers: list[CandidateRetriever],
        generator: RecommendationGenerator,
        rewriter: QueryRewriter,
    ) -> None:
        self._retrievers = retrievers
        self._generator = generator
        self._rewriter = rewriter

    async def _retrieve_context(
        self, question: str, history: list[ChatMessage]
    ) -> tuple[str, dict[str, list[RetrievedChunk]], dict[str, set[str]]]:
        """Rewrite (if there's history) → fan every retriever out concurrently
        → dedupe/group by film → format into the context blob the generator
        sees. Shared by `recommend`, `recommend_stream`, and
        `recommend_with_context`."""
        standalone = (
            await self._rewriter.rewrite(question, history) if history else question
        )
        results = await asyncio.gather(
            *(r.retrieve(standalone) for r in self._retrievers)
        )
        named_sets = list(zip((r.name for r in self._retrievers), results, strict=True))
        grouped, sources = _group_docs(named_sets)
        context = _format_grouped(grouped)
        return context, grouped, sources

    async def recommend(
        self, question: str, history: list[ChatMessage], verbose: bool = False
    ) -> tuple[str, list[str], CoverageReport | None]:
        context, grouped, sources = await self._retrieve_context(question, history)
        result = await self._generator.generate(question, context, history)
        answer, mentioned_ids = _build_answer(result, grouped)

        coverage = None
        if verbose:
            retriever_names = [r.name for r in self._retrievers]
            coverage = _build_coverage_report(
                grouped, sources, mentioned_ids, retriever_names
            )
        return answer, mentioned_ids, coverage

    async def recommend_with_context(
        self, question: str, history: list[ChatMessage]
    ) -> tuple[str, list[str], str]:
        """Same generation path as `recommend` (minus verbose coverage
        reporting), plus the formatted context string the generator actually
        saw — the one piece `recommend` doesn't expose. `recommend` doesn't
        need it (callers only care about the answer), but RAGAS's Faithfulness
        metric does: it scores whether `answer`'s claims are grounded in
        `context`, so evals/ needs the true retrieved grounding for this exact
        turn rather than a separately re-run, possibly-drifted approximation.
        See evals/README.md."""
        context, grouped, _sources = await self._retrieve_context(question, history)
        result = await self._generator.generate(question, context, history)
        answer, mentioned_ids = _build_answer(result, grouped)
        return answer, mentioned_ids, context

    async def recommend_stream(
        self, question: str, history: list[ChatMessage]
    ) -> StreamedAnswer:
        """Streaming counterpart to `recommend`, minus `verbose` coverage
        reporting (which needs the full response up front, so it stays on the
        non-streaming path used by the CLI). Boundary detection (deciding when
        a card is finished) is the generator's job now — see
        `GeminiRecommendationGenerator.stream`; this just forwards its events,
        dropping any card whose tmdb_id isn't a real candidate and building
        the plain-text `answer` (with headings synthesized from `grouped`) as
        events go by."""
        context, grouped, _sources = await self._retrieve_context(question, history)

        async def _events() -> AsyncIterator[StreamEvent]:
            answer_parts: list[str] = []
            card_index = 0
            async for event in self._generator.stream(question, context, history):
                if isinstance(event, SectionReady):
                    if event.tmdb_id not in grouped:
                        continue
                    card_index += 1
                    heading = _format_card_heading(card_index, event.tmdb_id, grouped)
                    answer_parts.append(f"{heading}\n{event.body_md}")
                    answer.tmdb_ids.append(event.tmdb_id)
                    yield event
                else:
                    if event.text:
                        answer_parts.append(event.text)
                    yield event
            answer.answer = "\n\n".join(answer_parts)

        answer = StreamedAnswer(events=_events())
        return answer
