import asyncio
import random
import re
from collections.abc import AsyncIterator
from dataclasses import dataclass, field

from langchain_core.documents import Document
from langchain_core.messages import BaseMessage

from app.domain.ports import CandidateRetriever, QueryRewriter, RecommendationGenerator
from app.formatting.sections import parse_sections, strip_section_heading

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
    named_sets: list[tuple[str, list[Document]]],
) -> tuple[dict[str, list[Document]], dict[str, set[str]]]:
    grouped: dict[str, list[Document]] = {}
    sources: dict[str, set[str]] = {}
    seen: set[tuple[str, str | None, str | None]] = set()
    for retriever_name, docs in named_sets:
        for doc in docs:
            imdb_id = str(doc.metadata.get("imdb_id"))
            key = (
                imdb_id,
                doc.metadata.get("embedding_type"),
                doc.metadata.get("section"),
            )
            if key not in seen:
                seen.add(key)
                grouped.setdefault(imdb_id, []).append(doc)
            sources.setdefault(imdb_id, set()).add(retriever_name)
    return grouped, sources


def _format_grouped(grouped: dict[str, list[Document]]) -> str:
    def sort_key(doc: Document) -> tuple[int, int]:
        is_enriched = 1 if doc.metadata.get("embedding_type") == "enriched" else 0
        return (is_enriched, _SECTION_ORDER.get(doc.metadata.get("section", ""), 99))

    entries = list(grouped.items())
    random.shuffle(entries)
    blocks = []
    for imdb_id, docs in entries:
        ordered = sorted(docs, key=sort_key)
        title = ordered[0].metadata.get("title", "Unknown")
        year = ordered[0].metadata.get("year", "")
        chunks = "\n\n".join(doc.page_content for doc in ordered)
        blocks.append(f"=== {title} ({year}) [imdb_id: {imdb_id}] ===\n{chunks}")
    return "\n\n---\n\n".join(blocks)


# The generator is instructed (see app/adapters/generators.py) to echo each
# recommended film's imdb_id back as a hidden comment right after its heading —
# exact-id matching instead of fuzzy title search, which used to misfire on
# sequels/reboots sharing a title (see docs/recommender.md).
_MARKER_RE = re.compile(r"<!--\s*imdb:(tt\d+)\s*-->")
_MARKER_LINE_RE = re.compile(r"[ \t]*<!--\s*imdb:tt\d+\s*-->[ \t]*\n?")


def _strip_markers(response: str) -> str:
    return _MARKER_LINE_RE.sub("", response)


def _match_section_id(
    text: str, grouped: dict[str, list[Document]], claimed: set[str]
) -> str | None:
    """Which grouped film a single (already-closed) section text recommends.
    Prefers its hidden imdb marker; falls back to fuzzy title search among
    films not already claimed by an earlier section this turn."""
    marker = _MARKER_RE.search(text)
    if marker and marker.group(1) in grouped:
        return marker.group(1)
    text_lower = text.lower()
    for imdb_id, docs in grouped.items():
        if imdb_id in claimed:
            continue
        title = re.sub(r"[#*_`]", "", docs[0].metadata.get("title", "")).lower()
        if title and re.search(r"\b" + re.escape(title) + r"\b", text_lower):
            return imdb_id
    return None


def _find_mentioned_ids_by_title(
    grouped: dict[str, list[Document]], response: str
) -> list[str]:
    response_lower = response.lower()

    def first_position(docs: list[Document]) -> int:
        title = re.sub(r"[#*_`]", "", docs[0].metadata.get("title", "")).lower()
        m = re.search(r"\b" + re.escape(title) + r"\b", response_lower)
        return m.start() if m else -1

    ordered = sorted(
        ((imdb_id, first_position(docs)) for imdb_id, docs in grouped.items()),
        key=lambda x: x[1],
    )
    return [imdb_id for imdb_id, pos in ordered if pos >= 0]


def _find_mentioned_ids(grouped: dict[str, list[Document]], response: str) -> list[str]:
    """Which grouped films the response actually recommends, in the order it
    recommends them. Prefers the generator's hidden imdb-id markers; falls back
    to fuzzy title search for the rare case the model omits them."""
    found = _MARKER_RE.findall(response)
    marker_ids = [imdb_id for imdb_id in found if imdb_id in grouped]
    if not marker_ids:
        return _find_mentioned_ids_by_title(grouped, response)
    seen: set[str] = set()
    ordered: list[str] = []
    for imdb_id in marker_ids:
        if imdb_id not in seen:
            seen.add(imdb_id)
            ordered.append(imdb_id)
    return ordered


def _build_coverage_report(
    grouped: dict[str, list[Document]],
    sources: dict[str, set[str]],
    mentioned_ids: list[str],
    retriever_names: list[str],
) -> CoverageReport:
    mentioned = set(mentioned_ids)
    recommended: list[CoverageEntry] = []
    dropped: list[CoverageEntry] = []

    for imdb_id, docs in grouped.items():
        title = str(docs[0].metadata.get("title", imdb_id))
        year = str(docs[0].metadata.get("year", ""))
        entry = CoverageEntry(
            title=title, year=year, sources=frozenset(sources.get(imdb_id, set()))
        )
        if imdb_id in mentioned:
            recommended.append(entry)
        else:
            dropped.append(entry)

    return CoverageReport(
        retriever_names=retriever_names, recommended=recommended, dropped=dropped
    )


@dataclass(frozen=True)
class TextDelta:
    """A finished block of plain prose — intro text, a trailing note, etc."""

    text: str


@dataclass(frozen=True)
class SectionReady:
    """One finished numbered recommendation, ready to render as a card.
    `imdb_id` is `None` in the rare case neither the marker nor a title
    fallback could identify which film it is."""

    imdb_id: str | None
    body_md: str


StreamEvent = TextDelta | SectionReady


@dataclass
class StreamedAnswer:
    """`events` yields one event per completed section, in generation order —
    a `SectionReady` the moment the generator finishes writing a numbered
    recommendation, rather than waiting for the whole response. `answer` and
    `imdb_ids` are only meaningful once `events` has been fully consumed."""

    events: AsyncIterator[StreamEvent]
    answer: str = ""
    imdb_ids: list[str] = field(default_factory=list)


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

    async def recommend(
        self, question: str, history: list[BaseMessage], verbose: bool = False
    ) -> tuple[str, list[str], CoverageReport | None]:
        standalone = (
            await self._rewriter.rewrite(question, history) if history else question
        )
        results = await asyncio.gather(
            *(r.retrieve(standalone) for r in self._retrievers)
        )
        named_sets = list(zip((r.name for r in self._retrievers), results, strict=True))
        grouped, sources = _group_docs(named_sets)
        context = _format_grouped(grouped)
        raw_response = await self._generator.generate(question, context, history)
        mentioned_ids = _find_mentioned_ids(grouped, raw_response)
        response = _strip_markers(raw_response)
        coverage = None
        if verbose:
            retriever_names = [name for name, _ in named_sets]
            coverage = _build_coverage_report(
                grouped, sources, mentioned_ids, retriever_names
            )
        return response, mentioned_ids, coverage

    async def recommend_stream(
        self, question: str, history: list[BaseMessage]
    ) -> StreamedAnswer:
        """Streaming counterpart to `recommend`, minus `verbose` coverage
        reporting (which needs the full response text up front, so it stays
        on the non-streaming path used by the CLI). Reuses `parse_sections`'
        section-boundary detection to know when a numbered recommendation is
        finished: everything before the last split point is guaranteed
        complete, since the model only starts the next section once it's
        done with the current one — which also means an in-progress imdb
        marker never leaks, since its section simply isn't flushed yet."""
        standalone = (
            await self._rewriter.rewrite(question, history) if history else question
        )
        results = await asyncio.gather(
            *(r.retrieve(standalone) for r in self._retrievers)
        )
        named_sets = list(zip((r.name for r in self._retrievers), results, strict=True))
        grouped, _sources = _group_docs(named_sets)
        context = _format_grouped(grouped)

        def to_events(
            parts: list[tuple[bool, str]], claimed: set[str]
        ) -> list[StreamEvent]:
            events: list[StreamEvent] = []
            for is_numbered, text in parts:
                if is_numbered:
                    imdb_id = _match_section_id(text, grouped, claimed)
                    if imdb_id is not None:
                        claimed.add(imdb_id)
                        answer.imdb_ids.append(imdb_id)
                    body = strip_section_heading(_strip_markers(text))
                    events.append(SectionReady(imdb_id=imdb_id, body_md=body))
                else:
                    cleaned = _strip_markers(text)
                    if cleaned:
                        events.append(TextDelta(text=cleaned))
            return events

        async def _events() -> AsyncIterator[StreamEvent]:
            raw_parts: list[str] = []
            claimed: set[str] = set()
            emitted = 0
            async for delta in self._generator.stream(question, context, history):
                raw_parts.append(delta)
                parts = parse_sections("".join(raw_parts))
                closed_end = max(0, len(parts) - 1)
                for event in to_events(parts[emitted:closed_end], claimed):
                    yield event
                emitted = closed_end
            parts = parse_sections("".join(raw_parts))
            for event in to_events(parts[emitted:], claimed):
                yield event
            answer.answer = _strip_markers("".join(raw_parts))

        answer = StreamedAnswer(events=_events())
        return answer
