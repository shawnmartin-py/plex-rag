import random
import re
from dataclasses import dataclass

from langchain_core.documents import Document
from langchain_core.messages import BaseMessage

from app.domain.ports import CandidateRetriever, QueryRewriter, RecommendationGenerator

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

    items = list(grouped.values())
    random.shuffle(items)
    blocks = []
    for docs in items:
        ordered = sorted(docs, key=sort_key)
        title = ordered[0].metadata.get("title", "Unknown")
        year = ordered[0].metadata.get("year", "")
        chunks = "\n\n".join(doc.page_content for doc in ordered)
        blocks.append(f"=== {title} ({year}) ===\n{chunks}")
    return "\n\n---\n\n".join(blocks)


def _find_mentioned_ids(grouped: dict[str, list[Document]], response: str) -> list[str]:
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


def _build_coverage_report(
    grouped: dict[str, list[Document]],
    sources: dict[str, set[str]],
    response: str,
    retriever_names: list[str],
) -> CoverageReport:
    response_lower = response.lower()
    recommended: list[CoverageEntry] = []
    dropped: list[CoverageEntry] = []

    for imdb_id, docs in grouped.items():
        title = str(docs[0].metadata.get("title", imdb_id))
        year = str(docs[0].metadata.get("year", ""))
        entry = CoverageEntry(
            title=title, year=year, sources=frozenset(sources.get(imdb_id, set()))
        )
        if title.lower() in response_lower:
            recommended.append(entry)
        else:
            dropped.append(entry)

    return CoverageReport(
        retriever_names=retriever_names, recommended=recommended, dropped=dropped
    )


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

    def recommend(
        self, question: str, history: list[BaseMessage], verbose: bool = False
    ) -> tuple[str, list[str], CoverageReport | None]:
        standalone = self._rewriter.rewrite(question, history) if history else question
        named_sets = [(r.name, r.retrieve(standalone)) for r in self._retrievers]
        grouped, sources = _group_docs(named_sets)
        context = _format_grouped(grouped)
        response = self._generator.generate(question, context, history)
        mentioned_ids = _find_mentioned_ids(grouped, response)
        coverage = None
        if verbose:
            retriever_names = [name for name, _ in named_sets]
            coverage = _build_coverage_report(
                grouped, sources, response, retriever_names
            )
        return response, mentioned_ids, coverage
