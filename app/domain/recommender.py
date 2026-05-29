import random

from langchain_core.documents import Document
from langchain_core.messages import BaseMessage

from app.domain.ports import CandidateRetriever, QueryRewriter, RecommendationGenerator

_SECTION_ORDER = {"": 0, "craft": 1, "meaning": 2, "context": 3}


def _group_docs(
    named_sets: list[tuple[str, list[Document]]],
) -> tuple[dict[str, list[Document]], dict[str, set[str]]]:
    grouped: dict[str, list[Document]] = {}
    sources: dict[str, set[str]] = {}
    seen: set[tuple] = set()
    for retriever_name, docs in named_sets:
        for doc in docs:
            imdb_id = doc.metadata.get("imdb_id")
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
    def sort_key(doc: Document) -> tuple:
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
    return [imdb_id for imdb_id, docs in grouped.items() if docs[0].metadata.get("title", "").lower() in response_lower]


def _print_coverage(
    grouped: dict[str, list[Document]],
    sources: dict[str, set[str]],
    response: str,
    retriever_names: list[str],
) -> None:
    response_lower = response.lower()
    col = 44

    recommended: list[tuple[str, str, set[str]]] = []
    dropped: list[tuple[str, str, set[str]]] = []

    for imdb_id, docs in grouped.items():
        title = docs[0].metadata.get("title", imdb_id)
        year = str(docs[0].metadata.get("year", ""))
        flags = sources.get(imdb_id, set())
        if title.lower() in response_lower:
            recommended.append((title, year, flags))
        else:
            dropped.append((title, year, flags))

    print("\n[Source coverage]")

    if recommended:
        print(f"  {'RECOMMENDED':<{col}}  source(s)")
        print(f"  {'─' * col}  {'─' * 22}")
        for title, year, flags in recommended:
            label = f"{title} ({year})"
            print(f"  {label:<{col}}  {', '.join(sorted(flags))}")

    if dropped:
        print(f"\n  {'DROPPED (in context, not recommended)':<{col}}  source(s)")
        print(f"  {'─' * col}  {'─' * 22}")
        for title, year, flags in dropped:
            label = f"{title} ({year})"
            print(f"  {label:<{col}}  {', '.join(sorted(flags))}")

    counts = {name: 0 for name in retriever_names}
    for _, _, flags in recommended:
        for name in flags:
            if name in counts:
                counts[name] += 1
    total = len(recommended)
    summary = "  · ".join(f"{name} {counts[name]}/{total}" for name in retriever_names)
    print(f"\n  Coverage: {summary}\n")


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

    def recommend(self, question: str, history: list[BaseMessage], verbose: bool = False) -> tuple[str, list[str]]:
        standalone = self._rewriter.rewrite(question, history) if history else question
        named_sets = [(r.name, r.retrieve(standalone)) for r in self._retrievers]
        grouped, sources = _group_docs(named_sets)
        context = _format_grouped(grouped)
        response = self._generator.generate(question, context, history)
        mentioned_ids = _find_mentioned_ids(grouped, response)
        if verbose:
            retriever_names = [name for name, _ in named_sets]
            _print_coverage(grouped, sources, response, retriever_names)
        return response, mentioned_ids
