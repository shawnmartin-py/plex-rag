from collections.abc import AsyncIterator
from unittest.mock import MagicMock

import pytest
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from app.domain.ports import CandidateRetriever, QueryRewriter, RecommendationGenerator
from app.domain.recommender import (
    MovieRecommender,
    SectionReady,
    TextDelta,
    _find_mentioned_ids,
    _format_grouped,
    _group_docs,
    _strip_markers,
)

_R = "retriever"  # generic retriever name for tests


def make_doc(
    imdb_id: str,
    title: str,
    embedding_type: str = "synopsis",
    section: str | None = None,
) -> Document:
    metadata = {
        "imdb_id": imdb_id,
        "title": title,
        "year": 2020,
        "embedding_type": embedding_type,
    }
    if section is not None:
        metadata["section"] = section
    return Document(
        page_content=f"Content for {title} ({embedding_type}/{section})",
        metadata=metadata,
    )


# --- _group_docs ---


def test_group_docs_groups_by_imdb_id() -> None:
    synopsis = make_doc("tt001", "Parasite", "synopsis")
    craft = make_doc("tt001", "Parasite", "enriched", "craft")
    grouped, _ = _group_docs([(_R, [synopsis, craft])])
    assert len(grouped["tt001"]) == 2


def test_group_docs_deduplicates_exact_same_doc() -> None:
    doc = make_doc("tt001", "Parasite", "synopsis")
    grouped, _ = _group_docs([(_R, [doc]), (_R, [doc])])
    assert len(grouped["tt001"]) == 1


def test_group_docs_keeps_different_sections_for_same_movie() -> None:
    synopsis = make_doc("tt001", "Parasite", "synopsis")
    craft = make_doc("tt001", "Parasite", "enriched", "craft")
    meaning = make_doc("tt001", "Parasite", "enriched", "meaning")
    grouped, _ = _group_docs([(_R, [synopsis]), (_R, [craft]), (_R, [meaning])])
    assert len(grouped["tt001"]) == 3


def test_group_docs_dedup_key_is_imdb_id_type_section() -> None:
    # Same imdb_id + type + section = duplicate even if retrieved by
    # different retrievers
    doc1 = make_doc("tt001", "Parasite", "enriched", "craft")
    doc2 = make_doc("tt001", "Parasite", "enriched", "craft")
    grouped, _ = _group_docs([("r1", [doc1]), ("r2", [doc2])])
    assert len(grouped["tt001"]) == 1


def test_group_docs_different_movies_stay_separate() -> None:
    a = make_doc("tt001", "Parasite", "synopsis")
    b = make_doc("tt002", "Oldboy", "synopsis")
    grouped, _ = _group_docs([(_R, [a, b])])
    assert "tt001" in grouped
    assert "tt002" in grouped
    assert len(grouped) == 2


def test_group_docs_synopsis_and_enriched_for_same_movie_are_both_kept() -> None:
    synopsis = make_doc("tt001", "Parasite", "synopsis")
    craft = make_doc("tt001", "Parasite", "enriched", "craft")
    grouped, _ = _group_docs([(_R, [synopsis]), (_R, [craft])])
    types = {doc.metadata["embedding_type"] for doc in grouped["tt001"]}
    assert "synopsis" in types
    assert "enriched" in types


def test_group_docs_empty_candidate_sets() -> None:
    grouped, sources = _group_docs([(_R, []), (_R, [])])
    assert grouped == {}
    assert sources == {}


def test_group_docs_preserves_all_movies_across_sets() -> None:
    a = make_doc("tt001", "Parasite", "synopsis")
    b = make_doc("tt002", "Oldboy", "synopsis")
    c = make_doc("tt003", "The Handmaiden", "synopsis")
    grouped, _ = _group_docs([(_R, [a]), (_R, [b]), (_R, [c])])
    assert set(grouped.keys()) == {"tt001", "tt002", "tt003"}


def test_group_docs_tracks_sources_per_retriever() -> None:
    doc_a = make_doc("tt001", "Parasite", "synopsis")
    doc_b = make_doc("tt001", "Parasite", "enriched", "craft")
    grouped, sources = _group_docs([("r1", [doc_a]), ("r2", [doc_b])])
    assert sources["tt001"] == {"r1", "r2"}


def test_group_docs_source_deduplication_does_not_inflate_retriever_set() -> None:
    doc = make_doc("tt001", "Parasite", "synopsis")
    _, sources = _group_docs([("r1", [doc]), ("r1", [doc])])
    assert sources["tt001"] == {"r1"}


# --- _format_grouped ---


def test_format_grouped_includes_movie_title_header() -> None:
    doc = make_doc("tt001", "Parasite", "synopsis")
    result = _format_grouped({"tt001": [doc]})
    assert "Parasite" in result


def test_format_grouped_includes_year_in_header() -> None:
    doc = make_doc("tt001", "Parasite", "synopsis")
    result = _format_grouped({"tt001": [doc]})
    assert "2020" in result


def test_format_grouped_includes_all_chunk_content() -> None:
    synopsis = make_doc("tt001", "Parasite", "synopsis")
    craft = make_doc("tt001", "Parasite", "enriched", "craft")
    result = _format_grouped({"tt001": [synopsis, craft]})
    assert synopsis.page_content in result
    assert craft.page_content in result


def test_format_grouped_separates_movies_with_delimiter() -> None:
    a = make_doc("tt001", "Parasite", "synopsis")
    b = make_doc("tt002", "Oldboy", "synopsis")
    result = _format_grouped({"tt001": [a], "tt002": [b]})
    assert "---" in result


def test_format_grouped_orders_synopsis_before_enrichment() -> None:
    craft = make_doc("tt001", "Parasite", "enriched", "craft")
    synopsis = make_doc("tt001", "Parasite", "synopsis")
    # Pass craft first — format should still put synopsis first
    result = _format_grouped({"tt001": [craft, synopsis]})
    assert result.index(synopsis.page_content) < result.index(craft.page_content)


def test_format_grouped_orders_enrichment_sections_craft_meaning_context() -> None:
    context = make_doc("tt001", "Parasite", "enriched", "context")
    craft = make_doc("tt001", "Parasite", "enriched", "craft")
    meaning = make_doc("tt001", "Parasite", "enriched", "meaning")
    result = _format_grouped({"tt001": [context, meaning, craft]})
    assert result.index(craft.page_content) < result.index(meaning.page_content)
    assert result.index(meaning.page_content) < result.index(context.page_content)


def test_format_grouped_empty_grouped() -> None:
    assert _format_grouped({}) == ""


def test_format_grouped_includes_imdb_id_in_header() -> None:
    doc = make_doc("tt001", "Parasite", "synopsis")
    result = _format_grouped({"tt001": [doc]})
    assert "tt001" in result


# --- _find_mentioned_ids / _strip_markers ---


def test_find_mentioned_ids_prefers_markers_over_title_search() -> None:
    grouped = {
        "tt001": [make_doc("tt001", "Parasite")],
        "tt002": [make_doc("tt002", "Oldboy")],
    }
    response = (
        "1. **Parasite** (2019)\n<!-- imdb:tt002 -->\nActually about Oldboy's vibe."
    )
    # Marker says tt002, title text says Parasite — marker wins.
    assert _find_mentioned_ids(grouped, response) == ["tt002"]


def test_find_mentioned_ids_orders_by_marker_appearance() -> None:
    grouped = {
        "tt001": [make_doc("tt001", "Parasite")],
        "tt002": [make_doc("tt002", "Oldboy")],
    }
    response = "1. Oldboy\n<!-- imdb:tt002 -->\n2. Parasite\n<!-- imdb:tt001 -->"
    assert _find_mentioned_ids(grouped, response) == ["tt002", "tt001"]


def test_find_mentioned_ids_dedupes_repeated_markers() -> None:
    grouped = {"tt001": [make_doc("tt001", "Parasite")]}
    response = "<!-- imdb:tt001 --> ... <!-- imdb:tt001 -->"
    assert _find_mentioned_ids(grouped, response) == ["tt001"]


def test_find_mentioned_ids_ignores_marker_for_unknown_film() -> None:
    grouped = {"tt001": [make_doc("tt001", "Parasite")]}
    response = "<!-- imdb:tt999 -->"
    # Unknown id falls back to title search, which finds nothing either.
    assert _find_mentioned_ids(grouped, response) == []


def test_find_mentioned_ids_falls_back_to_title_search_without_markers() -> None:
    grouped = {"tt001": [make_doc("tt001", "Parasite")]}
    response = "Parasite is a great pick."
    assert _find_mentioned_ids(grouped, response) == ["tt001"]


def test_strip_markers_removes_marker_lines() -> None:
    response = "1. **Parasite** (2019)\n<!-- imdb:tt001 -->\nGreat film."
    assert "imdb:" not in _strip_markers(response)
    assert "Great film." in _strip_markers(response)


def test_strip_markers_leaves_text_without_markers_unchanged() -> None:
    assert _strip_markers("no markers here") == "no markers here"


# --- MovieRecommender ---


class StubRetriever(CandidateRetriever):
    name = "stub"

    def __init__(self, docs: list[Document]) -> None:
        self._docs = docs

    async def retrieve(self, query: str) -> list[Document]:
        return self._docs


class StubRewriter(QueryRewriter):
    async def rewrite(self, question: str, history: list[BaseMessage]) -> str:
        return f"rewritten: {question}"


class StubGenerator(RecommendationGenerator):
    async def generate(
        self, question: str, context: str, history: list[BaseMessage]
    ) -> str:
        return f"answer for: {question}"

    async def stream(
        self, question: str, context: str, history: list[BaseMessage]
    ) -> AsyncIterator[str]:
        yield f"answer for: {question}"


class ChunkedStubGenerator(RecommendationGenerator):
    """Streams pre-set chunks, letting tests control exactly where a marker
    falls across chunk boundaries."""

    def __init__(self, chunks: list[str]) -> None:
        self._chunks = chunks

    async def generate(
        self, question: str, context: str, history: list[BaseMessage]
    ) -> str:
        return "".join(self._chunks)

    async def stream(
        self, question: str, context: str, history: list[BaseMessage]
    ) -> AsyncIterator[str]:
        for chunk in self._chunks:
            yield chunk


@pytest.fixture
def single_doc() -> Document:
    return make_doc("tt001", "Parasite")


async def test_recommend_with_no_history_skips_rewriter(single_doc: Document) -> None:
    rewriter = MagicMock(spec=QueryRewriter)
    recommender = MovieRecommender(
        [StubRetriever([single_doc])], StubGenerator(), rewriter
    )
    await recommender.recommend("recommend a thriller", history=[])
    rewriter.rewrite.assert_not_called()


async def test_recommend_with_history_calls_rewriter(single_doc: Document) -> None:
    rewriter = StubRewriter()
    generator = MagicMock(spec=RecommendationGenerator)
    generator.generate.return_value = "answer"
    history = [HumanMessage(content="hi"), AIMessage(content="hello")]
    recommender = MovieRecommender([StubRetriever([single_doc])], generator, rewriter)
    await recommender.recommend("something slower", history=history)
    generator.generate.assert_called_once()
    _, context, _ = generator.generate.call_args[0]
    assert "Parasite" in context


async def test_recommend_merges_multiple_retrievers() -> None:
    doc_a = make_doc("tt001", "Parasite")
    doc_b = make_doc("tt002", "Oldboy")
    doc_dup = make_doc("tt001", "Parasite")  # same key as doc_a

    generator = MagicMock(spec=RecommendationGenerator)
    generator.generate.return_value = "answer"
    recommender = MovieRecommender(
        retrievers=[StubRetriever([doc_a]), StubRetriever([doc_b, doc_dup])],
        generator=generator,
        rewriter=StubRewriter(),
    )
    await recommender.recommend("question", history=[])
    _, context, _ = generator.generate.call_args[0]
    assert context.count("Content for Parasite") == 1  # deduplicated
    assert "Oldboy" in context


async def test_recommend_all_sections_for_same_movie_reach_generator() -> None:
    synopsis = make_doc("tt001", "Parasite", "synopsis")
    craft = make_doc("tt001", "Parasite", "enriched", "craft")
    meaning = make_doc("tt001", "Parasite", "enriched", "meaning")

    generator = MagicMock(spec=RecommendationGenerator)
    generator.generate.return_value = "answer"
    recommender = MovieRecommender(
        retrievers=[StubRetriever([synopsis, craft, meaning])],
        generator=generator,
        rewriter=StubRewriter(),
    )
    await recommender.recommend("question", history=[])
    _, context, _ = generator.generate.call_args[0]
    assert synopsis.page_content in context
    assert craft.page_content in context
    assert meaning.page_content in context


async def test_recommend_passes_original_question_to_generator(
    single_doc: Document,
) -> None:
    generator = MagicMock(spec=RecommendationGenerator)
    generator.generate.return_value = "answer"
    recommender = MovieRecommender(
        [StubRetriever([single_doc])], generator, StubRewriter()
    )
    await recommender.recommend("my question", history=[])
    question, _, _ = generator.generate.call_args[0]
    assert question == "my question"


async def test_recommend_returns_generator_output(single_doc: Document) -> None:
    generator = MagicMock(spec=RecommendationGenerator)
    generator.generate.return_value = "the final answer"
    recommender = MovieRecommender(
        [StubRetriever([single_doc])], generator, StubRewriter()
    )
    answer, _, _ = await recommender.recommend("question", history=[])
    assert answer == "the final answer"


async def test_recommend_strips_markers_from_returned_answer(
    single_doc: Document,
) -> None:
    generator = MagicMock(spec=RecommendationGenerator)
    generator.generate.return_value = (
        "1. **Parasite** (2019)\n<!-- imdb:tt001 -->\nGreat pick."
    )
    recommender = MovieRecommender(
        [StubRetriever([single_doc])], generator, StubRewriter()
    )
    answer, mentioned_ids, _ = await recommender.recommend("question", history=[])
    assert "imdb:" not in answer
    assert mentioned_ids == ["tt001"]


async def test_recommend_stream_yields_section_ready_at_stream_end(
    single_doc: Document,
) -> None:
    # A single, never-followed-by-another-heading section only closes once
    # the stream itself ends.
    generator = ChunkedStubGenerator(
        ["1. **Parasite** (2019)\n<!-- imdb:tt001 -->\nGreat pick."]
    )
    recommender = MovieRecommender(
        [StubRetriever([single_doc])], generator, StubRewriter()
    )
    streamed = await recommender.recommend_stream("question", history=[])

    events = [event async for event in streamed.events]

    assert len(events) == 1
    assert isinstance(events[0], SectionReady)
    assert events[0].imdb_id == "tt001"
    assert "imdb:" not in events[0].body_md
    assert "Great pick." in events[0].body_md
    assert streamed.imdb_ids == ["tt001"]


async def test_recommend_stream_yields_multiple_sections_in_order() -> None:
    doc_a = make_doc("tt001", "Parasite")
    doc_b = make_doc("tt002", "Oldboy")
    generator = ChunkedStubGenerator(
        [
            "1. **Parasite** (2019)\n<!-- imdb:tt001 -->\nGreat pick.\n\n",
            "2. **Oldboy** (2003)\n<!-- imdb:tt002 -->\nAlso great.",
        ]
    )
    recommender = MovieRecommender(
        [StubRetriever([doc_a, doc_b])], generator, StubRewriter()
    )
    streamed = await recommender.recommend_stream("question", history=[])

    events = [event async for event in streamed.events]
    sections = [e for e in events if isinstance(e, SectionReady)]

    assert [s.imdb_id for s in sections] == ["tt001", "tt002"]
    assert "Great pick." in sections[0].body_md
    assert "Also great." in sections[1].body_md
    assert streamed.imdb_ids == ["tt001", "tt002"]


async def test_recommend_stream_yields_intro_prose_as_text_delta(
    single_doc: Document,
) -> None:
    generator = ChunkedStubGenerator(
        [
            "Here are some picks:\n\n",
            "1. **Parasite** (2019)\n<!-- imdb:tt001 -->\nGreat pick.",
        ]
    )
    recommender = MovieRecommender(
        [StubRetriever([single_doc])], generator, StubRewriter()
    )
    streamed = await recommender.recommend_stream("question", history=[])

    events = [event async for event in streamed.events]

    assert isinstance(events[0], TextDelta)
    assert "Here are some picks" in events[0].text
    assert isinstance(events[1], SectionReady)


async def test_recommend_stream_resolves_marker_split_across_chunks(
    single_doc: Document,
) -> None:
    generator = ChunkedStubGenerator(
        ["1. **Parasite** (2019)\n<!-- imdb:t", "t001 -->\nGreat pick."]
    )
    recommender = MovieRecommender(
        [StubRetriever([single_doc])], generator, StubRewriter()
    )
    streamed = await recommender.recommend_stream("question", history=[])

    events = [event async for event in streamed.events]

    assert len(events) == 1
    assert isinstance(events[0], SectionReady)
    assert events[0].imdb_id == "tt001"
    assert "imdb:" not in events[0].body_md


async def test_recommend_stream_falls_back_to_title_match_without_marker(
    single_doc: Document,
) -> None:
    generator = ChunkedStubGenerator(["1. Parasite is a great pick."])
    recommender = MovieRecommender(
        [StubRetriever([single_doc])], generator, StubRewriter()
    )
    streamed = await recommender.recommend_stream("question", history=[])

    events = [event async for event in streamed.events]

    assert isinstance(events[0], SectionReady)
    assert events[0].imdb_id == "tt001"


async def test_recommend_stream_sets_clean_answer_after_completion(
    single_doc: Document,
) -> None:
    generator = ChunkedStubGenerator(
        ["1. **Parasite** (2019)\n<!-- imdb:tt001 -->\nGreat pick."]
    )
    recommender = MovieRecommender(
        [StubRetriever([single_doc])], generator, StubRewriter()
    )
    streamed = await recommender.recommend_stream("question", history=[])

    async for _event in streamed.events:
        pass

    assert "imdb:" not in streamed.answer
    assert "Great pick." in streamed.answer


async def test_recommend_omits_coverage_report_by_default(single_doc: Document) -> None:
    generator = MagicMock(spec=RecommendationGenerator)
    generator.generate.return_value = "the final answer"
    recommender = MovieRecommender(
        [StubRetriever([single_doc])], generator, StubRewriter()
    )
    _, _, coverage = await recommender.recommend("question", history=[])
    assert coverage is None


async def test_recommend_verbose_builds_coverage_report(single_doc: Document) -> None:
    generator = MagicMock(spec=RecommendationGenerator)
    generator.generate.return_value = "Parasite is a great pick."
    recommender = MovieRecommender(
        [StubRetriever([single_doc])], generator, StubRewriter()
    )
    _, _, coverage = await recommender.recommend("question", history=[], verbose=True)
    assert coverage is not None
    assert coverage.retriever_names == ["stub"]
    assert [e.title for e in coverage.recommended] == ["Parasite"]
    assert coverage.dropped == []
    assert coverage.recommended[0].sources == frozenset({"stub"})
