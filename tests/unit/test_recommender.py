from collections.abc import AsyncIterator
from unittest.mock import MagicMock

import pytest

from app.domain.ports import (
    CandidateRetriever,
    ChatMessage,
    QueryRewriter,
    RecommendationCard,
    RecommendationGenerator,
    RecommendationResponse,
    RetrievedChunk,
    SectionReady,
    StreamEvent,
    TextDelta,
)
from app.domain.recommender import (
    MovieRecommender,
    _format_card_heading,
    _format_grouped,
    _group_docs,
)

_R = "retriever"  # generic retriever name for tests


def make_doc(
    tmdb_id: str,
    title: str,
    embedding_type: str = "synopsis",
    section: str | None = None,
) -> RetrievedChunk:
    metadata = {
        "tmdb_id": tmdb_id,
        "title": title,
        "year": 2020,
        "embedding_type": embedding_type,
    }
    if section is not None:
        metadata["section"] = section
    return RetrievedChunk(
        page_content=f"Content for {title} ({embedding_type}/{section})",
        metadata=metadata,
    )


# --- _group_docs ---


def test_group_docs_groups_by_tmdb_id() -> None:
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


def test_group_docs_dedup_key_is_tmdb_id_type_section() -> None:
    # Same tmdb_id + type + section = duplicate even if retrieved by
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


def test_format_grouped_includes_tmdb_id_in_header() -> None:
    doc = make_doc("tt001", "Parasite", "synopsis")
    result = _format_grouped({"tt001": [doc]})
    assert "tt001" in result


# --- _format_card_heading ---


def test_format_card_heading_includes_index_title_year() -> None:
    grouped = {"tt001": [make_doc("tt001", "Parasite")]}
    assert _format_card_heading(1, "tt001", grouped) == "1. **Parasite** (2020)"


def test_format_card_heading_uses_given_index() -> None:
    grouped = {"tt001": [make_doc("tt001", "Parasite")]}
    assert _format_card_heading(3, "tt001", grouped).startswith("3.")


# --- MovieRecommender ---


class StubRetriever(CandidateRetriever):
    name = "stub"

    def __init__(self, docs: list[RetrievedChunk]) -> None:
        self._docs = docs

    async def retrieve(self, query: str) -> list[RetrievedChunk]:
        return self._docs


class StubRewriter(QueryRewriter):
    async def rewrite(self, question: str, history: list[ChatMessage]) -> str:
        return f"rewritten: {question}"


class StubGenerator(RecommendationGenerator):
    async def generate(
        self, question: str, context: str, history: list[ChatMessage]
    ) -> RecommendationResponse:
        return RecommendationResponse(
            cards=[
                RecommendationCard(tmdb_id="tt001", body_md=f"answer for: {question}")
            ]
        )

    async def stream(
        self, question: str, context: str, history: list[ChatMessage]
    ) -> AsyncIterator[StreamEvent]:
        yield SectionReady(tmdb_id="tt001", body_md=f"answer for: {question}")


class EventStubGenerator(RecommendationGenerator):
    """Streams a pre-set sequence of already-decided events. Boundary
    detection (deciding when a card is "done") is now the real generator's
    job — see GeminiRecommendationGenerator.stream, covered separately in
    tests/integration/test_generators.py — so MovieRecommender only needs to
    be tested against events it might receive, not against raw text/JSON."""

    def __init__(
        self,
        events: list[StreamEvent],
        result: RecommendationResponse | None = None,
    ) -> None:
        self._events = events
        self._result = result if result is not None else RecommendationResponse()

    async def generate(
        self, question: str, context: str, history: list[ChatMessage]
    ) -> RecommendationResponse:
        return self._result

    async def stream(
        self, question: str, context: str, history: list[ChatMessage]
    ) -> AsyncIterator[StreamEvent]:
        for event in self._events:
            yield event


@pytest.fixture
def single_doc() -> RetrievedChunk:
    return make_doc("tt001", "Parasite")


async def test_recommend_with_no_history_skips_rewriter(
    single_doc: RetrievedChunk,
) -> None:
    rewriter = MagicMock(spec=QueryRewriter)
    recommender = MovieRecommender(
        [StubRetriever([single_doc])], StubGenerator(), rewriter
    )
    await recommender.recommend("recommend a thriller", history=[])
    rewriter.rewrite.assert_not_called()


async def test_recommend_with_history_calls_rewriter(
    single_doc: RetrievedChunk,
) -> None:
    rewriter = StubRewriter()
    generator = MagicMock(spec=RecommendationGenerator)
    generator.generate.return_value = RecommendationResponse()
    history = [
        ChatMessage(role="human", content="hi"),
        ChatMessage(role="ai", content="hello"),
    ]
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
    generator.generate.return_value = RecommendationResponse()
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
    generator.generate.return_value = RecommendationResponse()
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
    single_doc: RetrievedChunk,
) -> None:
    generator = MagicMock(spec=RecommendationGenerator)
    generator.generate.return_value = RecommendationResponse()
    recommender = MovieRecommender(
        [StubRetriever([single_doc])], generator, StubRewriter()
    )
    await recommender.recommend("my question", history=[])
    question, _, _ = generator.generate.call_args[0]
    assert question == "my question"


async def test_recommend_returns_answer_built_from_cards(
    single_doc: RetrievedChunk,
) -> None:
    generator = MagicMock(spec=RecommendationGenerator)
    generator.generate.return_value = RecommendationResponse(
        intro="Here's a pick:",
        cards=[RecommendationCard(tmdb_id="tt001", body_md="Great film.")],
    )
    recommender = MovieRecommender(
        [StubRetriever([single_doc])], generator, StubRewriter()
    )
    answer, mentioned_ids, _ = await recommender.recommend("question", history=[])
    assert "Here's a pick:" in answer
    assert "1. **Parasite** (2020)" in answer
    assert "Great film." in answer
    assert mentioned_ids == ["tt001"]


async def test_recommend_drops_hallucinated_tmdb_id(single_doc: RetrievedChunk) -> None:
    generator = MagicMock(spec=RecommendationGenerator)
    generator.generate.return_value = RecommendationResponse(
        cards=[
            RecommendationCard(tmdb_id="tt001", body_md="Real pick."),
            RecommendationCard(tmdb_id="tt999", body_md="Invented film."),
        ]
    )
    recommender = MovieRecommender(
        [StubRetriever([single_doc])], generator, StubRewriter()
    )
    answer, mentioned_ids, _ = await recommender.recommend("question", history=[])
    assert mentioned_ids == ["tt001"]
    assert "Invented film." not in answer


async def test_recommend_uses_closing_note_when_nothing_fits(
    single_doc: RetrievedChunk,
) -> None:
    generator = MagicMock(spec=RecommendationGenerator)
    generator.generate.return_value = RecommendationResponse(
        closing_note="Nothing here really fits that request."
    )
    recommender = MovieRecommender(
        [StubRetriever([single_doc])], generator, StubRewriter()
    )
    answer, mentioned_ids, _ = await recommender.recommend("question", history=[])
    assert answer == "Nothing here really fits that request."
    assert mentioned_ids == []


async def test_recommend_stream_yields_section_ready(
    single_doc: RetrievedChunk,
) -> None:
    generator = EventStubGenerator(
        [SectionReady(tmdb_id="tt001", body_md="Great pick.")]
    )
    recommender = MovieRecommender(
        [StubRetriever([single_doc])], generator, StubRewriter()
    )
    streamed = await recommender.recommend_stream("question", history=[])

    events = [event async for event in streamed.events]

    assert len(events) == 1
    assert isinstance(events[0], SectionReady)
    assert events[0].tmdb_id == "tt001"
    assert events[0].body_md == "Great pick."
    assert streamed.tmdb_ids == ["tt001"]


async def test_recommend_stream_yields_multiple_sections_in_order() -> None:
    doc_a = make_doc("tt001", "Parasite")
    doc_b = make_doc("tt002", "Oldboy")
    generator = EventStubGenerator(
        [
            SectionReady(tmdb_id="tt001", body_md="Great pick."),
            SectionReady(tmdb_id="tt002", body_md="Also great."),
        ]
    )
    recommender = MovieRecommender(
        [StubRetriever([doc_a, doc_b])], generator, StubRewriter()
    )
    streamed = await recommender.recommend_stream("question", history=[])

    events = [event async for event in streamed.events]
    sections = [e for e in events if isinstance(e, SectionReady)]

    assert [s.tmdb_id for s in sections] == ["tt001", "tt002"]
    assert streamed.tmdb_ids == ["tt001", "tt002"]


async def test_recommend_stream_yields_intro_as_text_delta(
    single_doc: RetrievedChunk,
) -> None:
    generator = EventStubGenerator(
        [
            TextDelta(text="Here are some picks:"),
            SectionReady(tmdb_id="tt001", body_md="Great pick."),
        ]
    )
    recommender = MovieRecommender(
        [StubRetriever([single_doc])], generator, StubRewriter()
    )
    streamed = await recommender.recommend_stream("question", history=[])

    events = [event async for event in streamed.events]

    assert isinstance(events[0], TextDelta)
    assert events[0].text == "Here are some picks:"
    assert isinstance(events[1], SectionReady)


async def test_recommend_stream_drops_hallucinated_card(
    single_doc: RetrievedChunk,
) -> None:
    generator = EventStubGenerator(
        [
            SectionReady(tmdb_id="tt999", body_md="Invented."),
            SectionReady(tmdb_id="tt001", body_md="Real pick."),
        ]
    )
    recommender = MovieRecommender(
        [StubRetriever([single_doc])], generator, StubRewriter()
    )
    streamed = await recommender.recommend_stream("question", history=[])

    events = [event async for event in streamed.events]

    assert len(events) == 1
    assert isinstance(events[0], SectionReady)
    assert events[0].tmdb_id == "tt001"
    assert streamed.tmdb_ids == ["tt001"]


async def test_recommend_stream_sets_answer_with_synthesized_heading(
    single_doc: RetrievedChunk,
) -> None:
    generator = EventStubGenerator(
        [SectionReady(tmdb_id="tt001", body_md="Great pick.")]
    )
    recommender = MovieRecommender(
        [StubRetriever([single_doc])], generator, StubRewriter()
    )
    streamed = await recommender.recommend_stream("question", history=[])

    async for _event in streamed.events:
        pass

    assert "1. **Parasite** (2020)" in streamed.answer
    assert "Great pick." in streamed.answer


async def test_recommend_stream_answer_includes_text_deltas_in_order(
    single_doc: RetrievedChunk,
) -> None:
    generator = EventStubGenerator(
        [
            TextDelta(text="Intro text."),
            SectionReady(tmdb_id="tt001", body_md="Great pick."),
            TextDelta(text="Closing text."),
        ]
    )
    recommender = MovieRecommender(
        [StubRetriever([single_doc])], generator, StubRewriter()
    )
    streamed = await recommender.recommend_stream("question", history=[])

    async for _event in streamed.events:
        pass

    assert streamed.answer == (
        "Intro text.\n\n1. **Parasite** (2020)\nGreat pick.\n\nClosing text."
    )


async def test_recommend_omits_coverage_report_by_default(
    single_doc: RetrievedChunk,
) -> None:
    generator = MagicMock(spec=RecommendationGenerator)
    generator.generate.return_value = RecommendationResponse(
        cards=[RecommendationCard(tmdb_id="tt001", body_md="Great pick.")]
    )
    recommender = MovieRecommender(
        [StubRetriever([single_doc])], generator, StubRewriter()
    )
    _, _, coverage = await recommender.recommend("question", history=[])
    assert coverage is None


async def test_recommend_verbose_builds_coverage_report(
    single_doc: RetrievedChunk,
) -> None:
    generator = MagicMock(spec=RecommendationGenerator)
    generator.generate.return_value = RecommendationResponse(
        cards=[RecommendationCard(tmdb_id="tt001", body_md="Great pick.")]
    )
    recommender = MovieRecommender(
        [StubRetriever([single_doc])], generator, StubRewriter()
    )
    _, _, coverage = await recommender.recommend("question", history=[], verbose=True)
    assert coverage is not None
    assert coverage.retriever_names == ["stub"]
    assert [e.title for e in coverage.recommended] == ["Parasite"]
    assert coverage.dropped == []
    assert coverage.recommended[0].sources == frozenset({"stub"})
