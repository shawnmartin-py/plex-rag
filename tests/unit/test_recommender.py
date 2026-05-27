from unittest.mock import MagicMock

import pytest
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage

from app.domain.ports import CandidateRetriever, QueryRewriter, RecommendationGenerator
from app.domain.recommender import MovieRecommender, _format_grouped, _group_docs

_R = "retriever"  # generic retriever name for tests


def make_doc(imdb_id: str, title: str, embedding_type: str = "synopsis", section: str | None = None) -> Document:
    metadata = {"imdb_id": imdb_id, "title": title, "year": 2020, "embedding_type": embedding_type}
    if section is not None:
        metadata["section"] = section
    return Document(page_content=f"Content for {title} ({embedding_type}/{section})", metadata=metadata)


# --- _group_docs ---


def test_group_docs_groups_by_imdb_id():
    synopsis = make_doc("tt001", "Parasite", "synopsis")
    craft = make_doc("tt001", "Parasite", "enriched", "craft")
    grouped, _ = _group_docs([(_R, [synopsis, craft])])
    assert len(grouped["tt001"]) == 2


def test_group_docs_deduplicates_exact_same_doc():
    doc = make_doc("tt001", "Parasite", "synopsis")
    grouped, _ = _group_docs([(_R, [doc]), (_R, [doc])])
    assert len(grouped["tt001"]) == 1


def test_group_docs_keeps_different_sections_for_same_movie():
    synopsis = make_doc("tt001", "Parasite", "synopsis")
    craft = make_doc("tt001", "Parasite", "enriched", "craft")
    meaning = make_doc("tt001", "Parasite", "enriched", "meaning")
    grouped, _ = _group_docs([(_R, [synopsis]), (_R, [craft]), (_R, [meaning])])
    assert len(grouped["tt001"]) == 3


def test_group_docs_dedup_key_is_imdb_id_type_section():
    # Same imdb_id + type + section = duplicate even if retrieved by different retrievers
    doc1 = make_doc("tt001", "Parasite", "enriched", "craft")
    doc2 = make_doc("tt001", "Parasite", "enriched", "craft")
    grouped, _ = _group_docs([("r1", [doc1]), ("r2", [doc2])])
    assert len(grouped["tt001"]) == 1


def test_group_docs_different_movies_stay_separate():
    a = make_doc("tt001", "Parasite", "synopsis")
    b = make_doc("tt002", "Oldboy", "synopsis")
    grouped, _ = _group_docs([(_R, [a, b])])
    assert "tt001" in grouped
    assert "tt002" in grouped
    assert len(grouped) == 2


def test_group_docs_synopsis_and_enriched_for_same_movie_are_both_kept():
    synopsis = make_doc("tt001", "Parasite", "synopsis")
    craft = make_doc("tt001", "Parasite", "enriched", "craft")
    grouped, _ = _group_docs([(_R, [synopsis]), (_R, [craft])])
    types = {doc.metadata["embedding_type"] for doc in grouped["tt001"]}
    assert "synopsis" in types
    assert "enriched" in types


def test_group_docs_empty_candidate_sets():
    grouped, sources = _group_docs([(_R, []), (_R, [])])
    assert grouped == {}
    assert sources == {}


def test_group_docs_preserves_all_movies_across_sets():
    a = make_doc("tt001", "Parasite", "synopsis")
    b = make_doc("tt002", "Oldboy", "synopsis")
    c = make_doc("tt003", "The Handmaiden", "synopsis")
    grouped, _ = _group_docs([(_R, [a]), (_R, [b]), (_R, [c])])
    assert set(grouped.keys()) == {"tt001", "tt002", "tt003"}


def test_group_docs_tracks_sources_per_retriever():
    doc_a = make_doc("tt001", "Parasite", "synopsis")
    doc_b = make_doc("tt001", "Parasite", "enriched", "craft")
    grouped, sources = _group_docs([("r1", [doc_a]), ("r2", [doc_b])])
    assert sources["tt001"] == {"r1", "r2"}


def test_group_docs_source_deduplication_does_not_inflate_retriever_set():
    doc = make_doc("tt001", "Parasite", "synopsis")
    _, sources = _group_docs([("r1", [doc]), ("r1", [doc])])
    assert sources["tt001"] == {"r1"}


# --- _format_grouped ---


def test_format_grouped_includes_movie_title_header():
    doc = make_doc("tt001", "Parasite", "synopsis")
    result = _format_grouped({"tt001": [doc]})
    assert "Parasite" in result


def test_format_grouped_includes_year_in_header():
    doc = make_doc("tt001", "Parasite", "synopsis")
    result = _format_grouped({"tt001": [doc]})
    assert "2020" in result


def test_format_grouped_includes_all_chunk_content():
    synopsis = make_doc("tt001", "Parasite", "synopsis")
    craft = make_doc("tt001", "Parasite", "enriched", "craft")
    result = _format_grouped({"tt001": [synopsis, craft]})
    assert synopsis.page_content in result
    assert craft.page_content in result


def test_format_grouped_separates_movies_with_delimiter():
    a = make_doc("tt001", "Parasite", "synopsis")
    b = make_doc("tt002", "Oldboy", "synopsis")
    result = _format_grouped({"tt001": [a], "tt002": [b]})
    assert "---" in result


def test_format_grouped_orders_synopsis_before_enrichment():
    craft = make_doc("tt001", "Parasite", "enriched", "craft")
    synopsis = make_doc("tt001", "Parasite", "synopsis")
    # Pass craft first — format should still put synopsis first
    result = _format_grouped({"tt001": [craft, synopsis]})
    assert result.index(synopsis.page_content) < result.index(craft.page_content)


def test_format_grouped_orders_enrichment_sections_craft_meaning_context():
    context = make_doc("tt001", "Parasite", "enriched", "context")
    craft = make_doc("tt001", "Parasite", "enriched", "craft")
    meaning = make_doc("tt001", "Parasite", "enriched", "meaning")
    result = _format_grouped({"tt001": [context, meaning, craft]})
    assert result.index(craft.page_content) < result.index(meaning.page_content)
    assert result.index(meaning.page_content) < result.index(context.page_content)


def test_format_grouped_empty_grouped():
    assert _format_grouped({}) == ""


# --- MovieRecommender ---


class StubRetriever(CandidateRetriever):
    name = "stub"

    def __init__(self, docs: list[Document]):
        self._docs = docs

    def retrieve(self, query: str) -> list[Document]:
        return self._docs


class StubRewriter(QueryRewriter):
    def rewrite(self, question: str, history) -> str:
        return f"rewritten: {question}"


class StubGenerator(RecommendationGenerator):
    def generate(self, question: str, context: str, history) -> str:
        return f"answer for: {question}"


@pytest.fixture
def single_doc():
    return make_doc("tt001", "Parasite")


def test_recommend_with_no_history_skips_rewriter(single_doc):
    rewriter = MagicMock(spec=QueryRewriter)
    recommender = MovieRecommender([StubRetriever([single_doc])], StubGenerator(), rewriter)
    recommender.recommend("recommend a thriller", history=[])
    rewriter.rewrite.assert_not_called()


def test_recommend_with_history_calls_rewriter(single_doc):
    rewriter = StubRewriter()
    generator = MagicMock(spec=RecommendationGenerator)
    generator.generate.return_value = "answer"
    history = [HumanMessage(content="hi"), AIMessage(content="hello")]
    recommender = MovieRecommender([StubRetriever([single_doc])], generator, rewriter)
    recommender.recommend("something slower", history=history)
    generator.generate.assert_called_once()
    _, context, _ = generator.generate.call_args[0]
    assert "Parasite" in context


def test_recommend_merges_multiple_retrievers():
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
    recommender.recommend("question", history=[])
    _, context, _ = generator.generate.call_args[0]
    assert context.count("Content for Parasite") == 1  # deduplicated
    assert "Oldboy" in context


def test_recommend_all_sections_for_same_movie_reach_generator():
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
    recommender.recommend("question", history=[])
    _, context, _ = generator.generate.call_args[0]
    assert synopsis.page_content in context
    assert craft.page_content in context
    assert meaning.page_content in context


def test_recommend_passes_original_question_to_generator(single_doc):
    generator = MagicMock(spec=RecommendationGenerator)
    generator.generate.return_value = "answer"
    recommender = MovieRecommender([StubRetriever([single_doc])], generator, StubRewriter())
    recommender.recommend("my question", history=[])
    question, _, _ = generator.generate.call_args[0]
    assert question == "my question"


def test_recommend_returns_generator_output(single_doc):
    generator = MagicMock(spec=RecommendationGenerator)
    generator.generate.return_value = "the final answer"
    recommender = MovieRecommender([StubRetriever([single_doc])], generator, StubRewriter())
    assert recommender.recommend("question", history=[]) == "the final answer"
