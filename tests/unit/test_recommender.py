from unittest.mock import MagicMock

import pytest
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage

from app.domain.ports import CandidateRetriever, QueryRewriter, RecommendationGenerator
from app.domain.recommender import MovieRecommender, _format_docs, _merge_docs


def make_doc(imdb_id: str, title: str) -> Document:
    return Document(page_content=f"Title: {title}", metadata={"imdb_id": imdb_id, "title": title})


# --- _format_docs ---


def test_format_docs_single():
    docs = [make_doc("tt001", "Parasite")]
    assert _format_docs(docs) == "Title: Parasite"


def test_format_docs_multiple():
    docs = [make_doc("tt001", "Parasite"), make_doc("tt002", "Oldboy")]
    result = _format_docs(docs)
    assert "Title: Parasite" in result
    assert "Title: Oldboy" in result
    assert "\n\n---\n\n" in result


def test_format_docs_empty():
    assert _format_docs([]) == ""


# --- _merge_docs ---


def test_merge_docs_deduplicates_across_sources():
    doc = make_doc("tt001", "Parasite")
    result = _merge_docs([[doc], [doc]])
    assert len(result) == 1


def test_merge_docs_preserves_rag_order():
    rag = [make_doc("tt001", "A"), make_doc("tt002", "B")]
    llm = [make_doc("tt003", "C"), make_doc("tt001", "A")]  # tt001 is a duplicate
    result = _merge_docs([rag, llm])
    assert [d.metadata["imdb_id"] for d in result] == ["tt001", "tt002", "tt003"]


def test_merge_docs_empty_sources():
    assert _merge_docs([[], []]) == []


def test_merge_docs_one_empty_source():
    docs = [make_doc("tt001", "Parasite")]
    assert _merge_docs([docs, []]) == docs
    assert _merge_docs([[], docs]) == docs


# --- MovieRecommender ---


class StubRetriever(CandidateRetriever):
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
    retriever = StubRetriever([single_doc])
    generator = StubGenerator()

    recommender = MovieRecommender([retriever], generator, rewriter)
    recommender.recommend("recommend a thriller", history=[])

    rewriter.rewrite.assert_not_called()


def test_recommend_with_history_calls_rewriter(single_doc):
    rewriter = StubRewriter()
    retriever = StubRetriever([single_doc])
    generator = MagicMock(spec=RecommendationGenerator)
    generator.generate.return_value = "answer"

    history = [HumanMessage(content="hi"), AIMessage(content="hello")]
    recommender = MovieRecommender([retriever], generator, rewriter)
    recommender.recommend("something slower", history=history)

    generator.generate.assert_called_once()
    _, context, _ = generator.generate.call_args[0]
    assert "Parasite" in context


def test_recommend_merges_multiple_retrievers():
    doc_a = make_doc("tt001", "Parasite")
    doc_b = make_doc("tt002", "Oldboy")
    doc_shared = make_doc("tt001", "Parasite")  # duplicate of doc_a

    generator = MagicMock(spec=RecommendationGenerator)
    generator.generate.return_value = "answer"

    recommender = MovieRecommender(
        retrievers=[StubRetriever([doc_a]), StubRetriever([doc_b, doc_shared])],
        generator=generator,
        rewriter=StubRewriter(),
    )
    recommender.recommend("question", history=[])

    _, context, _ = generator.generate.call_args[0]
    assert context.count("Title: Parasite") == 1  # deduplicated
    assert "Title: Oldboy" in context


def test_recommend_passes_original_question_to_generator(single_doc):
    generator = MagicMock(spec=RecommendationGenerator)
    generator.generate.return_value = "answer"

    recommender = MovieRecommender(
        retrievers=[StubRetriever([single_doc])],
        generator=generator,
        rewriter=StubRewriter(),
    )
    recommender.recommend("my question", history=[])

    question, _, _ = generator.generate.call_args[0]
    assert question == "my question"


def test_recommend_returns_generator_output(single_doc):
    generator = MagicMock(spec=RecommendationGenerator)
    generator.generate.return_value = "the final answer"

    recommender = MovieRecommender([StubRetriever([single_doc])], generator, StubRewriter())
    result = recommender.recommend("question", history=[])

    assert result == "the final answer"
