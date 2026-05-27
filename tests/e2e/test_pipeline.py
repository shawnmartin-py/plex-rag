import pytest

from app.adapters.generators import GeminiQueryRewriter, GeminiRecommendationGenerator
from app.adapters.retrievers import HyDEVectorRetriever, LLMEnrichmentRetriever, LLMKnowledgeRetriever
from app.domain.recommender import MovieRecommender
from app.services.recommendation import ConversationalRecommendationService
from tests.e2e.conftest import TEST_DOCS, StubLLM

DOC_BY_TITLE = {doc.metadata["title"].lower(): doc for doc in TEST_DOCS}
MOVIE_LIST = "\n".join(f"- {doc.metadata['title']} ({doc.metadata['year']})" for doc in TEST_DOCS)

HYDE_RESPONSE = "A poor family infiltrates the home of a wealthy household, leading to class conflict."
KNOWLEDGE_RESPONSE = '["Parasite", "Oldboy"]'
RECOMMENDATION_RESPONSE = "Based on your request, I recommend Parasite and Oldboy from your library."
REWRITER_RESPONSE = "Recommend a dark thriller with class themes, focusing on more recent films."


@pytest.fixture
def service(qdrant_store, stub_embeddings):
    hyde_llm = StubLLM(responses=[HYDE_RESPONSE])
    knowledge_llm = StubLLM(responses=[KNOWLEDGE_RESPONSE])
    generator_llm = StubLLM(responses=[RECOMMENDATION_RESPONSE])
    rewriter_llm = StubLLM(responses=[REWRITER_RESPONSE])

    recommender = MovieRecommender(
        retrievers=[
            HyDEVectorRetriever(qdrant_store, stub_embeddings, hyde_llm),
            LLMKnowledgeRetriever(knowledge_llm, MOVIE_LIST, DOC_BY_TITLE),
            # No enriched docs in the shared fixture store — retriever returns empty, pipeline degrades gracefully
            LLMEnrichmentRetriever(qdrant_store, stub_embeddings),
        ],
        generator=GeminiRecommendationGenerator(generator_llm),
        rewriter=GeminiQueryRewriter(rewriter_llm),
    )
    return ConversationalRecommendationService(recommender)


def test_first_question_returns_string_answer(service):
    answer = service.chat("recommend a dark thriller")
    assert isinstance(answer, str)
    assert len(answer) > 0


def test_first_question_returns_generator_response(service):
    answer = service.chat("recommend a dark thriller")
    assert answer == RECOMMENDATION_RESPONSE


def test_follow_up_question_uses_rewriter(qdrant_store, stub_embeddings):
    rewriter_llm = StubLLM(responses=[REWRITER_RESPONSE])
    hyde_llm = StubLLM(responses=[HYDE_RESPONSE])
    knowledge_llm = StubLLM(responses=[KNOWLEDGE_RESPONSE])
    generator_llm = StubLLM(responses=[RECOMMENDATION_RESPONSE, "Second answer"])

    recommender = MovieRecommender(
        retrievers=[
            HyDEVectorRetriever(qdrant_store, stub_embeddings, hyde_llm),
            LLMKnowledgeRetriever(knowledge_llm, MOVIE_LIST, DOC_BY_TITLE),
            LLMEnrichmentRetriever(qdrant_store, stub_embeddings),
        ],
        generator=GeminiRecommendationGenerator(generator_llm),
        rewriter=GeminiQueryRewriter(rewriter_llm),
    )
    svc = ConversationalRecommendationService(recommender)

    svc.chat("recommend a dark thriller")
    # The rewriter LLM should be invoked on the follow-up since history is non-empty
    # If rewriter wasn't called, the LLM index wouldn't have advanced
    svc.chat("something more recent from those?")
    assert rewriter_llm._index == 1


def test_history_is_accumulated_across_turns(service):
    service.chat("first question")
    service.chat("second question")
    # History has 2 exchanges (4 messages); verify via the service's internal state
    assert len(service._history) == 4


def test_knowledge_retriever_contributes_docs_to_context(qdrant_store, stub_embeddings):
    # Use a knowledge LLM that returns a title only in doc_by_title, not in RAG results
    # Since all vectors are identical, RAG can return any doc — we just verify the pipeline runs
    knowledge_llm = StubLLM(responses=['["The Handmaiden"]'])
    hyde_llm = StubLLM(responses=[HYDE_RESPONSE])

    captured_contexts = []

    class CapturingGenerator(GeminiRecommendationGenerator):
        def generate(self, question, context, history):
            captured_contexts.append(context)
            return "answer"

    recommender = MovieRecommender(
        retrievers=[
            HyDEVectorRetriever(qdrant_store, stub_embeddings, hyde_llm),
            LLMKnowledgeRetriever(knowledge_llm, MOVIE_LIST, DOC_BY_TITLE),
            LLMEnrichmentRetriever(qdrant_store, stub_embeddings),
        ],
        generator=CapturingGenerator(StubLLM(responses=["answer"])),
        rewriter=GeminiQueryRewriter(StubLLM(responses=[REWRITER_RESPONSE])),
    )
    svc = ConversationalRecommendationService(recommender)
    svc.chat("dark thriller with female lead")

    assert len(captured_contexts) == 1
    assert "The Handmaiden" in captured_contexts[0]
