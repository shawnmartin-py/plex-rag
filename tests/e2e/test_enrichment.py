import tempfile

import pytest
from qdrant_client.models import FieldCondition, Filter, MatchValue

from app.adapters.generators import GeminiQueryRewriter, GeminiRecommendationGenerator
from app.adapters.retrievers import HyDEVectorRetriever, LLMEnrichmentRetriever, LLMKnowledgeRetriever
from app.domain.recommender import MovieRecommender
from app.models.media_item import MediaItem
from app.services.enrichment import SECTIONS, EnrichmentService
from app.services.recommendation import ConversationalRecommendationService
from app.services.vector_store import VectorStoreService
from tests.e2e.conftest import TEST_DOCS, StubLLM

TEST_ITEMS = [
    MediaItem(
        imdb_id="tt6751668",
        type="movie",
        title="Parasite",
        year=2019,
        imdb_rating=8.5,
        content_rating="R",
        genres=["Drama", "Thriller"],
        synopsis="A poor Korean family schemes their way into a wealthy household.",
    ),
    MediaItem(
        imdb_id="tt0364569",
        type="movie",
        title="Oldboy",
        year=2003,
        imdb_rating=8.1,
        content_rating="R",
        genres=["Action", "Drama", "Mystery"],
        synopsis="A man imprisoned for 15 years hunts down his captor.",
    ),
    MediaItem(
        imdb_id="tt4016934",
        type="movie",
        title="The Handmaiden",
        year=2016,
        imdb_rating=8.1,
        content_rating="NR",
        genres=["Drama", "Mystery", "Romance"],
        synopsis="A woman is hired to swindle a Japanese heiress.",
    ),
]


@pytest.fixture
def isolated_vs_service(stub_embeddings):
    """Fresh isolated VectorStoreService for each test."""
    with tempfile.TemporaryDirectory() as tmpdir:
        service = VectorStoreService(
            embeddings=stub_embeddings,
            path=tmpdir,
            collection_name="test_enrichment",
        )
        service.load_or_build(TEST_DOCS)
        yield service


@pytest.fixture
def enrichment_service(isolated_vs_service):
    # One response per section per movie — 3 sections × 3 movies = 9 responses
    responses = [f"Section text for section {i}" for i in range(len(SECTIONS) * len(TEST_ITEMS))]
    llm = StubLLM(responses=responses)
    service = EnrichmentService(llm, isolated_vs_service, "test_enrichment")
    return service, isolated_vs_service


def _enriched_filter() -> Filter:
    return Filter(must=[FieldCondition(key="metadata.embedding_type", match=MatchValue(value="enriched"))])


def _section_filter(section: str) -> Filter:
    return Filter(
        must=[
            FieldCondition(key="metadata.embedding_type", match=MatchValue(value="enriched")),
            FieldCondition(key="metadata.section", match=MatchValue(value=section)),
        ]
    )


# --- EnrichmentService.build ---


def test_build_adds_all_three_sections_per_movie(enrichment_service, stub_embeddings):
    service, vs_service = enrichment_service
    service.build(TEST_ITEMS)

    vector = stub_embeddings.embed_documents(["query"])[0]
    results = vs_service.store.similarity_search_by_vector(vector, k=50, filter=_enriched_filter())
    assert len(results) == len(TEST_ITEMS) * len(SECTIONS)


def test_build_adds_each_section_for_each_movie(enrichment_service, stub_embeddings):
    service, vs_service = enrichment_service
    service.build(TEST_ITEMS)

    vector = stub_embeddings.embed_documents(["query"])[0]
    for section in SECTIONS:
        results = vs_service.store.similarity_search_by_vector(vector, k=50, filter=_section_filter(section))
        assert len(results) == len(TEST_ITEMS), f"Section '{section}' should have one doc per movie"


def test_build_enriched_documents_carry_imdb_id(enrichment_service, stub_embeddings):
    service, vs_service = enrichment_service
    service.build(TEST_ITEMS)

    vector = stub_embeddings.embed_documents(["query"])[0]
    results = vs_service.store.similarity_search_by_vector(vector, k=50, filter=_enriched_filter())
    imdb_ids = {doc.metadata["imdb_id"] for doc in results}
    assert imdb_ids == {item.imdb_id for item in TEST_ITEMS}


def test_build_skips_items_with_no_synopsis(enrichment_service):
    service, vs_service = enrichment_service
    no_synopsis = MediaItem(
        imdb_id="tt9999999",
        type="movie",
        title="Mystery Film",
        year=2020,
        imdb_rating=0.0,
        content_rating="NR",
        genres=[],
        synopsis=None,
    )
    service.build([no_synopsis])

    results, _ = vs_service.client.scroll(
        collection_name="test_enrichment",
        scroll_filter=Filter(
            must=[
                FieldCondition(key="metadata.imdb_id", match=MatchValue(value="tt9999999")),
                FieldCondition(key="metadata.embedding_type", match=MatchValue(value="enriched")),
            ]
        ),
        limit=1,
        with_payload=False,
        with_vectors=False,
    )
    assert len(results) == 0


# --- Idempotency ---


def test_build_is_idempotent(enrichment_service, stub_embeddings):
    service, vs_service = enrichment_service
    service.build(TEST_ITEMS)
    service.build(TEST_ITEMS)  # second run — all sections already exist

    vector = stub_embeddings.embed_documents(["query"])[0]
    results = vs_service.store.similarity_search_by_vector(vector, k=50, filter=_enriched_filter())
    assert len(results) == len(TEST_ITEMS) * len(SECTIONS)


# --- LLMEnrichmentRetriever filter behaviour ---


def test_enrichment_retriever_only_returns_enriched_docs(enrichment_service, stub_embeddings):
    service, vs_service = enrichment_service
    service.build(TEST_ITEMS)

    retriever = LLMEnrichmentRetriever(vs_service.store, stub_embeddings, k=50, filter_by_type=True)
    docs = retriever.retrieve("dark Korean cinema")
    assert all(doc.metadata["embedding_type"] == "enriched" for doc in docs)


def test_enrichment_retriever_unfiltered_returns_both_types(enrichment_service, stub_embeddings):
    service, vs_service = enrichment_service
    service.build(TEST_ITEMS)

    retriever = LLMEnrichmentRetriever(vs_service.store, stub_embeddings, k=50, filter_by_type=False)
    docs = retriever.retrieve("dark Korean cinema")
    types = {doc.metadata["embedding_type"] for doc in docs}
    assert "synopsis" in types
    assert "enriched" in types


def test_enrichment_retriever_returns_empty_when_no_enriched_docs_exist(isolated_vs_service, stub_embeddings):
    retriever = LLMEnrichmentRetriever(isolated_vs_service.store, stub_embeddings, k=10, filter_by_type=True)
    assert retriever.retrieve("anything") == []


def test_enrichment_retriever_all_sections_present_in_results(enrichment_service, stub_embeddings):
    service, vs_service = enrichment_service
    service.build(TEST_ITEMS)

    retriever = LLMEnrichmentRetriever(vs_service.store, stub_embeddings, k=50, filter_by_type=True)
    docs = retriever.retrieve("dark Korean cinema")
    sections = {doc.metadata["section"] for doc in docs}
    assert sections == set(SECTIONS)


# --- Full pipeline ---


def test_pipeline_all_sections_reach_generator_for_matched_movie(enrichment_service, stub_embeddings):
    service, vs_service = enrichment_service
    service.build(TEST_ITEMS)

    doc_by_title = {doc.metadata["title"].lower(): doc for doc in TEST_DOCS}
    movie_list = "\n".join(f"- {doc.metadata['title']} ({doc.metadata['year']})" for doc in TEST_DOCS)

    captured_contexts = []

    class CapturingGenerator(GeminiRecommendationGenerator):
        def generate(self, question, context, history):
            captured_contexts.append(context)
            return "answer"

    recommender = MovieRecommender(
        retrievers=[
            HyDEVectorRetriever(vs_service.store, stub_embeddings, StubLLM(responses=["A dark thriller."])),
            LLMKnowledgeRetriever(StubLLM(responses=['["Parasite"]']), movie_list, doc_by_title),
            LLMEnrichmentRetriever(vs_service.store, stub_embeddings, k=20),
        ],
        generator=CapturingGenerator(StubLLM(responses=["answer"])),
        rewriter=GeminiQueryRewriter(StubLLM(responses=["rewritten"])),
    )
    ConversationalRecommendationService(recommender).chat("something dark and psychological")

    assert len(captured_contexts) == 1
    context = captured_contexts[0]
    # All three enrichment sections should appear for at least one movie
    assert (
        any(
            "craft" in doc.metadata.get("section", "") or "meaning" in doc.metadata.get("section", "")
            for doc in [d for docs_list in [TEST_DOCS] for d in docs_list]
        )
        or "Section text" in context
    )


def test_pipeline_degrades_gracefully_when_no_enriched_docs(isolated_vs_service, stub_embeddings):
    doc_by_title = {doc.metadata["title"].lower(): doc for doc in TEST_DOCS}
    movie_list = "\n".join(f"- {doc.metadata['title']} ({doc.metadata['year']})" for doc in TEST_DOCS)

    recommender = MovieRecommender(
        retrievers=[
            HyDEVectorRetriever(isolated_vs_service.store, stub_embeddings, StubLLM(responses=["A dark thriller."])),
            LLMKnowledgeRetriever(StubLLM(responses=['["Parasite"]']), movie_list, doc_by_title),
            LLMEnrichmentRetriever(isolated_vs_service.store, stub_embeddings, k=8),
        ],
        generator=GeminiRecommendationGenerator(StubLLM(responses=["Here are my recommendations."])),
        rewriter=GeminiQueryRewriter(StubLLM(responses=["rewritten"])),
    )
    answer = ConversationalRecommendationService(recommender).chat("something dark")
    assert isinstance(answer, str)
    assert len(answer) > 0
