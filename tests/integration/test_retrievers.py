from unittest.mock import MagicMock

import pytest
from langchain_core.documents import Document

from app.adapters.retrievers import (
    DirectSynopsisRetriever,
    HyDEVectorRetriever,
    LLMEnrichmentRetriever,
    LLMKnowledgeRetriever,
)


def make_doc(imdb_id: str, title: str) -> Document:
    return Document(page_content=f"Title: {title}", metadata={"imdb_id": imdb_id})


# --- HyDEVectorRetriever ---


@pytest.fixture
def hyde_retriever():
    mock_vector_store = MagicMock()
    mock_embeddings = MagicMock()
    mock_llm = MagicMock()
    retriever = HyDEVectorRetriever(mock_vector_store, mock_embeddings, mock_llm, k=8)
    # Bypass LangChain chain construction — test retrieve() logic directly
    retriever._chain = MagicMock()
    retriever._chain.invoke.return_value = "A tense detective thriller set in a rainy city."
    mock_embeddings.embed_documents.return_value = [[0.1, 0.2, 0.3]]
    mock_vector_store.similarity_search_by_vector.return_value = [make_doc("tt001", "Parasite")]
    return retriever, mock_vector_store, mock_embeddings


def test_hyde_retriever_generates_hypothetical_doc(hyde_retriever):
    retriever, _, _ = hyde_retriever
    retriever.retrieve("recommend a thriller")
    retriever._chain.invoke.assert_called_once_with({"question": "recommend a thriller"})


def test_hyde_retriever_embeds_hypothetical_doc(hyde_retriever):
    retriever, _, mock_embeddings = hyde_retriever
    retriever.retrieve("recommend a thriller")
    mock_embeddings.embed_documents.assert_called_once_with(["A tense detective thriller set in a rainy city."])


def test_hyde_retriever_searches_by_vector(hyde_retriever):
    retriever, mock_vector_store, _ = hyde_retriever
    retriever.retrieve("recommend a thriller")
    call_kwargs = mock_vector_store.similarity_search_by_vector.call_args.kwargs
    assert call_kwargs["filter"] is not None
    assert mock_vector_store.similarity_search_by_vector.call_args[0][0] == [0.1, 0.2, 0.3]


def test_hyde_retriever_filter_targets_enriched_embedding_type(hyde_retriever):
    retriever, mock_vector_store, _ = hyde_retriever
    retriever.retrieve("recommend a thriller")
    f = mock_vector_store.similarity_search_by_vector.call_args.kwargs["filter"]
    assert f.must[0].match.value == "enriched"


def test_hyde_retriever_returns_docs(hyde_retriever):
    retriever, _, _ = hyde_retriever
    docs = retriever.retrieve("recommend a thriller")
    assert len(docs) == 1
    assert docs[0].metadata["imdb_id"] == "tt001"


def test_hyde_retriever_respects_k():
    mock_vector_store = MagicMock()
    mock_embeddings = MagicMock()
    mock_embeddings.embed_documents.return_value = [[0.0]]
    retriever = HyDEVectorRetriever(mock_vector_store, mock_embeddings, MagicMock(), k=3)
    retriever._chain = MagicMock(return_value="profile")
    retriever._chain.invoke.return_value = "profile"
    retriever.retrieve("query")
    call_kwargs = mock_vector_store.similarity_search_by_vector.call_args.kwargs
    assert call_kwargs["k"] == 3


# --- LLMKnowledgeRetriever ---


@pytest.fixture
def doc_by_title():
    return {
        "parasite": make_doc("tt001", "Parasite"),
        "oldboy": make_doc("tt002", "Oldboy"),
        "the handmaiden": make_doc("tt003", "The Handmaiden"),
    }


@pytest.fixture
def llm_retriever(doc_by_title):
    retriever = LLMKnowledgeRetriever(MagicMock(), "- Parasite\n- Oldboy", doc_by_title)
    retriever._chain = MagicMock()
    return retriever


def test_llm_retriever_returns_matched_docs(llm_retriever):
    llm_retriever._chain.invoke.return_value = '["Parasite", "Oldboy"]'
    docs = llm_retriever.retrieve("dark Korean cinema")
    assert len(docs) == 2


def test_llm_retriever_is_case_insensitive(llm_retriever):
    llm_retriever._chain.invoke.return_value = '["PARASITE", "OldBoy"]'
    docs = llm_retriever.retrieve("dark Korean cinema")
    assert len(docs) == 2


def test_llm_retriever_strips_markdown_fences(llm_retriever):
    llm_retriever._chain.invoke.return_value = '```json\n["Parasite"]\n```'
    docs = llm_retriever.retrieve("query")
    assert len(docs) == 1
    assert docs[0].metadata["imdb_id"] == "tt001"


def test_llm_retriever_handles_malformed_json_gracefully(llm_retriever):
    llm_retriever._chain.invoke.return_value = "Sorry, I cannot select movies."
    docs = llm_retriever.retrieve("query")
    assert docs == []


def test_llm_retriever_skips_unknown_titles(llm_retriever):
    llm_retriever._chain.invoke.return_value = '["Parasite", "Unknown Film 2099"]'
    docs = llm_retriever.retrieve("query")
    assert len(docs) == 1
    assert docs[0].metadata["imdb_id"] == "tt001"


def test_llm_retriever_passes_question_and_movie_list(llm_retriever):
    llm_retriever._chain.invoke.return_value = "[]"
    llm_retriever.retrieve("something tense")
    call_args = llm_retriever._chain.invoke.call_args[0][0]
    assert call_args["question"] == "something tense"
    assert call_args["movie_list"] == "- Parasite\n- Oldboy"


# --- LLMEnrichmentRetriever ---


def make_enrichment_retriever(filter_by_type: bool = True, k: int = 8):
    mock_vector_store = MagicMock()
    mock_embeddings = MagicMock()
    mock_embeddings.embed_documents.return_value = [[0.1, 0.2, 0.3]]
    mock_vector_store.similarity_search_by_vector.return_value = [make_doc("tt001", "Parasite")]
    retriever = LLMEnrichmentRetriever(mock_vector_store, mock_embeddings, k=k, filter_by_type=filter_by_type)
    return retriever, mock_vector_store, mock_embeddings


def test_enrichment_retriever_embeds_query_directly():
    retriever, _, mock_embeddings = make_enrichment_retriever()
    retriever.retrieve("something Kubrickian")
    mock_embeddings.embed_documents.assert_called_once_with(["something Kubrickian"])


def test_enrichment_retriever_does_not_use_an_llm():
    # Unlike HyDE, there is no _chain — embedding is the only API call
    retriever, _, _ = make_enrichment_retriever()
    assert not hasattr(retriever, "_chain")


def test_enrichment_retriever_passes_filter_when_filter_by_type_true():
    retriever, mock_vs, _ = make_enrichment_retriever(filter_by_type=True)
    retriever.retrieve("query")
    call_kwargs = mock_vs.similarity_search_by_vector.call_args.kwargs
    assert call_kwargs["filter"] is not None


def test_enrichment_retriever_filter_targets_enriched_embedding_type():
    retriever, mock_vs, _ = make_enrichment_retriever(filter_by_type=True)
    retriever.retrieve("query")
    f = mock_vs.similarity_search_by_vector.call_args.kwargs["filter"]
    assert f.must[0].match.value == "enriched"


def test_enrichment_retriever_passes_no_filter_when_filter_by_type_false():
    retriever, mock_vs, _ = make_enrichment_retriever(filter_by_type=False)
    retriever.retrieve("query")
    call_kwargs = mock_vs.similarity_search_by_vector.call_args.kwargs
    assert call_kwargs["filter"] is None


def test_enrichment_retriever_respects_k():
    retriever, mock_vs, _ = make_enrichment_retriever(k=4)
    retriever.retrieve("query")
    call_kwargs = mock_vs.similarity_search_by_vector.call_args.kwargs
    assert call_kwargs["k"] == 4


def test_enrichment_retriever_returns_docs_from_vector_store():
    retriever, _, _ = make_enrichment_retriever()
    docs = retriever.retrieve("something Kubrickian")
    assert len(docs) == 1
    assert docs[0].metadata["imdb_id"] == "tt001"


# --- DirectSynopsisRetriever ---


def make_synopsis_retriever(k: int = 8):
    mock_vector_store = MagicMock()
    mock_embeddings = MagicMock()
    mock_embeddings.embed_documents.return_value = [[0.1, 0.2, 0.3]]
    mock_vector_store.similarity_search_by_vector.return_value = [make_doc("tt001", "Parasite")]
    retriever = DirectSynopsisRetriever(mock_vector_store, mock_embeddings, k=k)
    return retriever, mock_vector_store, mock_embeddings


def test_synopsis_retriever_embeds_query_directly():
    retriever, _, mock_embeddings = make_synopsis_retriever()
    retriever.retrieve("something Tarkovsky-esque")
    mock_embeddings.embed_documents.assert_called_once_with(["something Tarkovsky-esque"])


def test_synopsis_retriever_does_not_use_an_llm():
    retriever, _, _ = make_synopsis_retriever()
    assert not hasattr(retriever, "_chain")


def test_synopsis_retriever_passes_filter_targeting_synopsis_type():
    retriever, mock_vs, _ = make_synopsis_retriever()
    retriever.retrieve("query")
    f = mock_vs.similarity_search_by_vector.call_args.kwargs["filter"]
    assert f.must[0].match.value == "synopsis"


def test_synopsis_retriever_respects_k():
    retriever, mock_vs, _ = make_synopsis_retriever(k=5)
    retriever.retrieve("query")
    assert mock_vs.similarity_search_by_vector.call_args.kwargs["k"] == 5


def test_synopsis_retriever_returns_docs_from_vector_store():
    retriever, _, _ = make_synopsis_retriever()
    docs = retriever.retrieve("a heist film")
    assert len(docs) == 1
    assert docs[0].metadata["imdb_id"] == "tt001"
