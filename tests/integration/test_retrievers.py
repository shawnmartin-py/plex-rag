from unittest.mock import MagicMock

import pytest
from langchain_core.documents import Document

from app.adapters.retrievers import HyDEVectorRetriever, LLMKnowledgeRetriever


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
    mock_vector_store.similarity_search_by_vector.assert_called_once_with([0.1, 0.2, 0.3], k=8)


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
    retriever._chain = MagicMock(return_value="synopsis")
    retriever._chain.invoke.return_value = "synopsis"
    retriever.retrieve("query")
    mock_vector_store.similarity_search_by_vector.assert_called_once_with([0.0], k=3)


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
