from unittest.mock import AsyncMock, MagicMock

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
def hyde_retriever() -> tuple[HyDEVectorRetriever, MagicMock, MagicMock, MagicMock]:
    mock_vector_store = MagicMock()
    mock_embeddings = MagicMock()
    mock_llm = MagicMock()
    retriever = HyDEVectorRetriever(mock_vector_store, mock_embeddings, mock_llm, k=8)
    # Bypass LangChain chain construction — test retrieve() logic directly
    mock_chain = MagicMock()
    retriever._chain = mock_chain
    mock_chain.ainvoke = AsyncMock(
        return_value="A tense detective thriller set in a rainy city."
    )
    mock_embeddings.aembed_documents = AsyncMock(return_value=[[0.1, 0.2, 0.3]])
    mock_vector_store.asimilarity_search_by_vector = AsyncMock(
        return_value=[make_doc("tt001", "Parasite")]
    )
    return retriever, mock_vector_store, mock_embeddings, mock_chain


async def test_hyde_retriever_generates_hypothetical_doc(
    hyde_retriever: tuple[HyDEVectorRetriever, MagicMock, MagicMock, MagicMock],
) -> None:
    retriever, _, _, mock_chain = hyde_retriever
    await retriever.retrieve("recommend a thriller")
    mock_chain.ainvoke.assert_called_once_with({"question": "recommend a thriller"})


async def test_hyde_retriever_embeds_hypothetical_doc(
    hyde_retriever: tuple[HyDEVectorRetriever, MagicMock, MagicMock, MagicMock],
) -> None:
    retriever, _, mock_embeddings, _ = hyde_retriever
    await retriever.retrieve("recommend a thriller")
    mock_embeddings.aembed_documents.assert_called_once_with(
        ["A tense detective thriller set in a rainy city."]
    )


async def test_hyde_retriever_searches_by_vector(
    hyde_retriever: tuple[HyDEVectorRetriever, MagicMock, MagicMock, MagicMock],
) -> None:
    retriever, mock_vector_store, _, _ = hyde_retriever
    await retriever.retrieve("recommend a thriller")
    call_kwargs = mock_vector_store.asimilarity_search_by_vector.call_args.kwargs
    assert call_kwargs["filter"] is not None
    assert mock_vector_store.asimilarity_search_by_vector.call_args[0][0] == [
        0.1,
        0.2,
        0.3,
    ]


async def test_hyde_retriever_filter_targets_enriched_embedding_type(
    hyde_retriever: tuple[HyDEVectorRetriever, MagicMock, MagicMock, MagicMock],
) -> None:
    retriever, mock_vector_store, _, _ = hyde_retriever
    await retriever.retrieve("recommend a thriller")
    f = mock_vector_store.asimilarity_search_by_vector.call_args.kwargs["filter"]
    assert f.must[0].match.value == "enriched"


async def test_hyde_retriever_returns_docs(
    hyde_retriever: tuple[HyDEVectorRetriever, MagicMock, MagicMock, MagicMock],
) -> None:
    retriever, _, _, _ = hyde_retriever
    docs = await retriever.retrieve("recommend a thriller")
    assert len(docs) == 1
    assert docs[0].metadata["imdb_id"] == "tt001"


async def test_hyde_retriever_respects_k() -> None:
    mock_vector_store = MagicMock()
    mock_vector_store.asimilarity_search_by_vector = AsyncMock(return_value=[])
    mock_embeddings = MagicMock()
    mock_embeddings.aembed_documents = AsyncMock(return_value=[[0.0]])
    retriever = HyDEVectorRetriever(
        mock_vector_store, mock_embeddings, MagicMock(), k=3
    )
    retriever._chain = MagicMock()
    retriever._chain.ainvoke = AsyncMock(return_value="profile")
    await retriever.retrieve("query")
    call_kwargs = mock_vector_store.asimilarity_search_by_vector.call_args.kwargs
    assert call_kwargs["k"] == 3


# --- LLMKnowledgeRetriever ---


@pytest.fixture
def doc_by_title() -> dict[str, Document]:
    return {
        "parasite": make_doc("tt001", "Parasite"),
        "oldboy": make_doc("tt002", "Oldboy"),
        "the handmaiden": make_doc("tt003", "The Handmaiden"),
    }


@pytest.fixture
def llm_retriever(
    doc_by_title: dict[str, Document],
) -> tuple[LLMKnowledgeRetriever, MagicMock]:
    retriever = LLMKnowledgeRetriever(MagicMock(), "- Parasite\n- Oldboy", doc_by_title)
    mock_chain = MagicMock()
    retriever._chain = mock_chain
    return retriever, mock_chain


async def test_llm_retriever_returns_matched_docs(
    llm_retriever: tuple[LLMKnowledgeRetriever, MagicMock],
) -> None:
    retriever, mock_chain = llm_retriever
    mock_chain.ainvoke = AsyncMock(return_value='["Parasite", "Oldboy"]')
    docs = await retriever.retrieve("dark Korean cinema")
    assert len(docs) == 2


async def test_llm_retriever_is_case_insensitive(
    llm_retriever: tuple[LLMKnowledgeRetriever, MagicMock],
) -> None:
    retriever, mock_chain = llm_retriever
    mock_chain.ainvoke = AsyncMock(return_value='["PARASITE", "OldBoy"]')
    docs = await retriever.retrieve("dark Korean cinema")
    assert len(docs) == 2


async def test_llm_retriever_strips_markdown_fences(
    llm_retriever: tuple[LLMKnowledgeRetriever, MagicMock],
) -> None:
    retriever, mock_chain = llm_retriever
    mock_chain.ainvoke = AsyncMock(return_value='```json\n["Parasite"]\n```')
    docs = await retriever.retrieve("query")
    assert len(docs) == 1
    assert docs[0].metadata["imdb_id"] == "tt001"


async def test_llm_retriever_handles_malformed_json_gracefully(
    llm_retriever: tuple[LLMKnowledgeRetriever, MagicMock],
) -> None:
    retriever, mock_chain = llm_retriever
    mock_chain.ainvoke = AsyncMock(return_value="Sorry, I cannot select movies.")
    docs = await retriever.retrieve("query")
    assert docs == []


async def test_llm_retriever_skips_unknown_titles(
    llm_retriever: tuple[LLMKnowledgeRetriever, MagicMock],
) -> None:
    retriever, mock_chain = llm_retriever
    mock_chain.ainvoke = AsyncMock(return_value='["Parasite", "Unknown Film 2099"]')
    docs = await retriever.retrieve("query")
    assert len(docs) == 1
    assert docs[0].metadata["imdb_id"] == "tt001"


async def test_llm_retriever_passes_question_and_movie_list(
    llm_retriever: tuple[LLMKnowledgeRetriever, MagicMock],
) -> None:
    retriever, mock_chain = llm_retriever
    mock_chain.ainvoke = AsyncMock(return_value="[]")
    await retriever.retrieve("something tense")
    call_args = mock_chain.ainvoke.call_args[0][0]
    assert call_args["question"] == "something tense"
    assert call_args["movie_list"] == "- Parasite\n- Oldboy"


# --- LLMEnrichmentRetriever ---


def make_enrichment_retriever(
    filter_by_type: bool = True, k: int = 8
) -> tuple[LLMEnrichmentRetriever, MagicMock, MagicMock]:
    mock_vector_store = MagicMock()
    mock_embeddings = MagicMock()
    mock_embeddings.aembed_documents = AsyncMock(return_value=[[0.1, 0.2, 0.3]])
    mock_vector_store.asimilarity_search_by_vector = AsyncMock(
        return_value=[make_doc("tt001", "Parasite")]
    )
    retriever = LLMEnrichmentRetriever(
        mock_vector_store, mock_embeddings, k=k, filter_by_type=filter_by_type
    )
    return retriever, mock_vector_store, mock_embeddings


async def test_enrichment_retriever_embeds_query_directly() -> None:
    retriever, _, mock_embeddings = make_enrichment_retriever()
    await retriever.retrieve("something Kubrickian")
    mock_embeddings.aembed_documents.assert_called_once_with(["something Kubrickian"])


async def test_enrichment_retriever_does_not_use_an_llm() -> None:
    # Unlike HyDE, there is no _chain — embedding is the only API call
    retriever, _, _ = make_enrichment_retriever()
    assert not hasattr(retriever, "_chain")


async def test_enrichment_retriever_passes_filter_when_filter_by_type_true() -> None:
    retriever, mock_vs, _ = make_enrichment_retriever(filter_by_type=True)
    await retriever.retrieve("query")
    call_kwargs = mock_vs.asimilarity_search_by_vector.call_args.kwargs
    assert call_kwargs["filter"] is not None


async def test_enrichment_retriever_filter_targets_enriched_embedding_type() -> None:
    retriever, mock_vs, _ = make_enrichment_retriever(filter_by_type=True)
    await retriever.retrieve("query")
    f = mock_vs.asimilarity_search_by_vector.call_args.kwargs["filter"]
    assert f.must[0].match.value == "enriched"


async def test_enrichment_retriever_no_filter_when_filter_by_type_false() -> None:
    retriever, mock_vs, _ = make_enrichment_retriever(filter_by_type=False)
    await retriever.retrieve("query")
    call_kwargs = mock_vs.asimilarity_search_by_vector.call_args.kwargs
    assert call_kwargs["filter"] is None


async def test_enrichment_retriever_respects_k() -> None:
    retriever, mock_vs, _ = make_enrichment_retriever(k=4)
    await retriever.retrieve("query")
    call_kwargs = mock_vs.asimilarity_search_by_vector.call_args.kwargs
    assert call_kwargs["k"] == 4


async def test_enrichment_retriever_returns_docs_from_vector_store() -> None:
    retriever, _, _ = make_enrichment_retriever()
    docs = await retriever.retrieve("something Kubrickian")
    assert len(docs) == 1
    assert docs[0].metadata["imdb_id"] == "tt001"


# --- DirectSynopsisRetriever ---


def make_synopsis_retriever(
    k: int = 8,
) -> tuple[DirectSynopsisRetriever, MagicMock, MagicMock]:
    mock_vector_store = MagicMock()
    mock_embeddings = MagicMock()
    mock_embeddings.aembed_documents = AsyncMock(return_value=[[0.1, 0.2, 0.3]])
    mock_vector_store.asimilarity_search_by_vector = AsyncMock(
        return_value=[make_doc("tt001", "Parasite")]
    )
    retriever = DirectSynopsisRetriever(mock_vector_store, mock_embeddings, k=k)
    return retriever, mock_vector_store, mock_embeddings


async def test_synopsis_retriever_embeds_query_directly() -> None:
    retriever, _, mock_embeddings = make_synopsis_retriever()
    await retriever.retrieve("something Tarkovsky-esque")
    mock_embeddings.aembed_documents.assert_called_once_with(
        ["something Tarkovsky-esque"]
    )


async def test_synopsis_retriever_does_not_use_an_llm() -> None:
    retriever, _, _ = make_synopsis_retriever()
    assert not hasattr(retriever, "_chain")


async def test_synopsis_retriever_passes_filter_targeting_synopsis_type() -> None:
    retriever, mock_vs, _ = make_synopsis_retriever()
    await retriever.retrieve("query")
    f = mock_vs.asimilarity_search_by_vector.call_args.kwargs["filter"]
    assert f.must[0].match.value == "synopsis"


async def test_synopsis_retriever_respects_k() -> None:
    retriever, mock_vs, _ = make_synopsis_retriever(k=5)
    await retriever.retrieve("query")
    assert mock_vs.asimilarity_search_by_vector.call_args.kwargs["k"] == 5


async def test_synopsis_retriever_returns_docs_from_vector_store() -> None:
    retriever, _, _ = make_synopsis_retriever()
    docs = await retriever.retrieve("a heist film")
    assert len(docs) == 1
    assert docs[0].metadata["imdb_id"] == "tt001"
