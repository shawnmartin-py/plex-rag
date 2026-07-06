from unittest.mock import MagicMock, patch

import pytest
from qdrant_client.models import Distance, VectorParams

from app.services.recommender_vector_store import (
    QdrantUnavailableError,
    connect_vector_store,
    load_synopsis_documents,
)
from tests.e2e.conftest import StubEmbeddings


def make_mock_client(*, exists: bool = True, vector_size: int = 3072) -> MagicMock:
    client = MagicMock()
    client.collection_exists.return_value = exists
    client.get_collection.return_value.config.params.vectors = VectorParams(
        size=vector_size, distance=Distance.COSINE
    )
    return client


# --- connect_vector_store ---


def test_connect_raises_when_server_unreachable() -> None:
    with patch(
        "app.services.recommender_vector_store.QdrantClient",
        side_effect=ConnectionError("refused"),
    ):
        with pytest.raises(QdrantUnavailableError, match="Could not reach Qdrant"):
            connect_vector_store("http://localhost:6333", "media_items", MagicMock())


def test_connect_raises_when_collection_missing() -> None:
    mock_client = make_mock_client(exists=False)
    with patch(
        "app.services.recommender_vector_store.QdrantClient", return_value=mock_client
    ):
        with pytest.raises(QdrantUnavailableError, match="does not exist"):
            connect_vector_store("http://localhost:6333", "media_items", MagicMock())


def test_connect_raises_when_vector_size_mismatched() -> None:
    mock_client = make_mock_client(vector_size=1536)
    with patch(
        "app.services.recommender_vector_store.QdrantClient", return_value=mock_client
    ):
        with pytest.raises(QdrantUnavailableError, match="vector size 1536"):
            connect_vector_store("http://localhost:6333", "media_items", MagicMock())


def test_connect_returns_vector_store_when_healthy() -> None:
    mock_client = make_mock_client()
    with patch(
        "app.services.recommender_vector_store.QdrantClient", return_value=mock_client
    ):
        store = connect_vector_store(
            "http://localhost:6333", "media_items", StubEmbeddings()
        )
    assert store is not None


# --- load_synopsis_documents ---


def test_load_synopsis_documents_builds_documents_from_scroll_results() -> None:
    mock_point = MagicMock()
    mock_point.payload = {
        "page_content": "Title: Parasite",
        "metadata": {"imdb_id": "tt6751668", "title": "Parasite"},
    }
    mock_vector_store = MagicMock()
    mock_vector_store.client.scroll.return_value = ([mock_point], None)

    docs = load_synopsis_documents(mock_vector_store, "media_items")

    assert len(docs) == 1
    assert docs[0].page_content == "Title: Parasite"
    assert docs[0].metadata["imdb_id"] == "tt6751668"


def test_load_synopsis_documents_filters_by_synopsis_embedding_type() -> None:
    mock_vector_store = MagicMock()
    mock_vector_store.client.scroll.return_value = ([], None)

    load_synopsis_documents(mock_vector_store, "media_items")

    call_kwargs = mock_vector_store.client.scroll.call_args.kwargs
    assert call_kwargs["scroll_filter"].must[0].match.value == "synopsis"
    assert call_kwargs["collection_name"] == "media_items"
