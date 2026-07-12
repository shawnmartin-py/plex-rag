from unittest.mock import MagicMock, patch

import pytest
from qdrant_client.models import Distance, VectorParams

from app.repositories.vector_store import (
    QdrantUnavailableError,
    connect_vector_store,
    load_synopsis_documents,
    load_synopsis_vectors,
    load_watch_history_points,
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
        "app.repositories.vector_store.QdrantClient",
        side_effect=ConnectionError("refused"),
    ):
        with pytest.raises(QdrantUnavailableError, match="Could not reach Qdrant"):
            connect_vector_store("http://localhost:6333", "media_items", MagicMock())


def test_connect_raises_when_collection_missing() -> None:
    mock_client = make_mock_client(exists=False)
    with patch("app.repositories.vector_store.QdrantClient", return_value=mock_client):
        with pytest.raises(QdrantUnavailableError, match="does not exist"):
            connect_vector_store("http://localhost:6333", "media_items", MagicMock())


def test_connect_raises_when_vector_size_mismatched() -> None:
    mock_client = make_mock_client(vector_size=1536)
    with patch("app.repositories.vector_store.QdrantClient", return_value=mock_client):
        with pytest.raises(QdrantUnavailableError, match="vector size 1536"):
            connect_vector_store("http://localhost:6333", "media_items", MagicMock())


def test_connect_returns_vector_store_when_healthy() -> None:
    mock_client = make_mock_client()
    with patch("app.repositories.vector_store.QdrantClient", return_value=mock_client):
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


# --- load_synopsis_vectors ---


def test_load_synopsis_vectors_builds_candidates_from_scroll_results() -> None:
    mock_point = MagicMock()
    mock_point.payload = {
        "metadata": {"imdb_id": "tt6751668", "imdb_rating": 8.5},
    }
    mock_point.vector = [0.1, 0.2, 0.3]
    mock_vector_store = MagicMock()
    mock_vector_store.client.scroll.return_value = ([mock_point], None)

    result = load_synopsis_vectors(mock_vector_store, "media_items")

    assert len(result) == 1
    assert result[0].imdb_id == "tt6751668"
    assert result[0].vector == [0.1, 0.2, 0.3]
    assert result[0].imdb_rating == 8.5


def test_load_synopsis_vectors_filters_by_synopsis_embedding_type() -> None:
    mock_vector_store = MagicMock()
    mock_vector_store.client.scroll.return_value = ([], None)

    load_synopsis_vectors(mock_vector_store, "media_items")

    call_kwargs = mock_vector_store.client.scroll.call_args.kwargs
    assert call_kwargs["scroll_filter"].must[0].match.value == "synopsis"
    assert call_kwargs["with_vectors"] is True


def test_load_synopsis_vectors_raises_on_non_flat_vector() -> None:
    mock_point = MagicMock()
    mock_point.payload = {"metadata": {"imdb_id": "tt0001", "imdb_rating": 7.0}}
    mock_point.vector = {"named": [0.1, 0.2]}
    mock_vector_store = MagicMock()
    mock_vector_store.client.scroll.return_value = ([mock_point], None)

    with pytest.raises(TypeError, match="tt0001"):
        load_synopsis_vectors(mock_vector_store, "media_items")


# --- load_watch_history_points ---


def test_load_watch_history_points_builds_points_from_scroll_results() -> None:
    mock_point = MagicMock()
    mock_point.payload = {
        "metadata": {
            "imdb_id": "tt3605418",
            "last_viewed_at": "2026-07-01T21:24:14",
        },
    }
    mock_point.vector = [0.4, 0.5]
    mock_vector_store = MagicMock()
    mock_vector_store.client.scroll.return_value = ([mock_point], None)

    result = load_watch_history_points(mock_vector_store, "watch_history")

    assert len(result) == 1
    assert result[0].imdb_id == "tt3605418"
    assert result[0].vector == [0.4, 0.5]
    assert result[0].last_viewed_at.isoformat() == "2026-07-01T21:24:14"
