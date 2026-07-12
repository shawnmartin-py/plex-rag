from unittest.mock import MagicMock, patch

import pytest

from app.bootstrap import build_diversity_service
from app.repositories.vector_store import QdrantUnavailableError
from app.services.diversity_recommendation import DiversityRecommendationService


def test_returns_none_when_watch_history_collection_unavailable() -> None:
    with (
        patch("app.bootstrap.GoogleGenerativeAIEmbeddings", return_value=MagicMock()),
        patch(
            "app.bootstrap.connect_vector_store",
            side_effect=QdrantUnavailableError("collection does not exist"),
        ),
    ):
        result = build_diversity_service()

    assert result is None


def test_returns_a_service_when_both_collections_available() -> None:
    with (
        patch("app.bootstrap.GoogleGenerativeAIEmbeddings", return_value=MagicMock()),
        patch("app.bootstrap.connect_vector_store", return_value=MagicMock()),
        patch("app.bootstrap.load_watch_history_points", return_value=[]),
        patch("app.bootstrap.load_synopsis_vectors", return_value=[]),
        patch("app.bootstrap.load_synopsis_documents", return_value=[]),
    ):
        result = build_diversity_service()

    assert isinstance(result, DiversityRecommendationService)


def test_main_collection_failure_propagates_not_swallowed() -> None:
    """Only the watch_history connection is optional -- if media_items itself is
    unavailable (the connection every feature depends on), that's a real failure
    and must not be silently swallowed into a None return."""

    def fake_connect(url: str, collection: str, embeddings: object) -> MagicMock:
        if collection != "media_items":
            return MagicMock()
        raise QdrantUnavailableError("media_items down")

    with (
        patch("app.bootstrap.GoogleGenerativeAIEmbeddings", return_value=MagicMock()),
        patch("app.bootstrap.connect_vector_store", side_effect=fake_connect),
        patch("app.bootstrap.load_watch_history_points", return_value=[]),
        pytest.raises(QdrantUnavailableError, match="media_items down"),
    ):
        build_diversity_service()
