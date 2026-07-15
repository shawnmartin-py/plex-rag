from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

import api_app.main as api_main
from app.domain.ports import ConversationTitler
from app.models.media_item import MediaItem
from app.repositories.qdrant_media_items import QdrantMediaItems
from app.services.recommendation import ConversationalRecommendationService


def make_item(tmdb_id: str = "1", title: str = "Heat") -> MediaItem:
    return MediaItem(
        tmdb_id=tmdb_id,
        imdb_id=f"tt{tmdb_id}",
        type="movie",
        title=title,
        year=1995,
        imdb_rating=8.2,
        content_rating="R",
        genres=["Crime", "Drama"],
    )


def make_built_service(
    answer: str = "1. **Heat** (1995)\nA tense heist.",
    items: list[MediaItem] | None = None,
) -> tuple[MagicMock, MagicMock, MagicMock]:
    service = MagicMock(spec=ConversationalRecommendationService)
    service.chat_with_items = AsyncMock(
        return_value=(answer, items if items is not None else [make_item()])
    )
    media_repo = MagicMock(spec=QdrantMediaItems)
    titler = MagicMock(spec=ConversationTitler)
    return service, media_repo, titler


@pytest.fixture
def client() -> TestClient:
    return TestClient(api_main.app)


def test_health_returns_ok(client: TestClient) -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_chat_returns_answer_and_items(client: TestClient) -> None:
    built = make_built_service()
    with patch.object(
        api_main, "get_service", AsyncMock(return_value=built)
    ) as mock_get_service:
        response = client.post(
            "/chat",
            json={"session_id": "session-a", "message": "something tense"},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "1. **Heat** (1995)\nA tense heist."
    assert body["items"] == [
        {
            "tmdb_id": "1",
            "imdb_id": "tt1",
            "type": "movie",
            "title": "Heat",
            "year": 1995,
            "imdb_rating": 8.2,
            "content_rating": "R",
            "genres": ["Crime", "Drama"],
            "description": None,
            "thumb_url": None,
            "video_resolution": None,
            "hdr_formats": [],
            "source_platform": None,
            "runtime_minutes": None,
        }
    ]
    mock_get_service.assert_called_once_with("session-a", False)
    built[0].chat_with_items.assert_called_once_with("something tense", built[1])


def test_chat_passes_through_spoiler_free_flag(client: TestClient) -> None:
    built = make_built_service()
    with patch.object(api_main, "get_service", AsyncMock(return_value=built)) as mock:
        client.post(
            "/chat",
            json={
                "session_id": "session-a",
                "message": "something tense",
                "spoiler_free": True,
            },
        )

    mock.assert_called_once_with("session-a", True)


def test_chat_rejects_blank_message(client: TestClient) -> None:
    with patch.object(
        api_main, "get_service", AsyncMock(return_value=make_built_service())
    ) as mock_get_service:
        response = client.post(
            "/chat", json={"session_id": "session-a", "message": "   "}
        )

    assert response.status_code == 422
    mock_get_service.assert_not_called()


def test_reset_reports_true_when_a_session_was_found(client: TestClient) -> None:
    with patch.object(api_main, "reset_session", return_value=True) as mock_reset:
        response = client.post("/chat/reset", json={"session_id": "session-a"})

    assert response.status_code == 200
    assert response.json() == {"reset": True}
    mock_reset.assert_called_once_with("session-a")


def test_reset_reports_false_when_nothing_was_cached(client: TestClient) -> None:
    with patch.object(api_main, "reset_session", return_value=False):
        response = client.post("/chat/reset", json={"session_id": "unknown"})

    assert response.status_code == 200
    assert response.json() == {"reset": False}
