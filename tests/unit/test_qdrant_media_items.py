from typing import Any

from langchain_core.documents import Document

from app.models.media_item import StreamingSource, VideoResolution
from app.repositories.qdrant_media_items import QdrantMediaItems


def make_synopsis_doc(
    imdb_id: str = "tt6751668", **metadata_overrides: Any
) -> Document:
    metadata = {
        "imdb_id": imdb_id,
        "type": "movie",
        "title": "Parasite",
        "year": 2019,
        "imdb_rating": 8.5,
        "content_rating": "R",
        "genres": "Drama, Thriller",
        "thumb_url": "http://example.com/thumb.jpg",
        "embedding_type": "synopsis",
        **metadata_overrides,
    }
    return Document(page_content="Title: Parasite", metadata=metadata)


def test_get_by_id_returns_media_item_for_known_imdb_id() -> None:
    repo = QdrantMediaItems([make_synopsis_doc()])
    item = repo.get_by_id("tt6751668")
    assert item is not None
    assert item.title == "Parasite"


def test_get_by_id_returns_none_for_unknown_imdb_id() -> None:
    repo = QdrantMediaItems([make_synopsis_doc()])
    assert repo.get_by_id("tt9999999") is None


def test_get_by_id_splits_comma_joined_genres_back_into_a_list() -> None:
    repo = QdrantMediaItems([make_synopsis_doc(genres="Drama, Thriller")])
    item = repo.get_by_id("tt6751668")
    assert item is not None
    assert item.genres == ["Drama", "Thriller"]


def test_get_by_id_handles_missing_thumb_url() -> None:
    doc = make_synopsis_doc()
    del doc.metadata["thumb_url"]
    repo = QdrantMediaItems([doc])
    item = repo.get_by_id("tt6751668")
    assert item is not None
    assert item.thumb_url is None


def test_get_by_id_handles_empty_genres_string() -> None:
    repo = QdrantMediaItems([make_synopsis_doc(genres="")])
    item = repo.get_by_id("tt6751668")
    assert item is not None
    assert item.genres == []


def test_get_by_id_parses_video_resolution() -> None:
    repo = QdrantMediaItems([make_synopsis_doc(video_resolution="4k")])
    item = repo.get_by_id("tt6751668")
    assert item is not None
    assert item.video_resolution is VideoResolution.R4K
    assert item.source_platform is None


def test_get_by_id_parses_source_platform() -> None:
    repo = QdrantMediaItems([make_synopsis_doc(source_platform="Netflix")])
    item = repo.get_by_id("tt6751668")
    assert item is not None
    assert item.source_platform is StreamingSource.NETFLIX
    assert item.video_resolution is None


def test_get_by_id_defaults_video_resolution_and_source_platform_to_none() -> None:
    repo = QdrantMediaItems([make_synopsis_doc()])
    item = repo.get_by_id("tt6751668")
    assert item is not None
    assert item.video_resolution is None
    assert item.source_platform is None


def test_get_by_id_treats_unrecognized_video_resolution_as_none() -> None:
    """A stale reader against a newer contract shouldn't break the chat response —
    see _enum_or_none's docstring in qdrant_media_items.py."""
    repo = QdrantMediaItems([make_synopsis_doc(video_resolution="8k")])
    item = repo.get_by_id("tt6751668")
    assert item is not None
    assert item.video_resolution is None


def test_multiple_documents_are_all_indexed() -> None:
    repo = QdrantMediaItems(
        [make_synopsis_doc("tt0001", title="A"), make_synopsis_doc("tt0002", title="B")]
    )
    item_a = repo.get_by_id("tt0001")
    item_b = repo.get_by_id("tt0002")
    assert item_a is not None
    assert item_b is not None
    assert item_a.title == "A"
    assert item_b.title == "B"
