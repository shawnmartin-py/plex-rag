from typing import Any

from langchain_core.documents import Document

from app.models.media_item import MediaItem


def _media_item_from_metadata(metadata: dict[str, Any]) -> MediaItem:
    genres = metadata["genres"]
    return MediaItem(
        imdb_id=metadata["imdb_id"],
        type=metadata["type"],
        title=metadata["title"],
        year=metadata["year"],
        imdb_rating=metadata["imdb_rating"],
        content_rating=metadata["content_rating"],
        genres=genres.split(", ") if genres else [],
        thumb_url=metadata.get("thumb_url"),
    )


class QdrantMediaItems:
    """Read-only MediaItem lookup sourced entirely from Qdrant synopsis-point
    metadata — the recommender's only external data dependency, per
    docs/vector-store-contract.md."""

    def __init__(self, synopsis_documents: list[Document]) -> None:
        self._by_id = {
            doc.metadata["imdb_id"]: _media_item_from_metadata(doc.metadata)
            for doc in synopsis_documents
        }

    def get_by_id(self, imdb_id: str) -> MediaItem | None:
        return self._by_id.get(imdb_id)
