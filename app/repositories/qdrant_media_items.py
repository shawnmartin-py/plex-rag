import enum
from typing import Any

from langchain_core.documents import Document

from app.models.media_item import (
    HdrFormat,
    MediaItem,
    StreamingSource,
    VideoResolution,
)


def _enum_or_none[E: enum.Enum](enum_cls: type[E], raw: Any) -> E | None:
    """Best-effort parse of an optional contract field into its local enum. Unlike
    plex-ingest's fail-fast parsing at write time, a value this side doesn't
    recognize (e.g. a stale reader against a newer contract) degrades to no badge
    rather than breaking the whole chat response — see docs/vector-store-contract.md."""
    if not isinstance(raw, str):
        return None
    for member in enum_cls:
        if member.value == raw:
            return member
    return None


def _hdr_formats_or_empty(raw: Any) -> list[HdrFormat]:
    """Same degrade-don't-break stance as _enum_or_none, applied per element:
    a missing/malformed field or an unrecognized member just means no HDR badge."""
    if not isinstance(raw, list):
        return []
    return [fmt for value in raw if (fmt := _enum_or_none(HdrFormat, value))]


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
        description=metadata.get("description"),
        thumb_url=metadata.get("thumb_url"),
        video_resolution=_enum_or_none(
            VideoResolution, metadata.get("video_resolution")
        ),
        hdr_formats=_hdr_formats_or_empty(metadata.get("hdr_formats")),
        source_platform=_enum_or_none(StreamingSource, metadata.get("source_platform")),
        runtime_minutes=metadata.get("runtime_minutes"),
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

    def all_items(self) -> list[MediaItem]:
        """Every synced item — powers the web UI's library snapshot. Cheap:
        the lookup table is already fully materialized at construction."""
        return list(self._by_id.values())
