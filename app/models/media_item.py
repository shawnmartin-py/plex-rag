import enum
from dataclasses import dataclass


class VideoResolution(enum.Enum):
    """Mirrors Plex's own `Media.videoResolution` vocabulary exactly — see
    plex-ingest's lib/media_source.py (mirrored by hand, no shared package, per
    docs/vector-store-contract.md)."""

    SD = "sd"
    R480 = "480"
    R576 = "576"
    R720 = "720"
    R1080 = "1080"
    R4K = "4k"


class StreamingSource(enum.Enum):
    """The platform a movie is actually available on when it's not a real Plex
    download — see docs/vector-store-contract.md's `source_platform` field."""

    NETFLIX = "Netflix"
    DISNEY_PLUS = "Disney+"


@dataclass
class MediaItem:
    imdb_id: str
    type: str
    title: str
    year: int
    imdb_rating: float
    content_rating: str
    genres: list[str]
    description: str | None = None
    thumb_url: str | None = None
    video_resolution: VideoResolution | None = None
    source_platform: StreamingSource | None = None
