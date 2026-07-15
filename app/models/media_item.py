import enum
from dataclasses import dataclass, field


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


class HdrFormat(enum.Enum):
    """Mirrors plex-ingest's lib/media_source.py by hand, like VideoResolution.
    A movie carries a *list* of these, not a single value: a Dolby Vision
    dual-layer file is also HDR10-compatible, so membership isn't mutually
    exclusive. `HDR` is one flat bucket for every HDR transfer function Plex
    reports (HDR10, HDR10+, HLG) — see docs/vector-store-contract.md."""

    HDR = "HDR"
    DV = "DV"


class StreamingSource(enum.Enum):
    """The platform a movie is actually available on when it's not a real Plex
    download — see docs/vector-store-contract.md's `source_platform` field."""

    NETFLIX = "Netflix"
    DISNEY_PLUS = "Disney+"


@dataclass
class MediaItem:
    tmdb_id: str
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
    hdr_formats: list[HdrFormat] = field(default_factory=list)
    source_platform: StreamingSource | None = None
    runtime_minutes: int | None = None
