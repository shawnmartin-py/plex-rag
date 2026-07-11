from dataclasses import dataclass

from app.models.media_item import MediaItem, StreamingSource, VideoResolution

_RESOLUTION_LABELS: dict[VideoResolution, str] = {
    VideoResolution.SD: "SD",
    VideoResolution.R480: "480p",
    VideoResolution.R576: "576p",
    VideoResolution.R720: "720p",
    VideoResolution.R1080: "1080p",
    VideoResolution.R4K: "2160p",
}


@dataclass(frozen=True)
class ResolutionBadge:
    label: str


@dataclass(frozen=True)
class PlatformBadge:
    platform: StreamingSource


type MediaBadge = ResolutionBadge | PlatformBadge


def describe_media_badge(item: MediaItem) -> MediaBadge | None:
    """The one quality/source badge to show for a card: a streaming platform's logo
    takes priority over a resolution tag. The two are mutually exclusive on
    `MediaItem` by contract (see docs/vector-store-contract.md) — a placeholder's
    ~4s stand-in file has no meaningful resolution of its own."""
    if item.source_platform is not None:
        return PlatformBadge(item.source_platform)
    if item.video_resolution is not None:
        return ResolutionBadge(_RESOLUTION_LABELS[item.video_resolution])
    return None
