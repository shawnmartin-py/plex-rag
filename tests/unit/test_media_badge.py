from app.formatting.media_badge import (
    PlatformBadge,
    ResolutionBadge,
    describe_media_badge,
)
from app.models.media_item import MediaItem, StreamingSource, VideoResolution


def make_item(**overrides: object) -> MediaItem:
    defaults: dict[str, object] = {
        "tmdb_id": "496243",
        "imdb_id": "tt6751668",
        "type": "movie",
        "title": "Parasite",
        "year": 2019,
        "imdb_rating": 8.5,
        "content_rating": "R",
        "genres": ["Drama", "Thriller"],
    }
    return MediaItem(**{**defaults, **overrides})  # type: ignore[arg-type]


def test_resolution_badge_for_a_real_download() -> None:
    item = make_item(video_resolution=VideoResolution.R4K)
    assert describe_media_badge(item) == ResolutionBadge("2160p")


def test_platform_badge_for_a_streaming_placeholder() -> None:
    item = make_item(source_platform=StreamingSource.NETFLIX)
    assert describe_media_badge(item) == PlatformBadge(StreamingSource.NETFLIX)


def test_platform_badge_takes_priority_over_resolution() -> None:
    """Shouldn't happen by contract (the two are mutually exclusive on write), but
    the read-side priority is still asserted directly since it's the behavior a
    future contract violation would actually hit."""
    item = make_item(
        video_resolution=VideoResolution.R1080,
        source_platform=StreamingSource.DISNEY_PLUS,
    )
    assert describe_media_badge(item) == PlatformBadge(StreamingSource.DISNEY_PLUS)


def test_no_badge_when_neither_field_is_set() -> None:
    assert describe_media_badge(make_item()) is None
