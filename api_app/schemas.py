from typing import Literal

from pydantic import BaseModel

from app.models.media_item import MediaItem


class ChatRequest(BaseModel):
    session_id: str
    message: str
    spoiler_free: bool = False


class MediaItemOut(BaseModel):
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
    video_resolution: str | None = None
    hdr_formats: list[str] = []
    source_platform: str | None = None
    runtime_minutes: int | None = None

    @classmethod
    def from_domain(cls, item: MediaItem) -> MediaItemOut:
        return cls(
            tmdb_id=item.tmdb_id,
            imdb_id=item.imdb_id,
            type=item.type,
            title=item.title,
            year=item.year,
            imdb_rating=item.imdb_rating,
            content_rating=item.content_rating,
            genres=item.genres,
            description=item.description,
            thumb_url=item.thumb_url,
            video_resolution=item.video_resolution.value
            if item.video_resolution
            else None,
            hdr_formats=[fmt.value for fmt in item.hdr_formats],
            source_platform=item.source_platform.value
            if item.source_platform
            else None,
            runtime_minutes=item.runtime_minutes,
        )


class ChatResponse(BaseModel):
    answer: str
    items: list[MediaItemOut]


# Wire events for POST /chat/stream — one of these, JSON-encoded, per
# newline-delimited line. `type` is a discriminator the tvOS client switches
# on. Mirrors app.services.recommendation.ChatStreamEvent (TextDelta /
# CardReady) one-for-one, just with the tmdb_id already resolved to a full
# MediaItemOut the way ChatResponse.items already is.
class ChatStreamTextOut(BaseModel):
    type: Literal["text"] = "text"
    text: str


class ChatStreamCardOut(BaseModel):
    type: Literal["card"] = "card"
    item: MediaItemOut


class ResetRequest(BaseModel):
    session_id: str


class ResetResponse(BaseModel):
    reset: bool


class HealthResponse(BaseModel):
    status: str
