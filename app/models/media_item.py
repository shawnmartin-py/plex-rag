from dataclasses import dataclass


@dataclass
class MediaItem:
    imdb_id: str
    type: str
    title: str
    year: int
    imdb_rating: float
    content_rating: str
    genres: list[str]
    thumb_url: str | None = None
