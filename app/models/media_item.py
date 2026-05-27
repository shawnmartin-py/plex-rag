from dataclasses import asdict, dataclass
from typing import Self

from langchain_core.documents import Document


@dataclass
class MediaItem:
    imdb_id: str
    type: str
    title: str
    year: int
    imdb_rating: float
    content_rating: str
    genres: list[str]
    synopsis: str | None = None

    @classmethod
    def from_plex(cls, plex_item) -> Self:
        imdb_guid = next((g for g in plex_item.guids if g.id.startswith("imdb://")), None)
        if imdb_guid is None:
            raise ValueError(f"No IMDb GUID found for '{plex_item.title}'")
        return cls(
            imdb_id=imdb_guid.id.replace("imdb://", ""),
            title=plex_item.title,
            type=plex_item.type,
            year=plex_item.year,
            imdb_rating=plex_item.ratings[0].value if plex_item.ratings else 0.0,
            content_rating=plex_item.contentRating,
            genres=[genre.tag for genre in plex_item.genres],
            synopsis=None,
        )

    def to_metadata(self) -> dict:
        metadata = asdict(self)
        metadata.pop("synopsis")
        metadata["genres"] = ", ".join(self.genres)
        return metadata

    def to_document(self) -> Document:
        content = (
            f"Title: {self.title}\n"
            f"Year: {self.year}\n"
            f"IMDb Rating: {self.imdb_rating}\n"
            f"Genres: {', '.join(self.genres)}\n"
            f"Synopsis: {self.synopsis}"
        )
        metadata = self.to_metadata()
        metadata["embedding_type"] = "synopsis"
        return Document(page_content=content, metadata=metadata)

    def to_enriched_document(self, enrichment_text: str, section: str) -> Document:
        metadata = self.to_metadata()
        metadata["embedding_type"] = "enriched"
        metadata["section"] = section
        return Document(page_content=enrichment_text, metadata=metadata)
