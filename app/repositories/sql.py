from dataclasses import asdict
from dataclasses import fields as dc_fields

from sqlalchemy import JSON, create_engine
from sqlalchemy.dialects.sqlite import insert
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    mapped_column,
    sessionmaker,
)

from app.models.media_item import MediaItem
from app.repositories.base import BaseRepo


class SqlMediaItems(BaseRepo):
    def __init__(self, db_url: str = "sqlite:///media_items.db"):
        super().__init__()
        engine = create_engine(db_url)
        self.Session = sessionmaker(engine)
        Base.metadata.create_all(engine)

    def load(self) -> list[MediaItem]:
        with self.Session() as session:
            items = session.query(TableMediaItem).all()
            media_items = [MediaItem(**{f.name: getattr(item, f.name) for f in dc_fields(MediaItem)}) for item in items]
            self._load_cache(media_items)
            return media_items

    def save(self, media_items: list[MediaItem]):
        with self.Session.begin() as session:
            query = insert(TableMediaItem).values([asdict(media_item) for media_item in media_items])
            upsert = query.on_conflict_do_update(index_elements=["imdb_id"], set_={**query.excluded})
            session.execute(upsert)

    def get_by_id(self, imdb_id: str) -> MediaItem | None:
        return self._cached_items.get(imdb_id)

    def delete(self, imdb_ids: set[str]):
        with self.Session.begin() as session:
            session.query(TableMediaItem).filter(TableMediaItem.imdb_id.in_(imdb_ids)).delete(synchronize_session=False)


class Base(DeclarativeBase):
    __abstract__ = True

    def __init_subclass__(cls):
        cls.__tablename__ = "".join("_" + c.lower() if c.isupper() else c for c in cls.__name__).lstrip("_")
        super().__init_subclass__()


class TableMediaItem(Base, MediaItem):
    imdb_id: Mapped[str] = mapped_column(primary_key=True)
    type: Mapped[str]
    title: Mapped[str]
    year: Mapped[int]
    imdb_rating: Mapped[float]
    content_rating: Mapped[str]
    genres: Mapped[list[str]] = mapped_column(JSON)
    synopsis: Mapped[str | None]
    thumb_url: Mapped[str | None]
