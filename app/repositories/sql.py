# from abc import ABCMeta
from dataclasses import asdict

# from typing import Self
from sqlalchemy import JSON, create_engine  # ,select
from sqlalchemy.dialects.sqlite import insert
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    mapped_column,
    registry,
    sessionmaker,
)

from app.models.media_item import MediaItem
from app.repositories.base import BaseRepo


class SqlMediaItems(BaseRepo):
    def __init__(self):
        super().__init__()
        engine = create_engine("sqlite:///media_items.db")
        self.Session = sessionmaker(engine)
        Base.metadata.create_all(engine)
        registry().map_imperatively(MediaItem, TableMediaItem.__table__)

    def load(self) -> list[MediaItem]:
        with self.Session() as session:
            items = session.query(TableMediaItem).all()
            media_items = [MediaItem(**item._asdict()) for item in items]
            self._load_cache(media_items)
            return media_items

    def save(self, media_items: list[MediaItem]):
        with self.Session.begin() as session:
            query = insert(MediaItem).values(
                [asdict(media_item) for media_item in media_items]
            )
            upsert = query.on_conflict_do_update(
                index_elements=["imdb_id"], set_={**query.excluded}
            )
            session.execute(upsert)

    def delete(self, imdb_ids: set[str]):
        with self.Session.begin() as session:
            session.query(TableMediaItem).filter(
                TableMediaItem.imdb_id.in_(imdb_ids)
            ).delete(synchronize_session=False)
            # session.add_all(items)


# class DeclarativeABCMeta(DeclarativeMeta, ABCMeta): ...
class Base(DeclarativeBase):
    __abstract__ = True

    def __init_subclass__(cls):
        super().__init_subclass__()
        cls.__tablename__ = "".join(
            "_" + c.lower() if c.isupper() else c for c in cls.__name__
        ).lstrip("_")


class TableMediaItem(Base, MediaItem):
    imdb_id: Mapped[str] = mapped_column(primary_key=True)
    type: Mapped[str]
    title: Mapped[str]
    year: Mapped[int]
    imdb_rating: Mapped[float]
    content_rating: Mapped[str]
    genres: Mapped[list[str]] = mapped_column(JSON)
    synopsis: Mapped[str | None]
