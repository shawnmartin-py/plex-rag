from abc import ABC, abstractmethod

from app.models.media_item import MediaItem


class BaseRepo(ABC):
    def __init__(self):
        self._cached_items: dict[str, MediaItem] = {}

    @abstractmethod
    def load(self) -> list[MediaItem]: ...

    @abstractmethod
    def save(self, media_items: list[MediaItem]): ...

    def _load_cache(self, media_items: list[MediaItem]):
        self._cached_items = {item.imdb_id: item for item in media_items}

    def __contains__(self, item: MediaItem):
        return item.imdb_id in self._cached_items
