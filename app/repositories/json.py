import json
from typing import Self

from pydantic import BaseModel

from app.models.media_item import MediaItem as BaseMediaItem
from app.repositories.base import BaseRepo


class JsonMediaItems(BaseRepo):
    def __init__(self, path: str = "media_items.json"):
        super().__init__()
        self.path = path

    def load(self) -> list[BaseMediaItem]:
        with open(self.path, "r") as file:
            item_dict = json.load(file)
        items = [BaseMediaItem(title=key, **values) for key, values in item_dict.items()]
        self._load_cache(items)
        return items

    def save(self, media_items: list[BaseMediaItem]):
        merged = {**self._cached_items, **{item.imdb_id: item for item in media_items}}
        items = [MediaItem.from_media_item(item) for item in merged.values()]
        item_dict = {item.title: item.model_dump(exclude={"title"}) for item in items}
        with open(self.path, "w") as file:
            json.dump(item_dict, file, indent=4)
        self._load_cache(list(merged.values()))


class MediaItem(BaseModel, BaseMediaItem):
    @classmethod
    def from_media_item(cls, media_item: BaseMediaItem) -> Self:
        return cls.model_validate(media_item, from_attributes=True)
