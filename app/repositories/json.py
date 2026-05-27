import json
from typing import Self

from pydantic import BaseModel

from app.models.media_item import MediaItem as BaseMediaItem
from app.repositories.base import BaseRepo


class JsonMediaItems(BaseRepo):
    path = "media_items.json"

    def load(self) -> list[BaseMediaItem]:
        with open(self.path, "r") as file:
            item_dict = json.load(file)
        items = [
            BaseMediaItem(title=key, **values) for key, values in item_dict.items()
        ]
        self._load_cache(items)
        return items

    def save(self, media_items: list[BaseMediaItem]):
        media_items = self.load() + media_items
        items = [MediaItem.from_media_item(media_item) for media_item in media_items]
        item_dict = {item.title: item.model_dump(exclude={"title"}) for item in items}
        with open(self.path, "w") as file:
            json.dump(item_dict, file, indent=4)


class MediaItem(BaseModel, BaseMediaItem):
    @classmethod
    def from_media_item(cls, media_item: BaseMediaItem) -> Self:
        return cls.model_validate(media_item, from_attributes=True)
