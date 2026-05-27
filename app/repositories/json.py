import json
from dataclasses import asdict

from app.models.media_item import MediaItem
from app.repositories.base import BaseRepo


class JsonMediaItems(BaseRepo):
    def __init__(self, path: str = "media_items.json"):
        super().__init__()
        self.path = path

    def load(self) -> list[MediaItem]:
        with open(self.path, "r") as file:
            item_dict = json.load(file)
        items = [MediaItem(title=key, **values) for key, values in item_dict.items()]
        self._load_cache(items)
        return items

    def save(self, media_items: list[MediaItem]):
        merged = {**self._cached_items, **{item.imdb_id: item for item in media_items}}
        item_dict = {}
        for item in merged.values():
            d = asdict(item)
            d.pop("title")
            item_dict[item.title] = d
        with open(self.path, "w") as file:
            json.dump(item_dict, file, indent=4)
        self._load_cache(list(merged.values()))
