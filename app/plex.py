from enum import Enum

from plexapi.server import PlexServer

from app.models.media_item import MediaItem


class Plex:
    class MediaType(Enum):
        MOVIE = "All Movies"
        SHOW = "All Shows"

    def __init__(self):
        self.server = PlexServer()

    def get_media_items(self, media_types: set[MediaType] | None = None, unwatched: bool = True) -> list[MediaItem]:
        if media_types is None:
            media_types = {self.MediaType.SHOW, self.MediaType.MOVIE}
        media_items = []
        for media_type in media_types:
            collection = self.server.library.section(media_type.value)
            items = [MediaItem.from_plex(item) for item in collection.search(unwatched=unwatched)]
            media_items.extend(items)
        return media_items
