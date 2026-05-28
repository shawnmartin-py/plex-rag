from plexapi.server import PlexServer

from app.config import PLEX_MOVIE_LIBRARY, PLEX_SHOW_LIBRARY
from app.models.media_item import MediaItem

_ALL_LIBRARIES = {PLEX_MOVIE_LIBRARY, PLEX_SHOW_LIBRARY}


class Plex:
    def __init__(self):
        self.server = PlexServer()

    def get_media_items(self, libraries: set[str] | None = None, unwatched: bool = True) -> list[MediaItem]:
        """Fetch media items from one or more Plex libraries by their library name.

        Pass a subset of library names to restrict the fetch, or omit to fetch all configured libraries.
        Library names must match what appears in the Plex UI under Libraries.
        """
        media_items: list[MediaItem] = []
        for library_name in libraries or _ALL_LIBRARIES:
            collection = self.server.library.section(library_name)
            media_items.extend(MediaItem.from_plex(item) for item in collection.search(unwatched=unwatched))
        return media_items
