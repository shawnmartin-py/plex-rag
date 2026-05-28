import os

# Qdrant vector store
QDRANT_PATH = os.environ.get("QDRANT_PATH", "./media_items_qdrant_db")
QDRANT_COLLECTION = os.environ.get("QDRANT_COLLECTION", "media_items")

# Plex library names — must match the names shown in the Plex UI under "Libraries"
PLEX_MOVIE_LIBRARY = os.environ.get("PLEX_MOVIE_LIBRARY", "Movies")
PLEX_SHOW_LIBRARY = os.environ.get("PLEX_SHOW_LIBRARY", "TV Shows")
