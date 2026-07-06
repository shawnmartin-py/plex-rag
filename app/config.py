import os

# Qdrant vector store — plex-ingest owns writes; the recommender connects
# read-only over the network. See docs/vector-store-contract.md.
QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
QDRANT_COLLECTION = os.environ.get("QDRANT_COLLECTION", "media_items")
