import os
from pathlib import Path

# Qdrant vector store — plex-ingest owns writes; the recommender connects
# read-only over the network. See docs/vector-store-contract.md.
QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
QDRANT_COLLECTION = os.environ.get("QDRANT_COLLECTION", "media_items")

# Encrypts NiceGUI's per-browser-tab storage (nicegui_app/); any local value
# works since nothing sensitive is stored, but it must stay stable across
# restarts or open tabs get a fresh (empty) transcript.
NICEGUI_STORAGE_SECRET = os.environ.get("NICEGUI_STORAGE_SECRET", "plex-rag-dev-secret")

# DuckDB file backing the web UI's Recent-conversations sidebar list — the only
# read-write store plex-rag owns itself (unlike the read-only Qdrant connection
# above). Directory is created on demand by ConversationStore, not committed.
CONVERSATIONS_DB_PATH = os.environ.get(
    "CONVERSATIONS_DB_PATH",
    str(Path(__file__).resolve().parent.parent / "data" / "conversations.duckdb"),
)
