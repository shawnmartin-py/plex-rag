import os
from pathlib import Path

# Qdrant vector store — plex-ingest owns writes; the recommender connects
# read-only over the network. See docs/vector-store-contract.md.
QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
QDRANT_COLLECTION = os.environ.get("QDRANT_COLLECTION", "media_items")

# Second, separate collection for the diversity/"surprise me" recommender — see
# docs/vector-store-contract.md's `watch_history` collection section. Optional:
# the feature disables itself gracefully if this collection doesn't exist yet
# (see app/bootstrap.py:build_diversity_service), so an unset/missing collection
# doesn't break the main chat feature.
QDRANT_WATCH_HISTORY_COLLECTION = os.environ.get(
    "QDRANT_WATCH_HISTORY_COLLECTION", "watch_history"
)

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
