from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

_DEFAULT_CONVERSATIONS_DB_PATH = str(
    Path(__file__).resolve().parent.parent / "data" / "conversations.duckdb"
)


class Settings(BaseSettings):
    """Single schema for plex-rag's env-var configuration. Validates at
    construction (process startup) instead of wherever a bad value is first
    used, several call frames deep."""

    model_config = SettingsConfigDict(case_sensitive=True)

    # Qdrant vector store — plex-ingest owns writes; the recommender connects
    # read-only over the network. See docs/vector-store-contract.md.
    QDRANT_URL: str = Field(default="http://localhost:6333", min_length=1)
    QDRANT_COLLECTION: str = Field(default="media_items", min_length=1)

    # Second, separate collection for the diversity/"surprise me" recommender —
    # see docs/vector-store-contract.md's `watch_history` collection section.
    # Optional: the feature disables itself gracefully if this collection
    # doesn't exist yet (see app/bootstrap.py:build_diversity_service), so an
    # unset/missing collection doesn't break the main chat feature.
    QDRANT_WATCH_HISTORY_COLLECTION: str = Field(default="watch_history", min_length=1)

    # Encrypts NiceGUI's per-browser-tab storage (nicegui_app/); any local
    # value works since nothing sensitive is stored, but it must stay stable
    # across restarts or open tabs get a fresh (empty) transcript.
    NICEGUI_STORAGE_SECRET: str = Field(default="plex-rag-dev-secret", min_length=1)

    # DuckDB file backing the web UI's Recent-conversations sidebar list — the
    # only read-write store plex-rag owns itself (unlike the read-only Qdrant
    # connection above). Directory is created on demand by ConversationStore,
    # not committed.
    CONVERSATIONS_DB_PATH: str = Field(
        default=_DEFAULT_CONVERSATIONS_DB_PATH, min_length=1
    )

    # When true, app/bootstrap.py swaps every Gemini call (chat + embeddings)
    # for a deterministic in-process fake (app/adapters/fake_gemini.py) — no
    # network calls, no GOOGLE_API_KEY needed, no quota spent. Qdrant is still
    # hit for real. For local/manual testing of the app's other parts only —
    # evals/ never reads this, since faking the model under test would make
    # those evals meaningless.
    FAKE_GEMINI: bool = False


settings = Settings()

QDRANT_URL = settings.QDRANT_URL
QDRANT_COLLECTION = settings.QDRANT_COLLECTION
QDRANT_WATCH_HISTORY_COLLECTION = settings.QDRANT_WATCH_HISTORY_COLLECTION
NICEGUI_STORAGE_SECRET = settings.NICEGUI_STORAGE_SECRET
CONVERSATIONS_DB_PATH = settings.CONVERSATIONS_DB_PATH
FAKE_GEMINI = settings.FAKE_GEMINI
