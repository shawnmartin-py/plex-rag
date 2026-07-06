# Phased breakdown

**All 7 phases are now complete (2026-07-06).** Ordering principle used
throughout: `plex-ingest` got built, populated, and proven to fully replace
the existing pipeline before anything was removed from `plex-rag`. Phases
1-3 happened entirely in the `plex-ingest` repo; phases 4-7 (below) are the
ones that changed `plex-rag`, and are all done — this repo is now
recommender-only.

## Phase 1 — Scaffold `plex-ingest` ✅ done

Scope: new repo. Doesn't touch `plex-rag`.

- Dagster project skeleton — done (`create-dagster project`, `dg`
  component-based layout).
- `docker-compose.yml` with the Qdrant service + named volume for
  persistent storage — done, running locally.
- Copy `docs/vector-store-contract.md` into the new repo — done.
- Prove Dagster + dockerized Qdrant boot cleanly and `plex-ingest` can write
  a trivial point to a collection matching the contract — done (the
  `qdrant_probe` asset).

## Phase 2 — Port pipeline logic into Dagster assets ✅ done

Scope: new repo. This is the biggest phase and the one explicitly deferred
for its own deep-dive per the epic decision log — don't over-design it
prematurely. See [phase-2-pipeline-design.md](phase-2-pipeline-design.md)
for the open questions (partitioning, frameworks) that still need to be
worked through jointly — don't assume they're settled just because the
items below are checked off.

- ✅ Plex sync → raw catalog write: `raw_movies` asset, full overwrite into
  DuckDB every run, unpartitioned (~3s for the whole library).
- ✅ Catalog/staging transform: `stg_movies` dbt model, resolves `imdb_id`
  from Plex's raw `guids`, with `not_null`/`unique` tests.
- ✅ Synopsis scraping (Playwright) as its own asset (`synopsis`), matching
  today's fallback chain (IMDb plot summary → Wikipedia → IMDb short
  description). Partitioned by `imdb_id`; triggered directly by the
  `sync_imdb_id_partitions` sensor (not `AutomationCondition.on_missing()`,
  which had a cold-start bug — see
  [phase-2-pipeline-design.md](phase-2-pipeline-design.md)'s "Known gaps").
- ✅ Enrichment (craft/meaning/context) as an asset (`enrichment`),
  preserving the existing retry/backoff and content-policy-retry behavior.
  Partitioning question resolved — dynamic partitions by `imdb_id`, same
  sensor-driven trigger relative to `synopsis`'s on-disk presence — see
  [phase-2-pipeline-design.md](phase-2-pipeline-design.md).
- ✅ Embed + upsert into the dockerized Qdrant, matching
  `vector-store-contract.md` exactly: a separate `embeddings` asset
  (`eager()`, embeds the synopsis document *and* every enrichment section —
  up to 4 documents per movie) feeds a final unpartitioned `qdrant_collection`
  asset that does a full delete+reinsert rebuild from every embeddings file
  on disk, attaching full catalog metadata (`title`/`year`/`imdb_rating`/
  `content_rating`/`genres`/`thumb_url`) and `embedding_type` read fresh
  from `stg_movies` at rebuild time.
- ✅ Removal sync — reworked from `_sync_removals_to_vector_store`'s
  Qdrant filter-delete into a simpler cascade: the `sync_imdb_id_partitions`
  sensor deletes the dynamic partition and on-disk files for any imdb_id no
  longer in `stg_movies`, and directly requests a `qdrant_collection`
  rebuild so the removal takes effect immediately (see
  [phase-2-pipeline-design.md](phase-2-pipeline-design.md)'s "Known gaps",
  item 3). No Qdrant-specific *deletion* code at all — the rebuild is
  still a full delete+reinsert from whatever's on disk, not a targeted
  Qdrant delete call.

Verified end to end against real Plex/Gemini/Qdrant, then run against the
full library with `PLEX_INGEST_PARTITION_LIMIT` lifted (phase 3). Ported
from `app/synopsis.py`, `app/scrape_imdb.py`, `app/browser.py`,
`app/services/enrichment.py`, and the write half of
`app/services/vector_store.py` — not 1:1, since the execution model changed
from CLI commands to partitioned Dagster assets plus a sensor-driven
partition-sync/deletion cascade. `plex-rag`'s copies of these files have
since been deleted (phase 7).

## Phase 3 — Prove out `plex-ingest` and migrate data ✅ done (2026-07-06)

Scope: new repo, plus a one-time data migration.

- `plex-ingest`'s full pipeline was run standalone (full ~156-movie library,
  `PLEX_INGEST_PARTITION_LIMIT` lifted) against the dockerized Qdrant, and
  `plex-rag`'s recommender was verified against a subset of that collection
  before the full run completed.
- **Data migration resolved by re-embedding, not volume copy**: rather than
  verifying on-disk→dockerized Qdrant volume/snapshot compatibility, the
  full library was re-embedded directly into the dockerized Qdrant via
  `plex-ingest`. Simpler, at the cost of one full re-embed run.
- `media_items.db` (SQLite) was not migrated — deleted outright in phase 7,
  since the recommender no longer has any catalog dependency (everything
  needed is in Qdrant payloads).
- Exit criterion met: `plex-ingest`, running alone, populates a dockerized
  Qdrant collection that `plex-rag` reads from successfully, read-only.

## Phase 4 — Stop the recommender writing to Qdrant ✅ done

Scope: `plex-rag`.

- Removed the `load_or_build` write path from `app/rag.py` and
  `streamlit_app/init.py`. Replaced with `connect_vector_store` +
  `load_synopsis_documents` (`app/services/recommender_vector_store.py`) —
  a read-only connect that assumes the collection already exists and fails
  fast with a clear error (`QdrantUnavailableError`) if it doesn't.
- The old `VectorStoreService` (`app/services/vector_store.py`, write path)
  has been deleted entirely rather than split, since nothing on the
  `plex-rag` side needs any of its write methods anymore — the read path
  was rewritten from scratch as `recommender_vector_store.py` instead.
- `app/adapters/retrievers.py` never assumed write access or the on-disk
  client specifically — confirmed clean, no changes needed.

## Phase 5 — Point `plex-rag` at the networked Qdrant ✅ done

Scope: `plex-rag`.

- `plex-rag`'s config now uses `QDRANT_URL` / `QDRANT_COLLECTION` only —
  `QDRANT_PATH` has been removed from `app/config.py` entirely.
- `connect_vector_store` (`app/services/recommender_vector_store.py`) is
  the startup preflight check used by both `app/rag.py` and
  `streamlit_app/init.py`: connects, confirms the collection exists,
  confirms vector size matches `vector-store-contract.md`. Fails fast with
  a clear message if not.
- Verified end-to-end: `plex-rag` chatting against the dockerized Qdrant
  instance `plex-ingest` populated in phase 3, with zero write access.

## Phase 6 — Drop the SQLite/`MediaItem` catalog dependency from the recommender ✅ done

Scope: `plex-rag`.

- `app/services/recommendation.py:chat_with_items` now takes a
  `MediaItemLookup` (`app/domain/ports.py`) instead of `SqlMediaItems`,
  satisfied by `QdrantMediaItems` (`app/repositories/qdrant_media_items.py`)
  — built from the same `embedding_type=synopsis` Qdrant points already
  loaded via `load_synopsis_documents`, no second round-trip to any store.
- The read-side shape is a full `MediaItem` reconstruction (minus
  `synopsis`, which nothing on this side needs) — matches exactly what
  `streamlit_app/components.py` renders (`thumb_url`, `imdb_rating`,
  `title`, plus the fields carried through for completeness).

## Phase 7 — Cleanup and cutover ✅ done

Scope: `plex-rag`. `plex-rag` is now fully decoupled and running against
`plex-ingest`'s networked Qdrant.

- Deleted `app/repositories/sql.py`, `base.py`, `json.py`, and
  `app/enums.py` from `plex-rag` entirely.
- Deleted the pipeline code: `app/plex.py`, `app/main.py`,
  `app/synopsis.py`, `app/scrape_imdb.py`, `app/browser.py`,
  `app/services/enrichment.py`, `app/services/vector_store.py`; the
  `sync`/`scrape`/`enrich`/`clear-enrichments` CLI commands (`app/cli.py`
  now only has `chat`); and their tests (`unit/test_sync.py`,
  `unit/test_synopsis.py`, `unit/test_enrichment.py`,
  `integration/test_safety_settings.py`, `e2e/test_enrichment.py`,
  `e2e/test_vector_store.py`).
  **Correction to this doc's earlier plan:** `e2e/test_pipeline.py` was
  listed above as moving to `plex-ingest` — that was wrong. Its actual
  content tests the *recommendation* RAG pipeline (retrievers, generator,
  `ConversationalRecommendationService`), not the ingest pipeline. It stays
  in `plex-rag`; only `tests/e2e/conftest.py`'s fixtures were reworked to
  build an in-memory `QdrantVectorStore` directly instead of via the
  now-deleted `VectorStoreService`.
- Deleted `media_items.db` and (already absent) `media_items_qdrant_db/`
  from `plex-rag`.
- Stripped `app/models/media_item.py` down to a plain dataclass (dropped
  `synopsis` field, `from_plex`, `to_document`, `to_enriched_document`,
  `to_metadata` — all pipeline-only). Deleted
  `tests/unit/test_media_item.py` since nothing meaningful remained to test
  on a bare dataclass.
- Removed pipeline-only dependencies from `pyproject.toml`:
  `beautifulsoup4`, `playwright`, `plexapi`, `requests`, `sqlalchemy`.
- Updated `README.md` and `CLAUDE.md` to drop pipeline-related
  setup/usage instructions and point to `plex-ingest` for that half.
- Deleted `docs/pipeline.md` (pipeline no longer lives here).