# Target architecture

**This target has been achieved (2026-07-06)** — see [breakdown.md](breakdown.md)
for the phase-by-phase record. The "current (monolith)" section below is
kept for historical context; `plex-rag` no longer matches it.

## Before (monolith)

```
                         ┌─────────────────────────────────────┐
                         │              plex-rag                │
                         │                                       │
  Plex Server ──────────▶│ app/plex.py, main.py, synopsis.py,   │
                         │ scrape_imdb.py, services/enrichment.py│
                         │        │                              │
                         │        ▼                              │
                         │ media_items.db (SQLite) ◀──────┐      │
                         │        │                        │      │
                         │        ▼                        │      │
                         │ media_items_qdrant_db (on-disk) │      │
                         │        ▲                        │      │
                         │        │ (recommender ALSO      │      │
                         │        │  writes here on        │      │
                         │        │  startup — see below)  │      │
                         │        │                        │      │
                         │ app/rag.py, streamlit_app/ ──────┘      │
                         │  (reads SqlMediaItems +               │
                         │   read/writes Qdrant)                 │
                         └─────────────────────────────────────┘
```

Both halves run in one process/repo, share one SQLite file and one on-disk
Qdrant directory. The recommender isn't actually read-only today — `rag.py`
and `streamlit_app/init.py` both call `load_or_build`, which embeds and
upserts any not-yet-embedded synopsis documents at chat startup.

## After (two services) ✅ current state

```
┌────────────────────────────────────────┐          ┌──────────────────────────────────┐
│              plex-ingest                 │          │             plex-rag              │
│           (Dagster orchestrator)         │          │       (recommender only)          │
│                                           │          │                                    │
│  Plex Server ──▶ raw_movies (DuckDB)    │          │  CLI chat / Streamlit UI          │
│       │                                  │          │       │                            │
│       ▼                                  │          │       ▼                            │
│  stg_movies (dbt, DuckDB)                │          │  MovieRecommender                  │
│       │                                  │          │  (retrievers, generator, rewriter) │
│       ▼                                  │          │       │                            │
│  sync_imdb_id_partitions (sensor)        │          │       │  read-only query           │
│  adds/removes imdb_id partitions +       │          │       ▼                            │
│  on-disk files for movies added/removed  │          │  QdrantClient(url=...)            │
│       │                                  │          │       ▲                            │
│       ▼   (partitioned by imdb_id, no    │          │       │                            │
│  synopsis  automation_condition — sensor │          │       │                            │
│  ▼         triggers it directly)         │          └───────┼────────────────────────────┘
│  enrichment  (partitioned, sensor-       │                  │
│  triggered off synopsis's on-disk file)  │                  │
│       ▼                                  │                  │
│  embeddings  (partitioned, eager —       │                  │
│  embeds synopsis doc + every enrichment  │                  │
│  section, up to 4 per movie)             │                  │
│       ▼                                  │                  │
│  qdrant_collection (UNpartitioned,       │                  │
│  eager — full delete+reinsert from       │                  │
│  every embeddings/*.json, attaching      │                  │
│  catalog metadata read fresh from        │                  │
│  stg_movies) ─────────────────────────────┼──────────────────┘
│       │                                  │        networked, read-only from plex-rag
│       ▼                                  │
│  ┌───────────────────────┐              │
│  │  Qdrant (Docker)        │              │  ◀── docker-compose.yml + volume live here
│  │  networked, server mode │              │
│  └───────────────────────┘              │
└────────────────────────────────────────┘
```

Single cross-repo dependency: the Qdrant collection, governed by
[vector-store-contract.md](../../vector-store-contract.md). `plex-rag` has
no SQLite/DuckDB/parquet dependency on `plex-ingest` at all — everything it
needs (including synopsis text, via `page_content`) comes from Qdrant point
payloads.

Internal storage inside `plex-ingest` (implementation detail, not part of
the cross-repo contract): DuckDB for `raw_movies`/`stg_movies` (SQL-shaped,
single-writer, unpartitioned); one JSON file per movie per stage for
`synopsis`/`enrichment`/`embeddings` (avoids DuckDB's single-writer lock
contention under the intended partition concurrency) — see
[phase-2-pipeline-design.md](phase-2-pipeline-design.md) for the full
reasoning.

For the exact, authoritative file-by-file record of what was deleted, kept,
or rewritten (including a correction to this doc's original file-mapping
plan), see [breakdown.md](breakdown.md) phase 7.
