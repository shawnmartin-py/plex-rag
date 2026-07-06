# Epic: Extract the data pipeline into `plex-ingest`

**Status:** phases 1-7 complete. `plex-rag` is now recommender-only, running
against the networked Qdrant `plex-ingest` populates — see
[breakdown.md](breakdown.md) for phase-by-phase detail and what (if
anything) remains open in `plex-ingest` itself (framework choice, full data
migration verification).
**Started:** 2026-07-05, cut over 2026-07-06

## What this is

Before this epic, `plex-rag` contained two largely independent halves in
one repo: a **pipeline** half (Plex sync, synopsis scraping, LLM
enrichment, embedding into Qdrant — previously documented in
`docs/pipeline.md`, deleted in phase 7 since that code no longer lives
here) and a **recommender** half (the multi-retriever RAG chat/web UI,
documented in [recommender.md](../../recommender.md)). They shared a
SQLite file (`media_items.db`) and an on-disk Qdrant store, and ran from
the same CLI (`plex-rag`).

This epic split that monolith into two independently deployable services:

- **`plex-rag`** (this repo) — becomes recommender-only. Loses its pipeline
  code, its SQLite dependency, and its ability to write to Qdrant. Becomes a
  pure read-only consumer of a Qdrant collection it doesn't own.
- **`plex-ingest`** (new repo) — owns everything data-related: Plex
  polling, scraping, enrichment, and all writes to Qdrant. Rebuilt on
  Dagster, with pipeline stages modeled as assets. Internal storage
  (parquet, DuckDB, etc.) is an implementation detail of that repo — the
  only externally-visible artifact is the Qdrant collection.

The handoff point between the two repos is a single Qdrant collection,
governed by [vector-store-contract.md](../../vector-store-contract.md).

## Why

The two halves have genuinely different operational shapes: the pipeline is
a slow, rate-limited batch job (minutes per run, throttled against Gemini
and scraping targets) that should be orchestrated and scheduled; the
recommender is a low-latency interactive service. Bundling them in one repo
under one CLI made sense at small scale but conflates their deploy
lifecycles, dependencies (Playwright/browser automation has nothing to do
with chat), and now blocks moving the pipeline onto Dagster, which is a
fundamentally different execution model (asset-based, not CLI commands).

## Decisions made

These were resolved during planning (2026-07-05) and shape every phase
below — don't relitigate them without a reason:

1. **Qdrant runs as a Docker container (server mode), not on-disk.** Two
   separate processes/repos can't share an on-disk Qdrant path, so it has
   to become networked. The persistent volume and `docker-compose.yml` live
   in `plex-ingest` (the data-owning repo); `plex-rag` connects to it over
   the network and does not start/manage the container itself.
2. **`plex-rag` drops direct catalog storage entirely.** No SQLite, no
   DuckDB/parquet read on the recommender side. Every point payload already
   duplicates all catalog fields (title, year, rating, genres, thumb_url)
   and — confirmed during investigation — `langchain_qdrant`'s default
   `content_payload_key="page_content"` means synopsis text is retrievable
   straight from the payload too. Qdrant is `plex-rag`'s only external data
   dependency. See [vector-store-contract.md](../../vector-store-contract.md).
3. **Shared contract is docs-only, not a shared code package.** A handful
   of values (embedding model name, collection name, payload field names)
   must match across repos. Given how different the two stacks are about to
   become (Dagster vs. a chat CLI/Streamlit app), a shared Python package
   would re-couple the repos at the code level for very little payoff. Kept
   in sync manually via `vector-store-contract.md`, with a cheap runtime
   check (vector size validation) as a tripwire against silent drift.
4. **New repo name: `plex-ingest`.**
5. **Build and prove out `plex-ingest` before removing anything from
   `plex-rag`.** `plex-rag` keeps running exactly as it does today (on-disk
   Qdrant, SQLite, the existing pipeline CLI commands) until `plex-ingest`
   is a verified, working replacement producing a matching Qdrant
   collection. Only then do phases in `plex-rag` start changing/removing
   code. There is no intermediate state where neither repo is capable of
   maintaining the vector store. See [breakdown.md](breakdown.md) — phases
   1-3 are `plex-ingest`-only and touch this repo not at all; phases 4+ are
   the only ones that change `plex-rag`.

## Open / deferred questions

- **Qdrant data migration: resolved by re-embedding, not volume copy.** The
  full library was re-embedded directly into `plex-ingest`'s dockerized
  Qdrant rather than migrating the old on-disk `media_items_qdrant_db/`
  collection — simpler than verifying volume/snapshot compatibility across
  Qdrant deployment modes, at the cost of one full re-embed run. The old
  on-disk collection and `media_items.db` have been deleted from `plex-rag`
  (phase 7).
- **Dagster asset design for scrape/enrich/embed**: resolved and
  implemented — see [breakdown.md](breakdown.md) phase 2 and
  [phase-2-pipeline-design.md](phase-2-pipeline-design.md) for the detail.
  Still open: the LlamaIndex/LangChain framework choice for the
  enrichment/embedding stage — a joint decision, not one to make
  unilaterally.
- **Local dev ergonomics**: whether there's any tooling to make "spin up
  Qdrant + run the recommender against it" a one-command affair for local
  development, versus just documenting the manual two-repo dance.

## Docs in this epic

- [target-architecture.md](target-architecture.md) — before/after diagrams.
- [breakdown.md](breakdown.md) — phased task breakdown.
- [phase-2-pipeline-design.md](phase-2-pipeline-design.md) — phase 2
  design questions: partitioning, storage, deletion cascade, and
  automation semantics are all decided and implemented; the
  LlamaIndex/LangChain framework choice is still open.
- [../../vector-store-contract.md](../../vector-store-contract.md) — the
  living data contract (not epic-scoped; outlives this epic and should be
  copied into `plex-ingest` too).
