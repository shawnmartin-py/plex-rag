# Recommender: Conversational Movie Recommendations

This is the query-time half of the project: a multi-retriever RAG pipeline
that turns a natural-language request into ranked movie recommendations drawn
only from the user's own Plex library. It's exposed two ways — a CLI chat loop
and a Streamlit web UI — both of which wire up the same underlying pieces.
Both entry points read **exclusively from the Qdrant collection**
`plex-ingest` owns, connecting via `QDRANT_URL` (server mode) per
[docs/vector-store-contract.md](vector-store-contract.md) — there is no
SQLite/`SqlMediaItems` dependency and no write path here anymore (that was a
temporary bridge during the `plex-ingest` extraction; it's gone as of the
phase-2 cutover — see
[docs/epics/plex-ingest-extraction/](epics/plex-ingest-extraction/README.md)).
On startup, `connect_vector_store` (`app/services/recommender_vector_store.py`)
does a preflight check — reachable, collection exists, vector size matches
`gemini-embedding-001` — and fails fast with a clear message if not; it never
tries to create or repair the collection itself.

## Entry points

- **CLI** — `plex-rag chat [--no-spoilers] [--verbose]`
  (`app/cli.py:chat` → `app/rag.py:main`). Runs a blocking `input()` loop in
  the terminal.
- **Web UI** — `streamlit run streamlit_app/main.py`. `streamlit_app/init.py`
  is a thin `@st.cache_resource`-cached wrapper (cached per `spoiler_free`
  toggle value) around the same builder the CLI uses; `streamlit_app/main.py`
  drives the chat UI and session state, `streamlit_app/components.py` renders
  each response.

Both entry points call `build_recommender_service` (`app/bootstrap.py`) — the
single composition root that constructs the Gemini clients, connects to
Qdrant via `connect_vector_store` + `load_synopsis_documents`
(`app/services/recommender_vector_store.py`), wires up the retriever stack,
and returns a `ConversationalRecommendationService` plus a `QdrantMediaItems`
lookup. Movie titles and per-film `MediaItem`s (for `chat_with_items`'s
poster/rating display) are both derived from the same
`embedding_type=synopsis` Qdrant points via `QdrantMediaItems`
(`app/repositories/qdrant_media_items.py`), which implements the
`MediaItemLookup` port — no local database read anywhere in this path. The
CLI passes `include_knowledge_retriever=True` to also wire in
`LLMKnowledgeRetriever` (which needs the full movie title list) since
terminal usage isn't latency-sensitive in the same way; the Streamlit UI
leaves it off (the default) for a snappier browser experience.

## Architecture (`app/domain/`, `app/adapters/`, `app/services/`)

This follows a small ports-and-adapters split:

- `app/domain/ports.py` — abstract interfaces: `CandidateRetriever`,
  `QueryRewriter`, `RecommendationGenerator`, `MediaItemLookup`.
- `app/domain/recommender.py` — `MovieRecommender`, the orchestrator. Knows
  nothing about Gemini or Qdrant specifically, only the port interfaces.
- `app/adapters/retrievers.py` / `app/adapters/generators.py` — concrete
  Gemini/Qdrant implementations of those ports.
- `app/services/recommendation.py` — `ConversationalRecommendationService`,
  which owns chat history across turns.

### `MovieRecommender.recommend` (`app/domain/recommender.py`)

Per turn:

1. **Rewrite** — if there's prior history, `QueryRewriter.rewrite` turns a
   follow-up ("what about something shorter?") into a standalone query using
   the conversation so far. Skipped on the first turn.
2. **Retrieve** — every configured `CandidateRetriever` runs against the
   (rewritten) query, each returning a list of `Document`s.
3. **Group** — `_group_docs` dedupes by `(imdb_id, embedding_type, section)`
   and merges all documents for the same film into one bucket, while also
   tracking which retriever(s) surfaced each film (`sources`) — this is what
   powers `--verbose` coverage reporting.
4. **Format** — `_format_grouped` renders one context block per film
   (`=== Title (Year) ===` + its documents), synopsis first then
   craft/meaning/context in order, with **films shuffled** to avoid position
   bias in the generator's ranking.
5. **Generate** — `RecommendationGenerator.generate` (Gemini) produces the
   final prose response, constrained by prompt to only recommend films present
   in the context.
6. **Extract mentions** — `_find_mentioned_ids` finds which grouped films are
   actually named in the response text (in first-mention order), so the UI can
   pair each numbered recommendation with its `MediaItem` (poster, rating).

### Retrievers (`app/adapters/retrievers.py`)

Four independent strategies, run every turn and merged (not chosen between):

| Retriever | `name` | Targets | Strategy |
|---|---|---|---|
| `DirectSynopsisRetriever` | `synopsis` | `embedding_type=synopsis` | Embeds the query directly, vector search. Best for plot-specific / meta queries (cast, language, content rating) where "critic vocabulary" doesn't help. |
| `HyDEVectorRetriever` | `hyde` | `embedding_type=enriched` | LLM writes a hypothetical dense expert film profile matching the request, *that* gets embedded and searched — surfaces films matching the critic vocabulary of the request rather than its literal words. |
| `LLMKnowledgeRetriever` | `llm-knowledge` | full title list (no vector search) | Sends the entire movie title list to Gemini and asks it to pick up to 8 by its own film knowledge (director, subgenre, cultural context). CLI-only; scales to a few hundred titles. |
| `LLMEnrichmentRetriever` | `enricher` | `embedding_type=enriched` | Embeds the query directly (no HyDE step) and searches enrichment documents — brings in retrieval signal that doesn't exist in synopses (cinematographer names, movement labels, tone words). |

All vector retrievers filter by `metadata.embedding_type` via Qdrant
`Filter`/`FieldCondition`, which is why the pipeline's dual embedding scheme
(synopsis vs. enriched, tagged per point) matters — see
[vector-store-contract.md](vector-store-contract.md).

### Generators (`app/adapters/generators.py`)

- `GeminiQueryRewriter` — single-turn LLM chain, system prompt instructs it to
  fold history into a standalone question, nothing else.
- `GeminiRecommendationGenerator` — the main response chain. Two guideline
  variants baked in at construction time: normal vs. `spoiler_free=True`
  (forbids plot details / twists / outcomes, reasons only from genre/tone/
  pacing/cast/style). Both variants share the same hard constraint: **never
  recommend a film outside the provided context**, rank best-match-first, and
  explicitly acknowledge weak matches rather than oversell them.

### Conversation state (`app/services/recommendation.py`)

`ConversationalRecommendationService` holds `history: list[BaseMessage]` (plain
Python list, in-memory, per-process/per-session — not persisted). Two entry
methods:

- `chat(question, verbose=False)` → str — used by the CLI.
- `chat_with_items(question, media_repo)` → `(str, list[MediaItem])` — used by
  Streamlit; resolves the recommender's returned `imdb_ids` back to full
  `MediaItem`s (for posters/ratings) via `media_repo.get_by_id`, where
  `media_repo` is a `QdrantMediaItems` instance (any `MediaItemLookup` works).

`reset_history()` clears history — wired to the Streamlit "New conversation"
button.

## Streamlit UI specifics (`streamlit_app/`)

- `init.py:build_service` is `@st.cache_resource`-cached per `spoiler_free`
  value, so toggling spoiler-free mode swaps to a distinct cached pipeline
  instance (and distinct chat history) rather than mutating one in place.
- `components.py:render_recommendations` splits the generator's markdown
  response into numbered sections via regex (`_parse_sections`), peeling off
  any trailing "Summary" / "Note" blocks, then pairs each numbered section
  **positionally** with the `MediaItem` list returned alongside the answer
  (not by title text-matching — see the "fix wrong movie posters" commit for
  why: title-matching was fragile against sequels/reboots with shared titles).
  Each pairing renders as a two-column poster + reasoning card.

## Configuration

Both entry points read `QDRANT_URL` / `QDRANT_COLLECTION` from
`app/config.py`, and construct `ChatGoogleGenerativeAI` with
`gemini-3.1-flash-lite` (temperature 0, all four Gemini safety categories set
to `BLOCK_NONE` — film content routinely trips default safety thresholds) and
`GoogleGenerativeAIEmbeddings` with `gemini-embedding-001`, matching the
pipeline's embedding model exactly (embeddings from a different model would
not be comparable in the same Qdrant collection).
