# Recommender: Conversational Movie Recommendations

This is the query-time half of the project: a multi-retriever RAG pipeline
that turns a natural-language request into ranked movie recommendations drawn
only from the user's own Plex library. It's exposed three ways — a CLI chat
loop, a NiceGUI web UI, and a FastAPI HTTP API — all of which wire up the
same underlying pieces.
All three entry points read **exclusively from the Qdrant collection**
`plex-ingest` owns, connecting via `QDRANT_URL` (server mode) per
[docs/vector-store-contract.md](vector-store-contract.md) — there is no
SQLite/`SqlMediaItems` dependency and no write path here anymore (that was a
temporary bridge during the `plex-ingest` extraction; it's gone as of the
cutover to the networked Qdrant collection).
On startup, `connect_vector_store` (`app/repositories/vector_store.py`)
does a preflight check — reachable, collection exists, vector size matches
`gemini-embedding-001` — and fails fast with a clear message if not; it never
tries to create or repair the collection itself.

A second recommendation mode — recommending movies *farthest* from recent
watch history rather than nearest to a query, exposed as `plex-rag
surprise` and the web UI's "Surprise me" button — is described in
[docs/diversity-recommender.md](diversity-recommender.md).

## Entry points

- **CLI** — `plex-rag chat [--no-spoilers] [--verbose]`
  (`app/cli.py:chat` → `app/rag.py:main`). Runs a blocking `input()` loop in
  the terminal.
- **Web UI** — `plex-rag-web` (`nicegui_app/main.py:main`, registered as a
  console script in `pyproject.toml`). `nicegui_app/service_cache.py`
  is a thin module-level cache (keyed by `spoiler_free` toggle value) around
  the same builder the CLI uses; `nicegui_app/main.py` drives the chat UI and
  per-tab state, `nicegui_app/components.py` renders each response.
- **API** — `plex-rag-api` (`api_app/main.py:main`, registered as a console
  script, serves on port 8100 — distinct from NiceGUI's 8080 so both can run
  at once — bound to `0.0.0.0` so LAN clients like the `plex-tvos` app can
  reach it). Built for a native tvOS client, but framework-agnostic — any
  HTTP client works. `api_app/service_cache.py` mirrors
  `nicegui_app/service_cache.py`'s pattern but keys its cache by
  `(session_id, spoiler_free)` instead of `spoiler_free` alone, since API
  clients don't share NiceGUI's browser-tab/`app.storage` machinery: each
  caller supplies its own `session_id` and gets its own
  `ConversationalRecommendationService` and chat history, never shared across
  sessions the way NiceGUI intentionally shares across tabs. Endpoints:
  - `POST /chat` — `{session_id, message, spoiler_free?}` → `{answer, items}`,
    a thin wrapper over `ConversationalRecommendationService.chat_with_items`.
  - `POST /chat/reset` — `{session_id}` → `{reset: bool}`, clears that
    session's chat history.
  - `GET /health` — liveness check.

  No streaming endpoint yet (unlike the web UI's `chat_with_items_stream`) —
  not needed until a client actually wants partial results. No `surprise`/
  diversity endpoint yet either; add one the same way if/when a client needs
  it, following the `/chat` handler as the template.

All three entry points call `build_recommender_service` (`app/bootstrap.py`) — the
single composition root that constructs the Gemini clients, connects to
Qdrant via `connect_vector_store` + `load_synopsis_documents`
(`app/repositories/vector_store.py`), wires up the retriever stack,
and returns a `ConversationalRecommendationService` plus a `QdrantMediaItems`
lookup. Movie titles and per-film `MediaItem`s (for `chat_with_items`'s
poster/rating display) are both derived from the same
`embedding_type=synopsis` Qdrant points via `QdrantMediaItems`
(`app/repositories/qdrant_media_items.py`), which implements the
`MediaItemLookup` port — no local database read anywhere in this path. The
CLI passes `include_knowledge_retriever=True` to also wire in
`LLMKnowledgeRetriever` (which needs the full movie title list) since
terminal usage isn't latency-sensitive in the same way; the web UI and the
API both leave it off (the default) for a snappier response — a voice-driven
tvOS client is exactly the kind of interactive, latency-sensitive caller the
web UI's default was already tuned for.

## Architecture (`app/domain/`, `app/adapters/`, `app/services/`, `app/repositories/`)

This follows a small ports-and-adapters split:

- `app/domain/ports.py` — abstract interfaces: `CandidateRetriever`,
  `QueryRewriter`, `RecommendationGenerator`, `MediaItemLookup`.
- `app/domain/recommender.py` — `MovieRecommender`, the orchestrator. Knows
  nothing about Gemini or Qdrant specifically, only the port interfaces.
- `app/adapters/retrievers.py` / `app/adapters/generators.py` — concrete
  Gemini/Qdrant implementations of those ports.
- `app/services/recommendation.py` — `ConversationalRecommendationService`,
  which owns chat history across turns.
- `app/repositories/` — read-only Qdrant data access: `qdrant_media_items.py`
  (`QdrantMediaItems`, implements `MediaItemLookup`) and `vector_store.py`
  (`connect_vector_store` + `load_synopsis_documents`, the Qdrant
  connection/schema-validation/document-loading helpers used by the
  composition root).

### `MovieRecommender.recommend` (`app/domain/recommender.py`)

Per turn:

1. **Rewrite** — if there's prior history, `QueryRewriter.rewrite` turns a
   follow-up ("what about something shorter?") into a standalone query using
   the conversation so far. Skipped on the first turn.
2. **Retrieve** — every configured `CandidateRetriever` runs against the
   (rewritten) query, each returning a list of `Document`s. All retrievers
   are independent of each other, so `MovieRecommender.recommend` fans them
   out concurrently via `asyncio.gather` rather than awaiting them one at a
   time.
3. **Group** — `_group_docs` dedupes by `(imdb_id, embedding_type, section)`
   and merges all documents for the same film into one bucket, while also
   tracking which retriever(s) surfaced each film (`sources`) — this is what
   powers `--verbose` coverage reporting.
4. **Format** — `_format_grouped` renders one context block per film
   (`=== Title (Year) ===` + its documents), synopsis first then
   craft/meaning/context in order, with **films shuffled** to avoid position
   bias in the generator's ranking.
5. **Generate** — `RecommendationGenerator.generate` (Gemini, via
   `with_structured_output`) produces a `RecommendationResponse`
   (`app/domain/ports.py`): optional `intro`/`closing_note` prose plus a list
   of `RecommendationCard(imdb_id, body_md)`. Each context block in step 4
   carries its film's `imdb_id` (`[imdb_id: tt1234567]`), and the model is
   instructed to copy it exactly into the matching card — a typed field
   instead of a hidden `<!-- imdb:tt1234567 -->` comment the app used to
   parse back out of free text (see
   [plan-structured-recommendation-output.md](plan-structured-recommendation-output.md)
   for why that changed). No parsing or fuzzy title-matching is needed to
   know which film a card is about.
6. **Filter and build the answer** — `MovieRecommender` drops any card whose
   `imdb_id` isn't actually one of the grouped candidates (the model
   hallucinated it rather than copying from context — same shape as
   `LLMKnowledgeRetriever`'s title filter), then synthesizes a numbered
   heading (`N. **Title** (Year)`) for each surviving card from `grouped`'s
   own metadata to build the plain-text `answer` string used by the CLI and
   conversation history. The web UI never sees these headings — it renders
   title/year from the matched `MediaItem` and only displays `body_md`.

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
- `GeminiRecommendationGenerator` — the main response chain, built on
  `llm.with_structured_output(RecommendationResponse)` rather than plain text.
  Two guideline variants baked in at construction time: normal vs.
  `spoiler_free=True` (forbids plot details / twists / outcomes, reasons only
  from genre/tone/pacing/cast/style). Both variants share the same hard
  constraint: **never recommend a film outside the provided context**, rank
  best-match-first, and explicitly acknowledge weak matches rather than
  oversell them. `stream()` does its own boundary detection over Gemini's
  growing partial `RecommendationResponse` objects — a card is only
  guaranteed finished once the list has grown past it, since JSON generation
  is append-only — turning that growth into discrete `SectionReady`/
  `TextDelta` events (`app/domain/ports.py`) rather than raw text chunks.

### Conversation state (`app/services/recommendation.py`)

`ConversationalRecommendationService` holds `history: list[BaseMessage]` (plain
Python list, in-memory, per-process/per-session — not persisted). Two entry
methods:

- `chat(question, verbose=False)` → str — used by the CLI.
- `chat_with_items(question, media_repo)` → `(str, list[MediaItem])` — used by
  the web UI; resolves the recommender's returned `imdb_ids` back to full
  `MediaItem`s (for posters/ratings) via `media_repo.get_by_id`, where
  `media_repo` is a `QdrantMediaItems` instance (any `MediaItemLookup` works).

`reset_history()` clears history — wired to the web UI's "New conversation"
button.

## NiceGUI UI specifics (`nicegui_app/`)

- `service_cache.py:get_service` is a module-level, process-lifetime cache
  keyed by `spoiler_free`, the direct equivalent of the previous Streamlit
  `@st.cache_resource`-per-argument behavior. **This means every browser tab
  sharing a `spoiler_free` value gets the same `ConversationalRecommendationService`
  instance, including its chat history** — one tab's turn becomes context for
  another tab's next answer, and "New conversation" in any tab clears history
  for every tab sharing that setting (though each tab's *displayed* transcript,
  stored in `app.storage.tab`, stays independent). This is a pre-existing
  behavior carried over unchanged from the Streamlit implementation, not a bug.
- `components.py:render_recommendations` splits the persisted `answer` string
  (its numbered headings synthesized by `MovieRecommender`, not written by the
  model — see step 6 above) into numbered sections via regex
  (`app/formatting/sections.py:parse_sections`, framework-agnostic and shared
  with any future front end), peeling off any trailing "Summary" / "Note"
  blocks, then pairs each numbered section **positionally** with the
  `MediaItem` list returned alongside the answer (not by title text-matching —
  see the "fix wrong movie posters" commit for why: title-matching was fragile
  against sequels/reboots with shared titles). Each pairing renders as a
  poster + reasoning card built from `ui.row`/`ui.column` rather than any
  built-in chat-bubble widget, to keep the flat, borderless look of the
  original UI (see `nicegui_app/styles.py` for the full rationale).

## Configuration

All three entry points read `QDRANT_URL` / `QDRANT_COLLECTION` from
`app/config.py`, and construct `ChatGoogleGenerativeAI` with
`gemini-3.1-flash-lite` (temperature 0, all four Gemini safety categories set
to `BLOCK_NONE` — film content routinely trips default safety thresholds) and
`GoogleGenerativeAIEmbeddings` with `gemini-embedding-001`, matching the
pipeline's embedding model exactly (embeddings from a different model would
not be comparable in the same Qdrant collection).

### `FAKE_GEMINI` — local testing without Gemini quota

Setting `FAKE_GEMINI=true` makes `build_recommender_service` and
`build_diversity_service` (`app/bootstrap.py`) swap both Gemini clients for
`FakeChatModel`/`DeterministicEmbeddings` (`app/adapters/fake_gemini.py`)
instead of `ChatGoogleGenerativeAI`/`GoogleGenerativeAIEmbeddings`. Every
other component — retrievers, `MovieRecommender`, the services, Qdrant — runs
exactly as it does in production, so this exercises the CLI, web UI, and API
end to end (session handling, coverage reporting, streaming, UI card
rendering, poster/rating lookups) with real candidates from the real Qdrant
collection, without spending Gemini quota or needing `GOOGLE_API_KEY` set at
all.

The fakes are deterministic, not realistic: `DeterministicEmbeddings` hashes
text into unit vectors with no real semantic relationship, so Qdrant still
returns *some* nearest neighbors but not relevant ones, and
`FakeChatModel`'s recommendation cards are templated bodies built from
whichever real `[tmdb_id: ...]` candidates the (semantically meaningless)
retrieval actually surfaced — not real reasoning about fit. Recommendation
*quality* can't be evaluated this way; only the surrounding plumbing can.
This is why `evals/` (`evals/faithfulness_eval.py`, `evals/judge.py`) never
reads `FAKE_GEMINI` — they build their own real Gemini clients independently
of `app/bootstrap.py`, deliberately.

Add a fake for a new `with_structured_output` schema in
`app/adapters/fake_gemini.py`'s `FakeChatModel.with_structured_output` if one
is ever added beyond `RecommendationResponse`/`TitleSelection` — it raises
`NotImplementedError` for anything else rather than returning something
structurally wrong.
