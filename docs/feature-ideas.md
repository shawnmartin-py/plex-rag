# Feature Ideas (Brainstorm)

Notes from a brainstorming pass over the recommender pipeline, captured for
future reference. Nothing here is scheduled or committed to — this is a
parking lot of ideas grounded in the current architecture
([docs/recommender.md](recommender.md),
[docs/vector-store-contract.md](vector-store-contract.md)), not a plan.

## Low-effort wins (data already exists, just unused)

- **Structured filters** — `genres`, `content_rating`, `video_resolution`,
  `source_platform`, `year`, `imdb_rating` are all already in every Qdrant
  point's metadata, but nothing lets a user hard-filter on them (e.g. "PG-13
  or under," "only real downloads, not the Netflix/Disney+ stub clips")
  instead of relying on the prompt to honor it. Retrievers already build
  `Filter`/`FieldCondition` objects in `app/adapters/retrievers.py`, so this
  is mostly UI + passing extra conditions through — no new retrieval
  strategy needed.
- **Surface the coverage report in the web UI** — `--verbose` already
  produces a `CoverageReport` (which retriever(s) found each film, what got
  dropped) in `app/domain/recommender.py`, but it's CLI-only. An expandable
  "why this?" per card in NiceGUI, showing e.g. "matched via HyDE +
  enrichment" vs. "synopsis only," would make the quad-retriever design
  visible instead of a black box.
- **"Not interested" / already-seen exclusion** — the recommender has no
  concept of exclusion today. A per-card "seen it" / "not this" action that
  adds an `imdb_id` to a per-tab (or persisted) exclude-set, filtered out of
  `_group_docs` results, would stop the same film resurfacing every session.

## Building on the new ConversationStore

The DuckDB conversation history (`app/repositories/conversation_store.py`) is
new and currently only powers the Recent-conversations sidebar list. It
opens up:

- **Rename/delete/pin conversations** — `ConversationStore` only has
  `save`/`list_recent`/`get`; no `delete`. Trivial to add, and the sidebar
  already has the row UI to hang a delete icon off of.
- **True resume** — loading a Recent conversation is currently read-only
  (see `on_load_recent` in `nicegui_app/main.py` — "resuming isn't
  supported… no LLM/RAG context was restored"). Since `history` is just a
  `list[BaseMessage]`, it could be reconstructed from stored messages and
  handed back to `ConversationalRecommendationService` to actually continue
  a past thread instead of always forking a new one.
- **Search past conversations** — a simple title/content search box over the
  DuckDB table, useful once there's more history than the 10-row retention
  window shows at once.

## Personalization / feedback loop

- **Thumbs up/down per recommendation** — persisted alongside the
  conversation (extend `ConversationMessage`/`items` or a new small table),
  then fed back in as a lightweight signal — e.g. down-weighting
  genres/directors from disliked picks in future prompts, or surfacing a
  "you've liked 3 Kubrick-esque picks" affinity insight. This is the one
  place a taste profile could emerge without a new ML component — just
  accumulate signal and inject a summary into the generator prompt.
- **"Surprise me" button** — skips the chat input, sends a canned prompt
  like "pick something great I probably forgot I own." Useful given the
  library-only constraint makes undirected browsing awkward otherwise.

## Retrieval/architecture-adjacent ideas

- **Runtime-aware requests** — "something under 2 hours" isn't answerable
  today since runtime isn't in the payload/synopsis point at all. Would
  require a `plex-ingest` change (cross-repo contract change, out of scope
  for this repo alone — flag and coordinate before touching).
- **Multi-title comparison** — "why this over that" for two specific films
  already in context — mostly a generator prompt variant, no new retriever
  needed.
- **Household/multi-profile mode** — `ConversationalRecommendationService`
  is deliberately a single shared instance per `spoiler_free` value today.
  Separate taste profiles per person would need real per-user session
  state — a bigger architectural change, worth scoping carefully before
  committing.
