# Diversity Recommender ("Palette Cleanser")

**Status: implemented (2026-07-12)** — CLI (`plex-rag surprise`) and the
NiceGUI web app ("Surprise me" button). See `plex-ingest`'s
`docs/pipeline-design.md` (section "Watch-history diversity-recommender
pipeline") for the data-pipeline side this depends on, and
[docs/vector-store-contract.md](vector-store-contract.md) for the
`watch_history` collection schema.

## Concept

A second recommendation mode, distinct from the existing query-driven RAG
chat flow described in [docs/recommender.md](recommender.md): instead of
nearest-neighbor similarity to a user's typed query, this mode recommends
unwatched movies that are semantically **farthest** from what the user has
recently watched — a deliberate "mix it up" alternative to
`MovieRecommender`'s similarity search, e.g. following a heavy thriller
with something light. Takes no text query at all; driven entirely by watch
history. Implementation and exact mechanics: `app/domain/diversity.py`
(`DiversityRecommender`) — its own docstrings cover the distance-band/MMR/
softmax reasoning in detail; not duplicated here.

## Why this needs a new plex-ingest pipeline rather than a live call here

`plex-rag` has no Plex connection today, by design (see
[docs/recommender.md](recommender.md) / `README.md` — `plex-ingest` owns
the Plex connection). Computing watch-history embeddings live on every app
open would also risk `gemini-embedding-001`'s rate limits. See
`plex-ingest`'s `docs/pipeline-design.md` for the full investigation (what
Plex's watch-history API actually returns, why a short description is
sufficient embedding input).

`app/bootstrap.py:build_diversity_service` is a separate composition root
from `build_recommender_service`, not folded into it: the `watch_history`
collection is genuinely optional (depends on the plex-ingest pipeline
above having run) in a way `media_items` isn't, and a missing collection
here must disable this one feature, not crash the whole app.

## Outlier wildcard (2026-07-12)

Candidates beyond `band_high_percentile` (the true distance outliers) were
originally discarded outright — see `_distance_band`'s reasoning for why the
band excludes them (likely vector-space noise as often as genuinely great
contrasting picks). That has a real downside: a title that's a stable,
extreme outlier relative to the whole library could sit in that excluded
zone on essentially every call regardless of what the user watches next,
and never surface through this feature.

Each `recommend()` call now has an independent
`outlier_wildcard_probability` (default 0.15) chance of adding one extra
pick sampled from that excluded tail into the pool before MMR narrows to
the final `k`. Deliberately implemented as a flat per-call coin flip
rather than blending the tail into the same distance-scaled softmax as the
core band with a dampened weight — the latter was tried first and rejected
because the softmax's exponential term scales with the *raw* distance gap,
so a sufficiently extreme outlier can dominate regardless of how small its
weight is. A flat probability keeps "how rare" independent of how extreme
any given outlier's distance happens to be. A wildcard added to the pool
still isn't guaranteed a final slot — MMR only picks it if it wins on
relevance/diversity against the rest of the pool.

## Open items

- Distance-band percentiles, MMR/softmax parameters, and the new
  `outlier_wildcard_probability` (`DiversityRecommender`'s defaults) are
  unvalidated — worth revisiting once there's real usage to tune against.
- No LLM-generated commentary per card (unlike the main chat flow's
  `body_md`) — deliberately kept simple for v1; cards render with poster/
  rating/genre only, no generated blurb.
- The plex-ingest pipeline this depends on hasn't been exercised through
  Dagster's own sensors/scheduling yet (see that repo's CLAUDE.md
  "Environment gotchas" for why) — only verified via direct invocation.
  Once that's fixed, confirm the sensor-driven path produces the same
  data this was verified against.
