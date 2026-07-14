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
from `build_recommender_service`, not folded into it, because a missing
`watch_history` collection must disable this one feature rather than crash
the whole app — see that function's own docstring for the mechanics.

## Outlier wildcard (2026-07-12)

Candidates beyond `band_high_percentile` (the true distance outliers)
aren't discarded outright: each `recommend()` call has an independent
`outlier_wildcard_probability` (default 0.15) chance of pulling one extra
pick from that excluded tail into the pool before MMR narrows to `k` —
added so a stable, extreme-outlier title isn't permanently excluded from
ever surfacing. Full mechanics and why it's a flat coin flip rather than a
blended softmax weight: `DiversityRecommender`'s class docstring in
`app/domain/diversity.py` — not duplicated here.

## MMR weight tuning (2026-07-13)

`mmr_diversity_weight` (`DiversityRecommender.__init__`,
`app/domain/diversity.py`) was raised from its original 0.5 to 0.6. At 0.5,
a chain of picks within one `recommend()` call could each look locally
diverse — MMR's `diversity_penalty` only repels from *prior picks in the
same call*, with no memory of the original aversion vector — while
drifting back around into territory similar to the watch history itself
(e.g. watched=blue, chain yellow → red → green → turquoise, where
turquoise ends up close to blue again). Raising the weight toward
`relevance` re-anchors every pick to the fixed aversion vector instead of
just the previous pick.

That fix has a tradeoff in the other direction: weighting `relevance`
higher lowers `diversity_penalty`'s effective weight, which increases the
risk of two mutually-similar candidates (e.g. two sequels from the same
franchise) both landing in one result list. This is sharper than it sounds
because `_softmax_sample` (the pre-MMR pool-selection step) has zero
similarity-awareness of its own — MMR is the *only* backstop against
same-list near-duplicates. 0.6 was chosen as a middle ground: still favors
escaping the aversion vector over intra-batch diversity, but keeps more of
the anti-duplicate force than a more aggressive value would.

Neither the drift problem nor the duplicate-risk tradeoff has been
empirically validated against real watch history/candidate data — both are
reasoned from the formula, not measured. The validation script to do that
(sample many `recommend()` calls; check pairwise similarity within each
returned list for duplicate risk; check drift-toward-aversion-vector across
sequential calls) hasn't been written yet.

## Open items

- The MMR weight tuning above is unvalidated — see that section for the
  specific validation script that would settle it.
- Distance-band percentiles and softmax parameters (`DiversityRecommender`'s
  other defaults) are similarly unvalidated — worth revisiting once there's
  real usage to tune against.
- No LLM-generated commentary per card (unlike the main chat flow's
  `body_md`) — deliberately kept simple for v1; cards render with poster/
  rating/genre only, no generated blurb.
- The plex-ingest pipeline this depends on hasn't been exercised through
  Dagster's own sensors/scheduling yet (see that repo's CLAUDE.md
  "Environment gotchas" for why) — only verified via direct invocation.
  Once that's fixed, confirm the sensor-driven path produces the same
  data this was verified against.
