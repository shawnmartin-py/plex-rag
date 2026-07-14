# Tooling & Library Improvements (Brainstorm)

Notes from a dependency/code-quality pass over `app/` and `nicegui_app/`,
captured for future reference. Nothing here is scheduled or committed to —
like [docs/feature-ideas.md](feature-ideas.md), this is a parking lot, not a
plan. Each item below has enough context (file/line, current behavior, why
it matters, concrete next step) to be picked up independently without
re-deriving the analysis.

The codebase already has good bones for this kind of work: a real
ports/adapters split (`app/domain/` never imports LangChain or pydantic
except at the schema boundary), strict mypy, and ruff with bandit/bugbear
already enabled. None of these items propose new architecture — they either
reuse a dependency already in `pyproject.toml` or fix a concrete bug found
during the pass.


## 2. `tenacity` for retrying Gemini/Qdrant calls

**Current state**: no retry or backoff logic anywhere in the repo (checked
via grep for `retry`/`backoff`/`tenacity`). Every Gemini call
(`app/adapters/generators.py`, `app/adapters/retrievers.py`'s
`HyDEVectorRetriever`/`LLMKnowledgeRetriever`) and every Qdrant call
(`app/repositories/vector_store.py`) propagates a transient failure (429
rate limit, brief network blip) straight up and kills the whole turn — the
user sees an error for something that would likely succeed on a second try
half a second later.

**Why it matters**: this is an interactive chat app calling an external LLM
API on literally every turn (`MovieRecommender.recommend` in
`app/domain/recommender.py` fans out `asyncio.gather` across 3-4 retrievers
per turn, each hitting Gemini or Qdrant) — it's the single largest
reliability gap relative to how often external calls happen.

**Concrete step**: add `tenacity` to `pyproject.toml`. Wrap the retryable
boundary calls — `ChatGoogleGenerativeAI`/`GoogleGenerativeAIEmbeddings`
invocations in `app/adapters/generators.py` and `app/adapters/retrievers.py`,
and `QdrantClient` calls in `app/repositories/vector_store.py` — with
`@tenacity.retry(...)` using `stop_after_attempt` + `wait_exponential`,
scoped to the specific transient exception types those clients raise (check
`google.api_core.exceptions` for Gemini, `qdrant_client.http.exceptions` for
Qdrant — don't blanket-catch `Exception`, since
`_connect_and_validate`'s existing fail-fast-on-missing-collection behavior
in `vector_store.py` should stay fail-fast, not retry). Keep retries out of
`_connect_and_validate`'s startup validation path specifically — that one
should still fail fast and loud, per its existing docstring intent.

## 6. Vectorize candidate scoring in `app/domain/diversity.py` (no new library)

**Current state**: `DiversityRecommender.recommend`
(`app/domain/diversity.py`) computes
`scored = [(c, cosine_distance(aversion, c.vector)) for c in candidates]` —
a Python-level loop calling `cosine_distance` (which itself builds two
fresh `np.array`s and calls `np.linalg.norm` twice) once per candidate.
`candidates` is sourced from `load_synopsis_vectors`
(`app/repositories/vector_store.py`), which scrolls up to 10,000 points.

**Why it matters**: at current library sizes this is not a measured
bottleneck, but it's a pure-function, easily-isolated hot path that's a
single vectorized numpy call away from being O(1) numpy ops instead of O(n)
Python-level ones — worth doing opportunistically if the watch_history/
media_items collections grow, or if `DiversityRecommender.recommend`
latency is ever profiled and shows up here.

**Concrete step**: replace the per-candidate loop with a single matrix
operation — stack `candidates` vectors into one `(n, d)` matrix, compute
`norms = np.linalg.norm(matrix, axis=1)`, `aversion_norm =
np.linalg.norm(aversion)`, then
`cosine_sim = (matrix @ aversion) / (norms * aversion_norm)`,
`distances = 1.0 - cosine_sim`. Re-pair with `candidates` via `zip` to
rebuild the `scored` list shape the rest of the function (`_distance_band`,
`_mmr_select`) already expects, so this stays a localized change that
doesn't ripple into their signatures.

## 7. LangSmith tracing (optional, no code changes)

**Current state**: the app already runs entirely on `langchain` /
`langchain-google-genai` (`app/adapters/generators.py`,
`app/adapters/retrievers.py`), but no `LANGCHAIN_TRACING_V2` or LangSmith
env vars are set anywhere (checked via grep — none found).

**Why it matters**: with four retrievers fanning out per turn
(`DirectSynopsisRetriever`, `HyDEVectorRetriever`, `LLMEnrichmentRetriever`,
optionally `LLMKnowledgeRetriever` — see `app/bootstrap.py`'s
`build_recommender_service`), debugging *why* a given film did or didn't
surface currently means reading the CLI's `--verbose` coverage report
(`app/rag.py`'s `_print_coverage`, backed by `CoverageReport` in
`app/domain/recommender.py`) after the fact. LangSmith tracing would give
per-retriever, per-LLM-call visibility (prompts, latencies, token counts)
without touching application code, since LangChain's tracing hooks are
already present in every chain (`ChatPromptTemplate | llm | ...` pipelines
throughout `app/adapters/`).

**Concrete step**: no code change — set `LANGCHAIN_TRACING_V2=true`,
`LANGCHAIN_API_KEY`, and optionally `LANGCHAIN_PROJECT` in the environment
(README's env var section is the natural place to document this). Purely
opt-in/observability; skip if there's no appetite for a LangSmith account.
