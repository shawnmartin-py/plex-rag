# RAGAS evals

A RAGAS-based evaluation harness for the recommender pipeline
(`app/domain/recommender.py` and everything it wires up) — a different tier
from `tests/` (see "Why not `tests/`" below). This is the first slice of it:
one metric (Faithfulness), one small hand-curated golden dataset, run
against the real pipeline with real Gemini calls. More metrics and a larger
dataset are meant to be added incrementally on top of this, not all at once
— see "Next steps".

## Run it

```bash
make eval-faithfulness   # needs GOOGLE_API_KEY; makes real, billed Gemini calls
```

Prints one faithfulness score (0–1) per golden query plus a mean at the end.
Read the per-case output, not just the mean — a single badly-scored case
tells you something concrete (which claim wasn't grounded); the mean alone
doesn't.

## Why Faithfulness first

RAGAS ships many metrics (context precision/recall, answer relevancy, noise
sensitivity, and more), most designed for classic factual-QA RAG. This app
isn't quite that: it's a recommender with an explicit, already-stated
product invariant (docs/recommender.md) — *never recommend a film outside
the retrieved context, and don't oversell weak matches*. The app already
enforces the first half of that deterministically (`MovieRecommender`
drops any card whose `imdb_id` isn't in `grouped` — see `_build_answer` in
`app/domain/recommender.py`). It has no check at all for the second half:
whether the prose explaining *why* a real, legitimately-retrieved film fits
actually confines itself to what the retrieved context says, rather than the
model padding out its reasoning with plot details, cast facts, or claims it
"knows" about the film but was never given. That's exactly what RAGAS's
Faithfulness metric measures — decompose the response into atomic
statements, check each one against the retrieved context, score the
fraction supported. It's the one RAGAS metric that maps onto something this
app already promises but doesn't yet verify, which is why it's the starting
point rather than, say, context precision (see "Next steps" for why that's
next, not first).

## Design decisions and why

**A fixed, hand-curated golden corpus (`evals/golden_dataset.py`), not a
live Plex library.** Ten real, well-known films with hand-written
synopsis/craft/meaning/context text in the exact shape
`docs/vector-store-contract.md` specifies, loaded into an in-memory Qdrant
(`QdrantVectorStore.from_documents(..., location=":memory:")`, the same
pattern `tests/e2e/conftest.py` already uses) rather than pointing at
`QDRANT_URL`. This makes every eval run reproducible and diffable in git —
a score changing because a golden case's answer changed, not because
someone's Plex library changed underneath the run. The pipeline code itself
(`app/adapters/retrievers.py`, `app/domain/recommender.py`) runs completely
unmodified against it.

**Real retrieval and real generation, not stubs.** Unlike `tests/e2e`
(`StubLLM`/`StubEmbeddings`, deliberately deterministic and free), this
harness is evaluating whether the *real* Gemini-backed pipeline is faithful
— stubbing either half would defeat the point. `evals/faithfulness_eval.py`
builds the same retriever/generator stack `app/bootstrap.py` does
(`DirectSynopsisRetriever`, `HyDEVectorRetriever`, `LLMEnrichmentRetriever`,
`LLMKnowledgeRetriever` — the CLI's full four-retriever set, not the web
UI's leaner three), just against the in-memory store instead of the
networked one.

**`MovieRecommender.recommend_with_context`, a small new method, not a
wider `recommend()` return.** Faithfulness needs the actual retrieved
context a given answer was generated from, and `recommend()` never exposed
it (only the CLI/web UI ever needed the final answer). Re-deriving an
approximation by calling the retrievers again separately would double the
API calls and risk silently drifting from what the generator actually saw.
Instead `app/domain/recommender.py` was refactored to share its
rewrite→retrieve→group→format step (`_retrieve_context`) between
`recommend`, `recommend_stream`, and a new `recommend_with_context` — a
small, additive seam, not a behavior change to any existing caller. Existing
tests in `tests/unit/test_recommender.py` cover that this refactor didn't
change `recommend`'s observable behavior.

**Why `LangchainLLMWrapper`, not ragas's `llm_factory`.** ragas 0.4.3's
current, actively-promoted API for building an evaluator LLM
(`ragas.metrics.collections.*` + `ragas.llms.llm_factory`) turned out not to
work cleanly for Google/Gemini in this project, for two separate, currently
open upstream reasons discovered while building this:

1. `llm_factory(provider="google")` auto-detects Google's default adapter
   as `litellm`. `litellm` ships a Rust extension (`litellm-rust`, built via
   PyO3) that fails to build on Python 3.14 (this project's pinned
   interpreter — PyO3 0.23.5's max supported version is 3.13). `litellm`
   simply cannot be installed here.
2. Passing a `google.genai.Client` directly instead (bypassing litellm)
   *does* get picked up correctly by ragas's Google-specific adapter
   (`instructor.from_genai`), but ragas's adapter code
   (`ragas/llms/adapters/instructor.py`) calls `instructor.from_genai(client)`
   without `use_async=True`, so the wrapped client is always synchronous —
   and every ragas `collections` metric's `.ascore()` requires an
   async-capable LLM (`.score()` isn't a workaround either; it just calls
   `asyncio.run(self.ascore(...))` internally, hitting the same wall).
   Confirmed live against a real `GOOGLE_API_KEY` — this isn't a docs
   misread, the call genuinely raises
   `TypeError: Cannot use agenerate() with a synchronous client`.

Both are real, reported upstream bugs
(github.com/explodinggradients/ragas/issues/2741, /2745, and the
`use_async` gap above), not something specific to this project's setup.

The fallback that works: the older, `@deprecated`-but-still-functional
classic API (`ragas.metrics.Faithfulness` + `ragas.llms.LangchainLLMWrapper`
+ `ragas.dataset_schema.SingleTurnSample`), wrapping the exact same
`ChatGoogleGenerativeAI` LangChain class `app/bootstrap.py` already uses for
generation (`evals/judge.py`). This sidesteps litellm, `google-genai`, and
`instructor` entirely — one dependency surface this repo already trusts,
reused for judging instead of a second, newly-introduced Google client.
Confirmed working live: a deliberately-planted hallucinated claim ("a car
chase on the moon") scored 0.5, a real claim scored 1.0. Revisit this once
ragas fixes the `use_async` gap and litellm supports Python 3.14 — check
back at that point rather than assuming the classic API stays available
indefinitely (it's already deprecated).

**`langchain-community<0.4` as a required, seemingly-unrelated dependency.**
`ragas` (all currently-released versions checked, back through 0.2.15)
unconditionally imports `langchain_community.chat_models.vertexai.ChatVertexAI`
at `ragas.llms` import time — even though `langchain-community` isn't a
declared `ragas` dependency at all, and even though this project doesn't use
Vertex AI. Recent `langchain-community` releases (0.4.x) moved `ChatVertexAI`
out to `langchain-google-vertexai`, breaking that import outright
(`ModuleNotFoundError`). Pinning `langchain-community<0.4` (`pyproject.toml`,
`eval` dependency group) is the tracked workaround
(github.com/explodinggradients/ragas/issues/2745) — needed just to make
`import ragas` succeed, regardless of which part of ragas's API is actually
used. Drop the pin once ragas fixes the import itself.

**A new `eval` dependency group, not `dev`.** `ragas` pulls in a genuinely
heavy dependency tree (pandas, pyarrow, datasets, instructor, openai,
sqlalchemy...) that has nothing to do with day-to-day development on this
repo. Keeping it out of `dev` means `make install`/`uv sync` and the mypy
pre-commit hook (which only installs `--group dev`) stay fast for everyone
who isn't actively working on evals. This works because
`ignore_missing_imports = true` (`pyproject.toml`'s `[tool.mypy]`) already
makes mypy tolerate `ragas` not being installed in that hook's environment —
confirmed `mypy .` passes strict-mode over `evals/` with the `eval` group
absent.

**No pass/fail threshold yet, and not part of `make check`/CI.** This run
makes real, billed Gemini calls and — like any LLM-judged metric — carries
some run-to-run noise even at `temperature=0` for both the generator and
the judge. Gating CI on it before seeing what a real score distribution
looks like across enough runs would mean picking a threshold out of thin
air, which is exactly the kind of shortcut this project's engineering
standards rule out. `make eval-faithfulness` is deliberately a
human-reviewed reporting tool for now, invoked explicitly, not a gate.

## Why not `tests/`

`tests/unit`, `tests/integration`, and `tests/e2e` all share one property:
deterministic, free, and fast enough to run on every push (`tests/e2e` uses
`StubLLM`/`StubEmbeddings`, not real Gemini calls — see
`tests/e2e/conftest.py`). RAGAS evals are the opposite on all three counts —
real API calls (cost + latency), an LLM judge (irreducible run-to-run
variance even at `temperature=0`), and no fixed "correct" answer to assert
equality against, only a graded score to read and interpret. Folding this
into `pytest`/CI would either make CI slow, flaky, and billed on every push,
or push people toward `pytest.mark.skip`-ing it into irrelevance. A
separate, explicitly-invoked `make eval-*` tier keeps it useful instead.

## Next steps

Roughly in order, each meant to be its own deliberate step, not bundled:

- **Retrieval-quality metrics (context precision/recall).** The app's own
  stated design bet is that a single retrieval strategy isn't enough
  (`docs/recommender.md`) — it runs four in parallel and merges them. RAGAS
  context precision/recall, computed *per retriever* rather than only on
  the merged/grouped result, could give real evidence for whether e.g. HyDE
  is actually pulling its weight versus mostly adding noise — a question
  this project can't currently answer except by eyeballing `--verbose`
  coverage output. This needs `GoldenCase` extended with a
  reference/expected-relevant-imdb_ids annotation per query, deliberately
  left out of `GoldenCase` for now (see its docstring) rather than added
  speculatively ahead of the metric that would consume it.
- **A larger, still-curated golden set.** Ten films and eight queries is
  enough to prove the harness end-to-end; it's not enough to trust a mean
  score's stability. Grow it once the first metric's signal has been sanity
  checked against real regressions, not before.
- **A defensible pass/fail threshold**, once enough runs exist to know what
  a "normal" score distribution actually looks like for this pipeline
  specifically, rather than copying a threshold from someone else's RAG
  system. At that point, promote `make eval-faithfulness` from a reporting
  tool into a real, non-blocking-CI quality gate (e.g. a scheduled run with
  alerting), not into `make check`.
- **Per-card, not whole-answer, Faithfulness.** This first pass scores one
  turn's full answer against all its retrieved context blocks together.
  Scoring each recommendation card against only *its own* film's context
  block would catch a subtler failure mode: a true claim about the wrong
  film. Needs `MovieRecommender` to expose per-card context, not just the
  turn-level blob `recommend_with_context` returns now.
- **Concurrent case execution.** Cases currently run sequentially — simplest
  to reason about while this harness is still new, but wasteful once the
  golden set grows; `asyncio.gather` across cases is a safe, mechanical
  speedup once the sequential version's output has been trusted for a
  while.
