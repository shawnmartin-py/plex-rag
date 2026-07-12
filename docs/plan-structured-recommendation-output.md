# Plan: Structured Output for the Recommendation Generator

**Audience note for whoever picks this up:** this document is self-contained.
It was written for a fresh agent/model with no memory of the conversation
that produced it — it does not assume you've read anything except the files
it points to. Where a claim about library behavior is made, it is backed by
a concrete source-code reference so you can re-verify it against whatever
version is installed when you do the work (versions drift; verify before
trusting).

This is a **plan**, not a rubber-stamped spec. Section 8 lists real open
design decisions — make a call, document why, and proceed; don't stall on
them.

---

## 0. Context and precedent

This repo (`plex-rag`) is a movie-recommendation chatbot that only
recommends films the user actually has in their Plex library. See
[recommender.md](recommender.md) for the pipeline architecture:
rewriter → parallel retrievers → generator, with both a CLI (`app/rag.py`)
and NiceGUI web front end (`nicegui_app/`) sharing one composition root
(`app/bootstrap.py`).

A prior change in this same vein already landed:
`LLMKnowledgeRetriever` (`app/adapters/retrievers.py`) used to ask an LLM
for a JSON array of movie titles, then hand-parse the response (strip code
fences with regex, `json.loads`, silently contribute zero candidates on
parse failure). It was rewritten to use
`llm.with_structured_output(TitleSelection)` — a `pydantic.BaseModel` with
a `titles: list[str]` field — which pushes the parsing/validation into
LangChain and the model provider, deleting the regex and the
silent-failure branch entirely. That change is `TitleSelection` in
`app/adapters/retrievers.py` today — read it before starting; it's the
template this plan generalizes.

**This plan addresses the next, larger instance of the same anti-pattern:**
the recommendation generator's hidden "imdb marker" protocol. It's a bigger
lift than the retriever change — it touches the domain layer, both
adapters, the service layer, and both front ends — and it interacts with
streaming, which the retriever change did not have to consider. That
interaction is why this plan exists as its own document.

---

## 1. Current implementation (as of writing)

### 1.1 The marker protocol

`GeminiRecommendationGenerator` (`app/adapters/generators.py`) generates a
single free-form markdown response containing all recommendations. To let
the domain layer know *which film* each numbered recommendation is about,
the prompt instructs the model to embed a hidden HTML comment right after
each heading:

```python
# app/adapters/generators.py
_IMDB_MARKER_INSTRUCTION = (
    "- Immediately after each numbered heading line, on its own line, insert a "
    "hidden marker in the exact form `<!-- imdb:tt1234567 -->`, using that film's "
    "imdb_id exactly as given in its context block (`[imdb_id: ...]`). This marker "
    "must never be visible or mentioned as text — it exists only so the app can "
    "match your recommendation to the right film."
)
```

This instruction is folded into both `_RECOMMENDATION_GUIDELINES` and
`_SPOILER_FREE_GUIDELINES`, which build the system prompt
(`_SYSTEM_TEMPLATE`). The chain itself is just
`prompt | llm | StrOutputParser()` — plain text in, plain text out,
identical for both `generate()` (used by the CLI's non-streaming path) and
`stream()` (used by the web UI).

### 1.2 Parsing the marker back out

`app/domain/recommender.py` then has to recover structure from that text:

- `_MARKER_RE = re.compile(r"<!--\s*imdb:(tt\d+)\s*-->")` and
  `_MARKER_LINE_RE` (a variant that also matches surrounding whitespace/
  newline, used to strip the marker line before display) — regex constants
  near the top of the file.
- `_strip_markers(response)` — removes marker lines from the text before
  it's shown to the user.
- `_find_mentioned_ids(grouped, response)` — extracts every marker's
  imdb_id from the full response, in order, deduplicated. **Falls back**
  to `_find_mentioned_ids_by_title(grouped, response)` — fuzzy
  title-substring search — for the rare case the model omits a marker.
- `_match_section_id(text, grouped, claimed)` — the streaming-path
  counterpart: for one already-closed section of text, prefer its marker;
  fall back to fuzzy title search among titles not already `claimed` this
  turn (claiming prevents two sections both matching the same title on a
  fuzzy hit).
- `_build_coverage_report(...)` — uses the recovered `mentioned_ids` to
  split `grouped` (all candidate docs from every retriever) into
  `recommended` vs. `dropped`, for the CLI's `--verbose` coverage table
  (`_print_coverage` in `app/rag.py`).

### 1.3 The streaming path specifically

`MovieRecommender.recommend_stream()` (`app/domain/recommender.py`) is
the part that has to detect recommendation *boundaries* incrementally,
since the model is still generating. It buffers raw text deltas from
`self._generator.stream(...)`, and after each delta re-runs
`parse_sections()` (`app/formatting/sections.py`) — a regex-based splitter
that detects numbered-heading boundaries (`^(?:#{1,4} *|\*{1,2})?\d+[.)]`)
and trailing "notes" blocks (`_NOTES_RE`, matching things like "A Note on…",
"In Summary…"). Everything before the *last* split point is guaranteed
complete (the model only starts a new section once the previous one is
done), so those parts get converted to `StreamEvent`s:

- `TextDelta(text)` — a finished block of plain prose (intro/outro text).
- `SectionReady(imdb_id, body_md)` — one finished numbered recommendation.
  `imdb_id` is resolved via `_match_section_id`;
  `strip_section_heading()` (`app/formatting/sections.py`) removes the
  redundant "1. **Title** (2019)" heading line, since the UI already
  renders title/year/rating from the matched `MediaItem`'s own metadata
  — see the docstring on `render_movie_card` in
  `nicegui_app/components.py`, which spells this out explicitly.

### 1.4 The service and UI layers

`ConversationalRecommendationService` (`app/services/recommendation.py`)
wraps `MovieRecommender` and maps `SectionReady` → `CardReady` (its own
dataclass, same shape but with the imdb_id already resolved to a full
`MediaItem` via `MediaItemLookup`, or `None` if resolution failed).
`chat_with_items_stream()` yields `ChatStreamEvent = TextDelta | CardReady`.

`nicegui_app/main.py` (around line 294) consumes that stream directly:

```python
async for event in streamed.events:
    with assistant_body:
        if isinstance(event, TextDelta):
            ui.markdown(event.text).classes("plex-msg-prose")
        elif isinstance(event, CardReady) and event.item is not None:
            render_movie_card(event.item, event.body_md, top_pick=top_pick)
            top_pick = False
```

The CLI (`app/rag.py`) uses the non-streaming `recommend()` path via
`ConversationalRecommendationService.chat()`, and separately prints a
`CoverageReport` table when `--verbose` is passed.

`ConversationStore` (`app/repositories/conversation_store.py`) persists
the final `answer` string and `items` as an opaque `list[dict]` — it has
no coupling to the marker format, so it needs no changes here beyond
whatever the new `answer`-string construction produces.

---

## 2. Why change it

Same failure shape as the retriever problem this generalizes from: the
model is asked to carry structured data (`imdb_id`) through a free-text
side channel (a hidden HTML comment), and the code defends against the
model not complying — with a **two-tier fallback** (marker → fuzzy title
match) that's more defensive surface than the retriever case had. A
dedicated `imdb_id` field on a structured schema, separate from the prose
body, removes the need for either tier: the model can't "forget" a field
it's required to fill in, and there's no text to parse in the first place.

Bonus deletions this unlocks, if the model no longer writes a heading
line at all (see §5 and the open question in §8.3): `strip_section_heading`
and its `_NUMBERED_RE` regex in `app/formatting/sections.py` become
unnecessary too — the model never needs to restate the title/year the UI
already owns.

---

## 3. Confirmed library research

Verified by reading the **installed package source directly**, not
documentation websites (websites drift faster than installed code, and
this repo's exact behavior depends on the exact installed version).
Versions at time of writing, from `importlib.metadata`:

```
langchain-core          1.4.0
langchain-google-genai  4.2.3
langchain                1.3.2
pydantic                  2.13.4
```

**Re-run this before trusting any claim below** — `pip show <pkg>` or the
snippet above — and re-read the referenced source if versions differ.

### 3.1 `with_structured_output` + `.stream()` is supported and documented

`ChatGoogleGenerativeAI.with_structured_output`
(`langchain_google_genai/chat_models.py`, method starts around line 3383
in the installed 4.2.3 source) defaults to `method="json_schema"` and its
docstring says explicitly:

> `'json_schema'` (recommended): Uses native JSON schema support for
> reliable structured output. Supports streaming with fully-parsed
> Pydantic objects.
> ...
> When streaming, emits fully-parsed objects of the specified schema type
> (not incremental JSON strings).

The older `method="function_calling"` path (tool-calling based) is now
explicitly called "discouraged" in the same docstring. This is the
**current recommended default**, not an experimental flag — but it is a
comparatively recent default (superseding function-calling), so verify
it's still current when you pick this up.

### 3.2 How the streaming mechanism actually works

Traced through the source, in order:

1. **Gemini still token-streams under JSON-schema mode.** The chat model's
   `_stream`/`_astream` methods aren't bypassed when
   `response_mime_type="application/json"` is set — real incremental
   generation happens, not "buffer the whole response, then emit once."
2. **`with_structured_output` builds `llm | parser`**, where for a
   Pydantic schema `parser = PydanticOutputParser(pydantic_object=schema)`
   (`langchain_google_genai/chat_models.py`, the `method in ("json_mode",
   "json_schema")` branch).
3. **`PydanticOutputParser` extends `JsonOutputParser`**, which extends
   `BaseCumulativeTransformOutputParser`
   (`langchain_core/output_parsers/transform.py`). Its `_atransform`
   accumulates every raw text chunk into `acc_gen`, and on **every new
   chunk** calls `self.aparse_result([acc_gen], partial=True)`. If that
   succeeds and differs from the previous parse, it yields the new parsed
   object.
4. **`parse_result(..., partial=True)`** on `PydanticOutputParser`
   (`langchain_core/output_parsers/pydantic.py`) calls the JSON parser,
   then `_parse_obj` (a Pydantic `model_validate`); on failure it returns
   `None` instead of raising, when `partial=True`.
5. **The underlying JSON parse is lenient**: `parse_partial_json`
   (`langchain_core/utils/json.py`) auto-closes unterminated string
   literals and open `{`/`[` so an in-progress, syntactically-incomplete
   JSON blob still parses as far as it validly can.

**Net effect:** as the model writes the response, you get a stream of
**progressively larger, fully-validated instances of the whole schema
object** — not a stream of discrete "this item just finished" events, and
not raw incremental JSON text.

### 3.3 The consequence for our use case: no free per-item events

If the schema were e.g. `items: list[RecommendationCard]`, `.stream()`
would emit the *entire* `items` list every time it grows or its last
element's string field lengthens — including many emissions where the
last item's `body_md` is still visibly mid-sentence. **There is no
built-in "item N is now complete" signal.** To reconstruct that (which we
need, to know when to flush a `CardReady`/render a card), the application
still has to detect it itself — e.g., "the list just grew from N to N+1
items, therefore item N-1 is done and safe to finalize; the last item in
the list is always considered in-progress until the list grows again or
the stream ends." This is conceptually the same kind of boundary-detection
work `parse_sections()` already does today on markdown text — the
substrate changes from regex-on-markdown to diffing-a-growing-list, but
the *problem* (deciding when something is "done enough to render") doesn't
disappear.

**One clear improvement, though:** if `imdb_id` is declared *before*
`body_md` in the per-card schema, it becomes valid — and therefore
present in the parsed object — as soon as the model emits that key,
typically well before the prose for that card is finished. That's
strictly better than today, where nothing is known about a card until the
whole section (heading + marker + full body) is parsed out of finished
text. It means the UI could resolve and show a card's poster/rating while
its write-up is still visibly streaming in underneath — new capability,
not just a refactor.

### 3.4 What this means for `generate()` vs `stream()`

- **`generate()` (CLI's non-streaming path):** trivial change.
  `await self._chain.ainvoke(...)` on a structured-output chain just
  returns the final validated object directly — no parsing, no markers,
  no fallback.
- **`stream()`:** needs new boundary-detection logic (see §5, §8.2) to
  turn "the growing structured object" into discrete events, replacing
  `parse_sections()`'s job.

---

## 4. Files this touches (map before you start)

| File | What's there today | What changes |
|---|---|---|
| `app/adapters/generators.py` | `GeminiRecommendationGenerator`, `_IMDB_MARKER_INSTRUCTION`, `_RECOMMENDATION_GUIDELINES`, `_SPOILER_FREE_GUIDELINES`, `_SYSTEM_TEMPLATE` | New Pydantic schema; `generate()`/`stream()` rebuilt around `with_structured_output`; marker instruction deleted from guidelines |
| `app/domain/recommender.py` | `MovieRecommender`, `CoverageReport`/`CoverageEntry`, `_group_docs`, `_format_grouped`, marker regexes, `_find_mentioned_ids*`, `_match_section_id`, `_build_coverage_report`, `TextDelta`/`SectionReady`/`StreamEvent`/`StreamedAnswer` | Marker/regex machinery deleted; `recommend()`/`recommend_stream()` rebuilt to consume structured output; coverage-building simplified (mentioned ids come straight from the schema, no text parsing) |
| `app/formatting/sections.py` | `parse_sections`, `strip_section_heading`, `split_trailing_notes`, `_NOTES_RE`, `_NUMBERED_RE` | Likely deleted or heavily reduced — see §8.3 |
| `app/services/recommendation.py` | `ConversationalRecommendationService`, `CardReady`, `ChatStreamEvent`, `StreamedChatAnswer` | Probably unchanged in shape (still maps `SectionReady`-equivalent → `CardReady`), but verify against whatever `recommend_stream()`'s new event types look like |
| `app/rag.py` | `_print_coverage`, CLI main loop | `CoverageReport` consumption unchanged in *shape*, but verify field-population logic still lines up |
| `nicegui_app/main.py` (~line 294) | Consumes `TextDelta`/`CardReady` from `streamed.events` | Should need no changes if `ChatStreamEvent`'s shape is preserved — but this is the regression-test point for "did the streaming UX survive" |
| `tests/unit/test_recommender.py` (520 lines) | Heavy coverage of `_find_mentioned_ids`, `_match_section_id`, `_group_docs`, `CoverageReport`, `recommend_stream` boundary detection | Large rewrite — most marker/fallback-specific tests are deleted outright (dead code, like the retriever precedent), streaming-boundary tests rewritten against the new mechanism |
| `tests/unit/test_recommendation_sections.py` (85 lines) | Tests `parse_sections`/`strip_section_heading`/`split_trailing_notes` | Deleted or reduced per §8.3 outcome |
| `tests/integration/test_generators.py` (206 lines) | Tests `GeminiRecommendationGenerator` | Rewritten around structured-output chain, same pattern as `tests/integration/test_retrievers.py`'s `LLMKnowledgeRetriever` tests post-change |
| `tests/e2e/test_pipeline.py`, `tests/e2e/conftest.py` | `StubLLM`, full-pipeline fixtures | `StubLLM` already has a `with_structured_output` override from the retriever change (`tests/e2e/conftest.py`) — reusable as-is, but check it supports *streaming* structured output (see §8.4, it currently doesn't need to, verify before assuming) |
| `tests/integration/test_nicegui_main.py` (318 lines) | Web UI streaming render tests | Verify still pass; likely unaffected if `ChatStreamEvent` shape is preserved |

---

## 5. Proposed design (recommendation, not mandate — see §8 for the calls left open)

```python
# app/adapters/generators.py — sketch, not final
class RecommendationCard(BaseModel):
    imdb_id: str  # constrained to ids present in context; see §8.1 on enforcement
    body_md: str  # the "why it fits" bullets — no heading line, no marker

class RecommendationResponse(BaseModel):
    intro: str = ""          # optional lead-in prose before the first card
    cards: list[RecommendationCard]
    closing_note: str = ""   # optional trailing remarks / "nothing fits" case
```

Field order matters (§3.3): keep `imdb_id` before `body_md` on
`RecommendationCard` so identity resolves before prose does.

`generate()` becomes: build the chain as
`prompt | llm.with_structured_output(RecommendationResponse)`, `ainvoke`,
done — no `StrOutputParser`, no markers.

`stream()`/`recommend_stream()` needs a small state machine watching the
growing `RecommendationResponse.cards` list:

- On each yielded partial object, if `len(cards) > last_known_count`:
  the card at `last_known_count` (0-indexed) is now finalized — emit it
  as a `SectionReady`-equivalent event using the **previous** partial
  object's data for that index (not the current one, which is the *new*
  in-progress card). This mirrors `recommend_stream`'s existing comment
  about "everything before the last split point is guaranteed complete."
- `intro` (if non-empty and stable) can be flushed once as a `TextDelta`
  the first time `cards` becomes non-empty (i.e., once we're sure the
  model has moved past it).
- On stream completion, flush the final card and `closing_note` (if any)
  as the trailing `TextDelta`.
- The final `answer` string used for history/persistence needs to be
  reconstructed from the structured parts (see §8.3 for whether headings
  get re-synthesized here).

This is a sketch to seed the implementation, not a finished algorithm —
work through the edge cases (empty `cards`, a response with zero
recommendations, `closing_note` arriving before the last card finalizes
if the model emits fields out of declared order) before committing to it.

---

## 6. Implementation plan (suggested order)

1. **Read `app/adapters/retrievers.py`'s `TitleSelection`/
   `LLMKnowledgeRetriever` in full** — it's the precedent for schema
   style, chain construction, and how `cast()` was used to satisfy
   `mypy --strict` against `with_structured_output`'s broad
   `dict[str, Any] | BaseModel` return type. Reuse those idioms.
2. **Design and land the schema** (`RecommendationCard`,
   `RecommendationResponse`, or your revised version per §8) in
   `app/adapters/generators.py`. Update `_RECOMMENDATION_GUIDELINES`/
   `_SPOILER_FREE_GUIDELINES` to drop `_IMDB_MARKER_INSTRUCTION` and any
   "format each recommendation as a numbered item" instruction that's now
   redundant (schema enforces the shape; only the guidance content —
   *how* to write the analysis — should remain).
3. **Rebuild `GeminiRecommendationGenerator.generate()`** around
   `with_structured_output`. This is the low-risk half — do it first,
   get it passing tests, before touching `stream()`.
4. **Rebuild `GeminiRecommendationGenerator.stream()`** with the
   boundary-detection state machine from §5. This is the highest-risk
   piece — isolate it, test it heavily in isolation (feed it synthetic
   partial-object sequences, not just full LLM stub responses) before
   wiring it into `MovieRecommender`.
5. **Update `app/domain/recommender.py`**: delete the marker regexes,
   `_strip_markers`, `_find_mentioned_ids*`, `_match_section_id`. Rebuild
   `recommend()` (mentioned ids = `[c.imdb_id for c in response.cards]`,
   no parsing) and `recommend_stream()` against the generator's new event
   shape. Simplify `_build_coverage_report` accordingly — it no longer
   needs to reverse-engineer "what got mentioned" from text.
6. **Resolve §8.3** (heading/section-splitting fate) and update or delete
   `app/formatting/sections.py` accordingly.
7. **Verify `app/services/recommendation.py` and `nicegui_app/main.py`**
   need no changes (they consume `TextDelta`/`SectionReady`/`CardReady`
   by type, not by parsing text — if the new event types keep the same
   names/shapes, these layers may be untouched). If you change event type
   names/shapes, update both.
8. **Verify `app/rag.py`'s `_print_coverage`** still renders correctly —
   `CoverageReport`'s shape shouldn't need to change, only how it's built.
9. **Tests last, but not an afterthought** — see §7. Expect this to be
   the largest single piece of work in the whole change, given
   `tests/unit/test_recommender.py`'s current size.

---

## 7. Testing plan

- Follow the precedent in `tests/integration/test_retrievers.py`: bypass
  chain construction where reasonable (`retriever._chain = mock_chain`
  pattern), returning schema instances directly from mocks rather than
  raw text.
- `tests/e2e/conftest.py`'s `StubLLM.with_structured_output` currently
  parses one complete JSON response into one schema instance
  (`schema.model_validate_json(message.content)`) — fine for
  `generate()`/`ainvoke()`, but **`stream()`/`astream()` on that chain
  will not incrementally emit partial objects the way real Gemini does**,
  since the stub doesn't simulate token-by-token growth. Decide whether
  e2e tests need a second stub (e.g. `StubStreamingLLM` that yields a
  sequence of increasingly-complete JSON fragments) to actually exercise
  the boundary-detection logic end-to-end, or whether that logic is
  better covered by focused unit tests that hand `recommend_stream` (or
  whatever the new streaming entry point is) a synthetic sequence of
  `RecommendationResponse` partials directly, skipping the LLM layer
  entirely. The latter is likely cheaper and more precise — the risk
  being tested is "does our boundary-detection code do the right thing
  given this sequence of growing objects," not "does Gemini stream
  correctly" (that's Google's problem, not ours).
- Re-run the full suite (`python -m pytest -q`) and both `ruff check`
  and `mypy --strict` (see `.pre-commit-config.yaml` — this repo enforces
  pre-commit, not advisory) before considering this done. As of this
  writing the pinned ruff version is `0.15.16` (see
  `.pre-commit-config.yaml`) — the environment's plain `ruff`/`mypy` may
  not be on `PATH`; `uvx ruff@0.15.16 check <files>` and
  `python -m mypy <files>` (via the project's `.venv`) worked when this
  plan was drafted.
- Manually exercise the CLI (`app/rag.py`) and, per this repo's own
  CLAUDE.md standard for UI-adjacent changes, actually run the NiceGUI
  web app and drive a real streamed recommendation in a browser before
  calling this done — golden-path and at least one edge case (a request
  where nothing in the library is a good fit, exercising the
  `closing_note`/no-cards path).

---

## 8. Open decisions for the implementer

These are real design calls, not busywork — make a decision, write down
why in a code comment or commit message, and move on.

### 8.1 How to prevent the model from hallucinating an `imdb_id`

Today, "recommend only movies from the context above" is a prompt
instruction, unenforced beyond the marker/fuzzy-match recovery silently
dropping unmatched mentions. With a structured `imdb_id: str` field nothing
stops the model from inventing an id not in `grouped`. Options: (a) trust
the prompt instruction alone, same risk level as today, just without a
graceful silent-drop safety net — a hallucinated id would need to be
filtered post-hoc; (b) use a JSON-schema `enum` of the actual candidate
imdb_ids for this turn (dynamically built per-request from `grouped`,
since candidates differ every turn) — check whether
`with_structured_output` / Gemini's `response_json_schema` supports
per-call dynamic enums cleanly, since the schema would need to be
constructed fresh per request rather than declared once as a static
`BaseModel` class; (c) keep a thin post-hoc filter (`if imdb_id in
grouped`), same shape as today's `if t.lower() in self._doc_by_title` in
`LLMKnowledgeRetriever` — cheap, and doesn't require dynamic schemas.
(c) is likely the pragmatic default consistent with the retriever
precedent, but weigh (b) if hallucination turns out to be a real problem
in practice.

### 8.2 Exact boundary-detection algorithm for `stream()`

§5's sketch ("list just grew, so the previous item is done") is a
starting point, not verified against real Gemini streaming behavior.
Edge cases worth stress-testing before trusting it: what happens if the
model emits `cards` fields out of order (e.g. writes `body_md` before
`imdb_id` despite the schema declaring the reverse — nothing in JSON
enforces key order even if the Pydantic schema declares field order);
what happens if `intro`/`closing_note` interleave unexpectedly with
`cards` growth; whether `parse_partial_json`'s auto-closing behavior ever
produces a *spuriously* "valid" card with a truncated `imdb_id` (e.g.
`"tt00"` mid-write) that then gets treated as final before the id is
fully written. That last one specifically is worth a real test against
live output, not just reasoning about the parser — theory and an LLM's
actual token-chunking behavior can diverge.

### 8.3 Fate of headings and `app/formatting/sections.py`

§1.3 already established the UI doesn't need the model's heading text
(title/year come from `MediaItem`). But the CLI's non-`--verbose` output
today prints the model's raw markdown answer as-is, including its
numbered headings — that's the whole visual structure of the CLI
response. If the model stops writing headings (since the UI never needed
them), the CLI's plain-text output loses its numbering/structure unless
something re-synthesizes it (e.g., `app/rag.py` or
`ConversationalRecommendationService` numbering `response.cards` itself
when building the printed/stored `answer` string). Decide: keep the model
writing lightweight headings (simplest, least CLI-side change, but
retains a little of the "duplicated in two places" complaint) vs. strip
headings entirely and synthesize numbering at render time in both the CLI
and the `answer`-string builder (cleaner separation, more call sites
touched). Either way, `parse_sections`/`strip_section_heading`
(`app/formatting/sections.py`) are no longer doing the *boundary
detection* job (structured output replaces that) — the only question is
whether `strip_section_heading` specifically still has a job to do.

### 8.4 Whether `StubLLM` needs real incremental-streaming simulation

Covered in §7 — decide before writing tests, since it affects how much
new test-fixture machinery this change needs versus how much can be
covered by unit-testing the boundary-detection logic directly against
synthetic partial objects.

### 8.5 Whether `CoverageReport`'s "dropped" semantics still make sense

Today "dropped" means "in `grouped` context, but the marker/fuzzy-match
recovery didn't find it mentioned in the final text" — a proxy for "the
model saw it but chose not to recommend it," contaminated slightly by
recovery failures (a real mention the recovery missed would show as a
false "dropped"). With structured output, `mentioned_ids` is exact — no
recovery-miss noise. Confirm `_build_coverage_report`'s dropped-set
semantics are still what's wanted (they should be *more* accurate now,
which is a pure improvement, but double check `--verbose` output reads
sensibly with genuinely-exact data rather than the previous
best-effort-with-fallback data).

---

## 9. Non-goals / guardrails

- **Do not** also change `GeminiQueryRewriter` or `GeminiConversationTitler`
  (`app/adapters/generators.py`) as part of this work — both were
  evaluated in the research that produced this plan and judged poor
  candidates: their outputs are genuinely meant to stay free text (a
  standalone query fed back into embedding-based retrievers; a short
  human-readable sidebar title), not data consumed by typed code. Scope
  creep into those is out of bounds here.
- **Do not** touch `plex-ingest` (the sibling repo) or
  `docs/vector-store-contract.md` — this change is entirely within
  `plex-rag`'s recommender-only scope, no Qdrant schema implications.
- Per this repo's `CLAUDE.md`: no half-finished implementations, no
  premature abstraction beyond what this change needs, no defensive
  error handling for scenarios that can't happen. If §8.1's hallucination
  question leads you toward a filter, keep it as minimal as the
  retriever precedent's `if t.lower() in self._doc_by_title` — don't
  build a general-purpose validation framework for it.
- Pre-commit is enforced in this repo, not advisory — don't bypass hooks
  to land this.

---

## 10. Quick reference index

- Precedent change: `TitleSelection` / `LLMKnowledgeRetriever` in
  `app/adapters/retrievers.py`
- Marker protocol (to be deleted): `_IMDB_MARKER_INSTRUCTION` in
  `app/adapters/generators.py`; `_MARKER_RE`, `_MARKER_LINE_RE`,
  `_strip_markers`, `_find_mentioned_ids`, `_find_mentioned_ids_by_title`,
  `_match_section_id` in `app/domain/recommender.py`
- Streaming boundary detection (to be replaced):
  `MovieRecommender.recommend_stream` in `app/domain/recommender.py`,
  `parse_sections`/`strip_section_heading` in `app/formatting/sections.py`
- Service/UI consumers (verify, likely unchanged):
  `ConversationalRecommendationService` in
  `app/services/recommendation.py`; the streaming loop around line 294
  of `nicegui_app/main.py`; `render_movie_card` in
  `nicegui_app/components.py`
- CLI consumer: `_print_coverage` and `main` in `app/rag.py`
- Library internals backing §3's claims (re-verify against your installed
  versions):
  - `langchain_google_genai/chat_models.py` —
    `ChatGoogleGenerativeAI.with_structured_output`
  - `langchain_core/output_parsers/transform.py` —
    `BaseCumulativeTransformOutputParser._atransform`
  - `langchain_core/output_parsers/pydantic.py` —
    `PydanticOutputParser.parse_result`
  - `langchain_core/utils/json.py` — `parse_partial_json`
- Related docs: [recommender.md](recommender.md), [feature-ideas.md](feature-ideas.md)
