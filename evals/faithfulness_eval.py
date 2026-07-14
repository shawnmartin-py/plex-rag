"""First RAGAS metric wired into this project: Faithfulness, scored against
`MovieRecommender`'s real pipeline (real Qdrant-shaped retrieval, real Gemini
generation) run over the fixed golden corpus in `evals/golden_dataset.py`.

Why Faithfulness first, out of everything RAGAS offers: it's the one RAGAS
metric that maps directly onto a non-negotiable product invariant this app
already claims to enforce — "never recommend a film outside the provided
context, and don't oversell weak matches" (docs/recommender.md). The app
already deterministically guards *which films* get recommended (`grouped`
membership check in `app/domain/recommender.py`); Faithfulness is what checks
the part that guard can't: whether the *prose explaining why* a real film fits
actually says only things the retrieved context supports, rather than the
model inventing plot details or cast facts it "knows" but wasn't given.

Deliberately NOT part of `make test`/CI (see evals/README.md): this makes
real, billed Gemini calls (one generation + one judge call per golden case)
and its output is inherently a little noisy run to run, so it belongs in a
human-reviewed, explicitly-invoked tier, not a pass/fail gate every push.

Run: make eval  (needs GOOGLE_API_KEY; see evals/README.md)
"""

import asyncio
import statistics
import warnings

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_qdrant import QdrantVectorStore

from app.adapters.generators import GeminiQueryRewriter, GeminiRecommendationGenerator
from app.adapters.retrievers import (
    DirectSynopsisRetriever,
    HyDEVectorRetriever,
    LLMEnrichmentRetriever,
    LLMKnowledgeRetriever,
)
from app.domain.ports import CandidateRetriever
from app.domain.recommender import MovieRecommender
from evals.golden_dataset import DOC_BY_TITLE, GOLDEN_CASES, GOLDEN_CORPUS, MOVIE_LIST
from evals.judge import build_judge_llm

with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    from ragas.dataset_schema import SingleTurnSample
    from ragas.metrics import Faithfulness

GENERATION_MODEL = "gemini-3.1-flash-lite"  # matches app/bootstrap.py


def _build_recommender() -> MovieRecommender:
    """Wires the real retriever/generator stack from app/adapters against an
    in-memory Qdrant seeded with the golden corpus instead of the networked
    collection app/bootstrap.py connects to — everything downstream of that
    substitution (retrieval, grouping, generation) is the genuine
    production code path, not a stand-in."""
    embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
    llm = ChatGoogleGenerativeAI(model=GENERATION_MODEL, temperature=0)

    vector_store = QdrantVectorStore.from_documents(
        GOLDEN_CORPUS,
        embedding=embeddings,
        location=":memory:",
        collection_name="evals_media_items",
    )
    retrievers: list[CandidateRetriever] = [
        DirectSynopsisRetriever(vector_store, embeddings),
        HyDEVectorRetriever(vector_store, embeddings, llm),
        LLMEnrichmentRetriever(vector_store, embeddings),
        LLMKnowledgeRetriever(llm, MOVIE_LIST, DOC_BY_TITLE),
    ]
    return MovieRecommender(
        retrievers=retrievers,
        generator=GeminiRecommendationGenerator(llm),
        rewriter=GeminiQueryRewriter(llm),
    )


async def _run() -> int:
    recommender = _build_recommender()
    judge = Faithfulness(llm=build_judge_llm())

    scores: list[float] = []
    print(f"Running {len(GOLDEN_CASES)} golden cases against {GENERATION_MODEL}...\n")

    for case in GOLDEN_CASES:
        answer, mentioned_ids, context = await recommender.recommend_with_context(
            case.query, history=[]
        )
        print(f"--- {case.query!r}")
        if not answer.strip():
            print("    (empty answer — nothing to score, skipping)\n")
            continue

        # _format_grouped (app/domain/recommender.py) joins per-film blocks with
        # this exact separator — splitting back on it gives Faithfulness one
        # retrieved_context entry per film rather than one giant blob.
        retrieved_contexts = context.split("\n\n---\n\n") if context else []
        sample = SingleTurnSample(
            user_input=case.query,
            response=answer,
            retrieved_contexts=retrieved_contexts,
        )
        score = await judge.single_turn_ascore(sample)
        scores.append(score)
        print(f"    recommended: {mentioned_ids}")
        print(f"    faithfulness: {score:.2f}\n")

    if not scores:
        print("No scorable cases — nothing to report.")
        return 1

    mean = statistics.mean(scores)
    print(f"=== mean faithfulness over {len(scores)} case(s): {mean:.3f} ===")
    print(
        "No pass/fail threshold is enforced yet — see evals/README.md 'Next "
        "steps' for why. Read the per-case reasons above for anything "
        "surprisingly low before trusting this number."
    )
    return 0


def main() -> int:
    return asyncio.run(_run())


if __name__ == "__main__":
    raise SystemExit(main())
