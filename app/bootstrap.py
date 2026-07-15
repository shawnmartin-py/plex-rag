import asyncio
import logging

from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings,
    HarmBlockThreshold,
    HarmCategory,
)
from langchain_google_genai._common import GoogleGenerativeAIError
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

from app.adapters.fake_gemini import DeterministicEmbeddings, FakeChatModel
from app.adapters.generators import (
    GeminiConversationTitler,
    GeminiQueryRewriter,
    GeminiRecommendationGenerator,
)
from app.adapters.retrievers import (
    DirectSynopsisRetriever,
    HyDEVectorRetriever,
    LLMEnrichmentRetriever,
    LLMKnowledgeRetriever,
)
from app.config import (
    FAKE_GEMINI,
    QDRANT_COLLECTION,
    QDRANT_URL,
    QDRANT_WATCH_HISTORY_COLLECTION,
)
from app.domain.diversity import DiversityRecommender
from app.domain.ports import CandidateRetriever, ConversationTitler
from app.domain.recommender import MovieRecommender
from app.repositories.candidate_pool import QdrantCandidatePool
from app.repositories.qdrant_media_items import QdrantMediaItems
from app.repositories.vector_store import (
    QdrantUnavailableError,
    connect_vector_store,
    load_synopsis_documents,
    load_synopsis_vectors,
    load_watch_history_points,
)
from app.repositories.watch_history import QdrantWatchHistory
from app.services.diversity_recommendation import DiversityRecommendationService
from app.services.recommendation import ConversationalRecommendationService

logger = logging.getLogger(__name__)

_SAFETY_OFF = {
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
}


@retry(
    retry=retry_if_exception_type(GoogleGenerativeAIError),
    stop=stop_after_attempt(3),
    wait=wait_exponential_jitter(initial=1, max=10),
    reraise=True,
)
async def _aembed_documents_with_retry(
    inner: Embeddings, texts: list[str]
) -> list[list[float]]:
    """`GoogleGenerativeAIEmbeddings`, unlike `ChatGoogleGenerativeAI`, doesn't
    configure `HttpRetryOptions` on its underlying `google-genai` client — a
    single transient error (rate limit, brief network blip) reaches
    `MovieRecommender._retrieve_context`'s `asyncio.gather` fan-out and kills
    the whole turn, since every retriever embeds the query on every turn.
    `GoogleGenerativeAIError` is a flat wrapper with no status code attached,
    so this can't distinguish retryable (429/5xx) from permanent errors —
    a capped, jittered retry is still a net win over zero retries."""
    return await inner.aembed_documents(texts)


class _DedupingEmbeddings(Embeddings):
    """Wraps an Embeddings instance so concurrent aembed_documents([text]) calls
    for the identical single-text query are coalesced into one API call.
    DirectSynopsisRetriever and LLMEnrichmentRetriever both embed the same raw
    query in the same asyncio.gather fan-out (MovieRecommender.recommend),
    otherwise doubling that round trip on every turn."""

    def __init__(self, inner: Embeddings) -> None:
        self._inner = inner
        self._inflight: dict[str, asyncio.Task[list[float]]] = {}

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._inner.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._inner.embed_query(text)

    async def aembed_query(self, text: str) -> list[float]:
        return await self._inner.aembed_query(text)

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        if len(texts) != 1:
            return await _aembed_documents_with_retry(self._inner, texts)
        text = texts[0]
        task = self._inflight.get(text)
        if task is None:
            task = asyncio.ensure_future(self._embed_one(text))
            self._inflight[text] = task
        try:
            return [await asyncio.shield(task)]
        finally:
            if task.done():
                self._inflight.pop(text, None)

    async def _embed_one(self, text: str) -> list[float]:
        return (await _aembed_documents_with_retry(self._inner, [text]))[0]


def build_recommender_service(
    spoiler_free: bool = False,
    include_knowledge_retriever: bool = False,
) -> tuple[ConversationalRecommendationService, QdrantMediaItems, ConversationTitler]:
    """Composition root shared by the CLI (`app/rag.py`) and NiceGUI
    (`nicegui_app/service_cache.py`) entry points — connects to Qdrant, wires
    up the retriever stack, and returns the chat service, a MediaItem lookup
    for rendering, and a titler for the web UI's Recent-conversations sidebar
    (reusing the same `llm` instance rather than building a second Gemini
    client). `include_knowledge_retriever` adds `LLMKnowledgeRetriever`, which
    scans the full title list per turn — worth it for the CLI's
    non-latency-sensitive usage, skipped by default for the web UI. When
    FAKE_GEMINI=true (app/config.py), both clients below are swapped for
    deterministic in-process fakes (app/adapters/fake_gemini.py) — no Gemini
    calls, no GOOGLE_API_KEY needed; Qdrant is still hit for real."""
    llm: BaseChatModel
    embeddings: Embeddings
    if FAKE_GEMINI:
        logger.warning(
            "FAKE_GEMINI=true — using deterministic fakes instead of Gemini. No "
            "LLM or embedding calls will be made; recommendations will not be "
            "real. Qdrant is still used for real."
        )
        embeddings = _DedupingEmbeddings(DeterministicEmbeddings())
        llm = FakeChatModel()
    else:
        embeddings = _DedupingEmbeddings(
            GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
        )
        llm = ChatGoogleGenerativeAI(
            model="gemini-3.1-flash-lite", temperature=0, safety_settings=_SAFETY_OFF
        )

    vector_store = connect_vector_store(QDRANT_URL, QDRANT_COLLECTION, embeddings)
    documents = load_synopsis_documents(vector_store, QDRANT_COLLECTION)
    media_repo = QdrantMediaItems(documents)

    retrievers: list[CandidateRetriever] = [
        DirectSynopsisRetriever(vector_store, embeddings),
        HyDEVectorRetriever(vector_store, embeddings, llm),
        LLMEnrichmentRetriever(vector_store, embeddings),
    ]
    if include_knowledge_retriever:
        doc_by_title = {doc.metadata["title"].lower(): doc for doc in documents}
        movie_list_str = "\n".join(doc.metadata["title"] for doc in documents)
        retrievers.append(LLMKnowledgeRetriever(llm, movie_list_str, doc_by_title))

    recommender = MovieRecommender(
        retrievers=retrievers,
        generator=GeminiRecommendationGenerator(llm, spoiler_free=spoiler_free),
        rewriter=GeminiQueryRewriter(llm),
    )
    return (
        ConversationalRecommendationService(recommender),
        media_repo,
        GeminiConversationTitler(llm),
    )


def build_diversity_service() -> DiversityRecommendationService | None:
    """Composition root for the diversity/"surprise me" feature — deliberately
    separate from `build_recommender_service`, not folded into its return tuple:
    this feature is optional (depends on plex-ingest's watch_history pipeline
    having actually run — see docs/pipeline-design.md) in a way the main chat
    feature isn't, and `QdrantUnavailableError` here must disable the feature, not
    take down the whole app the way it would if raised during the main
    `connect_vector_store` call above. Callers (CLI, NiceGUI) treat `None` as
    "feature unavailable" and say so, rather than crashing. Also honors
    FAKE_GEMINI — see `build_recommender_service`."""
    embeddings: Embeddings = (
        DeterministicEmbeddings()
        if FAKE_GEMINI
        else GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
    )

    try:
        watch_history_store = connect_vector_store(
            QDRANT_URL, QDRANT_WATCH_HISTORY_COLLECTION, embeddings
        )
    except QdrantUnavailableError:
        logger.warning(
            "watch_history collection (%s) unavailable — diversity mode disabled. "
            "Run plex-ingest's watch_history pipeline to enable it.",
            QDRANT_WATCH_HISTORY_COLLECTION,
        )
        return None

    watch_history_points = load_watch_history_points(
        watch_history_store, QDRANT_WATCH_HISTORY_COLLECTION
    )

    media_store = connect_vector_store(QDRANT_URL, QDRANT_COLLECTION, embeddings)
    candidates = load_synopsis_vectors(media_store, QDRANT_COLLECTION)
    media_repo = QdrantMediaItems(
        load_synopsis_documents(media_store, QDRANT_COLLECTION)
    )

    recommender = DiversityRecommender(
        QdrantWatchHistory(watch_history_points), QdrantCandidatePool(candidates)
    )
    return DiversityRecommendationService(recommender, media_repo)
