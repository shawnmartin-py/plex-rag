from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings,
    HarmBlockThreshold,
    HarmCategory,
)

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
from app.config import QDRANT_COLLECTION, QDRANT_URL
from app.domain.ports import CandidateRetriever, ConversationTitler
from app.domain.recommender import MovieRecommender
from app.repositories.qdrant_media_items import QdrantMediaItems
from app.repositories.vector_store import connect_vector_store, load_synopsis_documents
from app.services.recommendation import ConversationalRecommendationService

_SAFETY_OFF = {
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
}


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
    non-latency-sensitive usage, skipped by default for the web UI."""
    embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
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
