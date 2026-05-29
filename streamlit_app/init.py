import streamlit as st
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings,
    HarmBlockThreshold,
    HarmCategory,
)

from app.adapters.generators import GeminiQueryRewriter, GeminiRecommendationGenerator
from app.adapters.retrievers import DirectSynopsisRetriever, HyDEVectorRetriever, LLMEnrichmentRetriever
from app.config import QDRANT_COLLECTION, QDRANT_PATH
from app.domain.recommender import MovieRecommender
from app.repositories.sql import SqlMediaItems
from app.services.recommendation import ConversationalRecommendationService
from app.services.vector_store import VectorStoreService

_SAFETY_OFF = {
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
}


@st.cache_resource
def build_service(spoiler_free: bool = False) -> tuple[ConversationalRecommendationService, SqlMediaItems]:
    embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
    llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite", temperature=0, safety_settings=_SAFETY_OFF)

    sql_repo = SqlMediaItems()
    all_items = sql_repo.load()

    vs_service = VectorStoreService(path=QDRANT_PATH, collection_name=QDRANT_COLLECTION, embeddings=embeddings)
    documents = [item.to_document() for item in all_items if item.synopsis]
    vector_store = vs_service.load_or_build(documents)

    recommender = MovieRecommender(
        retrievers=[
            DirectSynopsisRetriever(vector_store, embeddings),
            HyDEVectorRetriever(vector_store, embeddings, llm),
            LLMEnrichmentRetriever(vector_store, embeddings),
        ],
        generator=GeminiRecommendationGenerator(llm, spoiler_free=spoiler_free),
        rewriter=GeminiQueryRewriter(llm),
    )
    return ConversationalRecommendationService(recommender), sql_repo
