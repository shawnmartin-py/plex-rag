import streamlit as st

from app.bootstrap import build_recommender_service
from app.repositories.qdrant_media_items import QdrantMediaItems
from app.services.recommendation import ConversationalRecommendationService


@st.cache_resource
def build_service(spoiler_free: bool = False) -> tuple[ConversationalRecommendationService, QdrantMediaItems]:
    return build_recommender_service(spoiler_free=spoiler_free)
