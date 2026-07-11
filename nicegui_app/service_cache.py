import asyncio

from nicegui import run

from app.bootstrap import build_recommender_service
from app.repositories.qdrant_media_items import QdrantMediaItems
from app.services.recommendation import ConversationalRecommendationService

# Module-level, process-lifetime cache keyed by spoiler_free — the direct
# equivalent of streamlit_app/init.py's `@st.cache_resource`-keyed-by-argument
# behavior. Every browser tab sharing a spoiler_free value gets the SAME
# ConversationalRecommendationService instance, including its chat history —
# this is intentional, replicating the Streamlit app's existing behavior
# (see docs/recommender.md), not a bug to fix here.
_cache: dict[bool, tuple[ConversationalRecommendationService, QdrantMediaItems]] = {}
_lock = asyncio.Lock()


async def get_service(
    spoiler_free: bool,
) -> tuple[ConversationalRecommendationService, QdrantMediaItems]:
    if spoiler_free in _cache:
        return _cache[spoiler_free]
    async with _lock:
        if spoiler_free not in _cache:
            result = await run.io_bound(
                build_recommender_service, spoiler_free=spoiler_free
            )
            if result is None:
                raise RuntimeError("build_recommender_service was cancelled")
            _cache[spoiler_free] = result
        return _cache[spoiler_free]
