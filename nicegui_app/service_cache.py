import asyncio

from nicegui import run

from app.bootstrap import build_diversity_service, build_recommender_service
from app.domain.ports import ConversationTitler
from app.repositories.qdrant_media_items import QdrantMediaItems
from app.services.diversity_recommendation import DiversityRecommendationService
from app.services.recommendation import ConversationalRecommendationService

# Module-level, process-lifetime cache keyed by spoiler_free — the direct
# equivalent of streamlit_app/init.py's `@st.cache_resource`-keyed-by-argument
# behavior. Every browser tab sharing a spoiler_free value gets the SAME
# ConversationalRecommendationService instance, including its chat history —
# this is intentional, replicating the Streamlit app's existing behavior
# (see docs/recommender.md), not a bug to fix here. The titler rides along in
# the same cached tuple since it's stateless per-call (no history of its own).
_cache: dict[
    bool,
    tuple[ConversationalRecommendationService, QdrantMediaItems, ConversationTitler],
] = {}
_lock = asyncio.Lock()


async def get_service(
    spoiler_free: bool,
) -> tuple[ConversationalRecommendationService, QdrantMediaItems, ConversationTitler]:
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


# Separate cache from `_cache` above: the diversity service isn't keyed by
# spoiler_free (it has no spoiler concept), and its build can legitimately return
# None (watch_history collection not populated yet). That's exactly the value
# `get_service` above treats as "io_bound was cancelled" — ambiguous here, since a
# legitimate None and a cancellation would look identical. `_build_boxed` sidesteps
# it by always returning a 1-tuple, so only a genuine cancellation yields a bare
# None from `run.io_bound` itself.
_diversity_service: DiversityRecommendationService | None = None
_diversity_loaded = False
_diversity_lock = asyncio.Lock()


def _build_boxed() -> tuple[DiversityRecommendationService | None]:
    return (build_diversity_service(),)


async def get_diversity_service() -> DiversityRecommendationService | None:
    global _diversity_service, _diversity_loaded
    if _diversity_loaded:
        return _diversity_service
    async with _diversity_lock:
        if not _diversity_loaded:
            boxed = await run.io_bound(_build_boxed)
            if boxed is None:
                raise RuntimeError("build_diversity_service was cancelled")
            (_diversity_service,) = boxed
            _diversity_loaded = True
    return _diversity_service
