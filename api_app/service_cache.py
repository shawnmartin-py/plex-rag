import asyncio

from app.bootstrap import build_recommender_service
from app.domain.ports import ConversationTitler
from app.repositories.qdrant_media_items import QdrantMediaItems
from app.services.recommendation import ConversationalRecommendationService

# Session-keyed cache: every API client (the tvOS app, for now) supplies its
# own session_id, so — unlike nicegui_app/service_cache.py's cache keyed only
# by spoiler_free and intentionally shared across browser tabs — sessions
# here never share a ConversationalRecommendationService or its chat history.
# Changing spoiler_free mid-session swaps to a different cached instance for
# that session, same semantics as toggling the web UI's switch: a fresh
# history, not a migrated one.
_cache: dict[
    tuple[str, bool],
    tuple[ConversationalRecommendationService, QdrantMediaItems, ConversationTitler],
] = {}
_lock = asyncio.Lock()


async def get_service(
    session_id: str, spoiler_free: bool
) -> tuple[ConversationalRecommendationService, QdrantMediaItems, ConversationTitler]:
    key = (session_id, spoiler_free)
    if key in _cache:
        return _cache[key]
    async with _lock:
        if key not in _cache:
            _cache[key] = await asyncio.to_thread(
                build_recommender_service, spoiler_free=spoiler_free
            )
        return _cache[key]


def reset_session(session_id: str) -> bool:
    """Clears chat history for every cached spoiler_free variant of this
    session_id. Returns whether any cached service was found to reset."""
    found = False
    for (sid, _spoiler_free), (service, _media_repo, _titler) in _cache.items():
        if sid == session_id:
            service.reset_history()
            found = True
    return found
