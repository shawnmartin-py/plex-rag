import asyncio

from app.bootstrap import build_diversity_service, build_recommender_service
from app.domain.ports import ConversationTitler
from app.repositories.qdrant_media_items import QdrantMediaItems
from app.services.diversity_recommendation import DiversityRecommendationService
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


# Diversity mode has no per-session state worth keying on: unlike chat, a
# "surprise" pull has no conversation history, and build_diversity_service()
# takes no session-scoping params (no spoiler_free either — it's just movie
# picks, no LLM commentary to spoil). One process-wide instance, built lazily
# on first use. Its _recently_shown exclusion set is deliberately shared
# across every caller rather than per-session — this is a single-user LAN
# app (see plex-tvos/CLAUDE.md), so "don't repeat a pick" should hold across
# taps regardless of which session_id asked.
_diversity_service: DiversityRecommendationService | None = None
_diversity_built = False
_diversity_lock = asyncio.Lock()


async def get_diversity_service() -> DiversityRecommendationService | None:
    """None means the feature is unavailable (no watch_history collection
    indexed yet) — see build_diversity_service's own docstring."""
    global _diversity_service, _diversity_built
    if _diversity_built:
        return _diversity_service
    async with _diversity_lock:
        if not _diversity_built:
            _diversity_service = await asyncio.to_thread(build_diversity_service)
            _diversity_built = True
        return _diversity_service
