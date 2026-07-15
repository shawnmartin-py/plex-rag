import asyncio
import sys
from collections.abc import AsyncIterator

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse

from api_app.schemas import (
    ChatRequest,
    ChatResponse,
    ChatStreamCardOut,
    ChatStreamTextOut,
    HealthResponse,
    MediaItemOut,
    ResetRequest,
    ResetResponse,
)
from api_app.service_cache import get_diversity_service, get_service, reset_session
from app.config import QDRANT_COLLECTION, QDRANT_URL
from app.domain.diversity import NoWatchHistoryError
from app.domain.ports import TextDelta
from app.repositories.vector_store import (
    QdrantUnavailableError,
    ensure_qdrant_reachable,
)
from app.services.recommendation import CardReady

# Distinct from NiceGUI's default 8080 so `plex-rag-web` and `plex-rag-api`
# can run side by side against the same Qdrant collection.
API_PORT = 8100

app = FastAPI(title="plex-rag API")


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(status="ok")


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    message = request.message.strip()
    if not message:
        raise HTTPException(status_code=422, detail="message must not be empty")
    service, media_repo, _titler = await get_service(
        request.session_id, request.spoiler_free
    )
    answer, items = await service.chat_with_items(message, media_repo)
    out_items = [MediaItemOut.from_domain(i) for i in items]
    return ChatResponse(answer=answer, items=out_items)


# Streaming counterpart to /chat, for the tvOS client's progressive-card UX
# (see plex-tvos/CLAUDE.md). Newline-delimited JSON, one ChatStreamTextOut or
# ChatStreamCardOut per line, in generation order — the same
# chat_with_items_stream() the NiceGUI app already uses for its own live
# transcript, just serialized over the wire instead of consumed in-process.
# A CardReady with an unresolved item (tmdb_id didn't resolve to a
# MediaItem) is skipped rather than sent, same as the NiceGUI transcript
# skips rendering it.
@app.post("/chat/stream")
async def chat_stream(request: ChatRequest) -> StreamingResponse:
    message = request.message.strip()
    if not message:
        raise HTTPException(status_code=422, detail="message must not be empty")
    service, media_repo, _titler = await get_service(
        request.session_id, request.spoiler_free
    )
    streamed = await service.chat_with_items_stream(message, media_repo)

    async def ndjson() -> AsyncIterator[bytes]:
        async for event in streamed.events:
            out: ChatStreamTextOut | ChatStreamCardOut
            if isinstance(event, TextDelta):
                if not event.text:
                    continue
                out = ChatStreamTextOut(text=event.text)
            elif isinstance(event, CardReady):
                if event.item is None:
                    continue
                out = ChatStreamCardOut(item=MediaItemOut.from_domain(event.item))
            else:
                continue
            yield out.model_dump_json().encode() + b"\n"

    return StreamingResponse(ndjson(), media_type="application/x-ndjson")


@app.post("/chat/reset", response_model=ResetResponse)
async def reset(request: ResetRequest) -> ResetResponse:
    return ResetResponse(reset=reset_session(request.session_id))


# "Wildcards" in the tvOS client — the diversity/"palette cleanser" mode:
# movies picked for being furthest from recent watch history, not for
# matching a query. No request body (nothing to key off — see
# service_cache.get_diversity_service's docstring on why this is one
# process-wide instance rather than per-session), and answer is always ""
# since, like the NiceGUI "Surprise me" button, picks here carry no
# generated commentary — just the cards themselves.
@app.post("/surprise", response_model=ChatResponse)
async def surprise() -> ChatResponse:
    service = await get_diversity_service()
    if service is None:
        raise HTTPException(
            status_code=404,
            detail="Wildcards isn't available yet — no watch history has been indexed.",
        )
    try:
        items = await asyncio.to_thread(service.recommend)
    except NoWatchHistoryError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    return ChatResponse(answer="", items=[MediaItemOut.from_domain(i) for i in items])


def main() -> None:
    """Console-script entry point (`plex-rag-api`).

    Fails fast, before the server ever starts accepting connections, rather
    than letting a client's first request be where a dead Qdrant container is
    discovered — same reasoning as `nicegui_app/main.py:main`.
    """
    try:
        ensure_qdrant_reachable(QDRANT_URL, QDRANT_COLLECTION)
    except QdrantUnavailableError as e:
        print(f"\n{e}\n", file=sys.stderr)
        raise SystemExit(1) from None

    # host="0.0.0.0": the tvOS client is a separate device on the same LAN,
    # not localhost.
    uvicorn.run(app, host="0.0.0.0", port=API_PORT)  # noqa: S104


if __name__ == "__main__":
    main()
