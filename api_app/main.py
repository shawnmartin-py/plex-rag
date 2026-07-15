import sys

import uvicorn
from fastapi import FastAPI, HTTPException

from api_app.schemas import (
    ChatRequest,
    ChatResponse,
    HealthResponse,
    MediaItemOut,
    ResetRequest,
    ResetResponse,
)
from api_app.service_cache import get_service, reset_session
from app.config import QDRANT_COLLECTION, QDRANT_URL
from app.repositories.vector_store import (
    QdrantUnavailableError,
    ensure_qdrant_reachable,
)

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


@app.post("/chat/reset", response_model=ResetResponse)
async def reset(request: ResetRequest) -> ResetResponse:
    return ResetResponse(reset=reset_session(request.session_id))


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
