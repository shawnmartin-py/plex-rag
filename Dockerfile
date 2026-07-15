FROM python:3.14-slim

COPY --from=ghcr.io/astral-sh/uv:0.9 /uv /uvx /usr/local/bin/

WORKDIR /app

# Install deps in their own layer first so source edits don't bust the cache.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project

COPY app ./app
COPY api_app ./api_app
COPY README.md ./
RUN uv sync --frozen

ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 8100

# API only — the NiceGUI front end (nicegui_app/) isn't part of this image.
# See docker-compose.yml / README's "Running the API in Docker" section.
CMD ["plex-rag-api"]
