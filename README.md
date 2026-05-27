# plex-rag

A conversational movie recommendation chatbot that only recommends movies you actually have in your personal Plex library.

## What it does

### Library sync (`sync` command)

Connects to your Plex server, pulls all unwatched movies, and saves them to a local SQLite database. For any new movie without a synopsis, it scrapes one — trying IMDB's full plot summary first, falling back to Wikipedia's Plot section, then IMDB's short description. It uses a headless Chromium browser (Playwright) with anti-bot measures to do this scraping. It also prunes movies that have been removed from Plex.

### Vector store build

Each movie (title, year, IMDb rating, genres, synopsis) gets embedded using Google's Gemini embedding model and stored in a local Qdrant vector database. This enables semantic search — "find me something dark and psychological" can match movies by meaning, not just keywords.

### Conversational recommendations (`chat` command)

Ask questions like "what should I watch tonight?" or "something like Parasite but lighter" and get back ranked, reasoned recommendations. Under the hood:

- **Query rewriting** — follow-up questions ("what about something shorter?") are rewritten into standalone queries using conversation history, so context carries through multi-turn conversation.
- **Dual retrieval** — two strategies run in parallel and their results are merged and deduplicated:
  - *HyDE retriever*: the LLM generates a fictional synopsis matching your request, then finds real movies whose embeddings are similar to that hypothetical synopsis — great for vibe-based queries.
  - *LLM knowledge retriever*: the LLM uses its film expertise to scan your full movie list and select candidates by director, subgenre, cultural context, tone, etc. — great for queries like "classic Kubrick-esque films."
- **Recommendation generation** — the merged candidates are passed to Gemini, which ranks them and explains specifically why each fits, referencing themes, pacing, and director style. It acknowledges weak matches rather than overselling.
- **Spoiler-free mode** (`--no-spoilers`) — same flow, but the generator reasons only from genre, tone, cast, and style — never plot details or story outcomes.

The strict constraint throughout is that it only recommends movies from your library — the generator prompt explicitly forbids suggestions outside the retrieved candidate set.

## Setup

### Prerequisites

- Python 3.14+
- A running Plex Media Server (configured via `~/.config/plexapi/config.ini` or environment variables)
- A Google Gemini API key set as `GOOGLE_API_KEY`

### Install

```bash
uv sync
playwright install chromium
```

## Usage

```bash
# Sync your Plex library to the local DB and fetch missing synopses
plex-rag sync

# Fetch synopses for any existing items still missing one
plex-rag scrape

# Start an interactive recommendation session
plex-rag chat

# Start in spoiler-free mode
plex-rag chat --no-spoilers
```

## Architecture

```
app/
├── cli.py                      # Typer CLI entrypoint
├── main.py                     # sync_library: Plex → SQLite + synopsis scraping
├── rag.py                      # chat entrypoint: wires up the full RAG pipeline
├── plex.py                     # PlexAPI wrapper
├── synopsis.py                 # IMDB / Wikipedia synopsis scraper
├── scrape_imdb.py              # Standalone scrape job for missing synopses
├── domain/
│   ├── recommender.py          # MovieRecommender: orchestrates retrieve → generate
│   └── ports.py                # Interfaces: CandidateRetriever, RecommendationGenerator, QueryRewriter
├── adapters/
│   ├── retrievers.py           # HyDEVectorRetriever, LLMKnowledgeRetriever
│   └── generators.py           # GeminiRecommendationGenerator, GeminiQueryRewriter
├── services/
│   ├── recommendation.py       # ConversationalRecommendationService (manages chat history)
│   └── vector_store.py         # Qdrant vector store builder with retry/batching
├── repositories/
│   ├── sql.py                  # SQLAlchemy-backed media item persistence
│   └── json.py                 # JSON-backed alternative repository
└── models/
    └── media_item.py           # MediaItem dataclass + Plex/Document conversion
```