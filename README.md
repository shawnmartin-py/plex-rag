# plex-rag

A conversational movie recommendation chatbot that only recommends movies you actually have in your personal Plex library.

## What it does

### Library sync (`sync` command)

Connects to your Plex server, pulls all unwatched movies, and saves them to a local SQLite database. For any new movie without a synopsis, it scrapes one — trying IMDB's full plot summary first, falling back to Wikipedia's Plot section, then IMDB's short description. It uses a headless Chromium browser (Playwright) with anti-bot measures to do this scraping. It also prunes movies that have been removed from Plex.

### Vector store build

Each movie (title, year, IMDb rating, genres, synopsis) gets embedded using Google's Gemini embedding model (`gemini-embedding-001`, 3072 dimensions) and stored in a local Qdrant vector database. This enables semantic search — "find me something dark and psychological" can match movies by meaning, not just keywords. These synopsis embeddings remain the foundation of the store; enrichment embeddings are layered on top.

### Library enrichment (`enrich` command)

Generates and indexes deep expert film profiles for every movie in your library. For each film, three focused LLM profiles are written and stored as separate embeddings alongside the synopsis:

- **Craft** — subgenre positioning, cinematic movement, director style and filmography, visual grammar (camera movement, aspect ratio, lighting), editing rhythm, score and cinematographer.
- **Meaning** — narrative structure, core themes and recurring motifs, tone and emotional register, acting style, and how the film ends emotionally without spoiling plot.
- **Context** — cultural and historical moment, critical reception and retrospective reassessment, audience fit, six or more comparable films with specific axes of similarity, and a dense retrieval-optimised tag paragraph.

Each profile is generated offline, so at query time the system searches pre-computed expertise rather than sending your entire library to the LLM. This scales to any catalogue size. The enrichment step is idempotent — re-running it skips sections that already exist and only fills gaps. If a film's synopsis triggers a content policy block, the enrichment retries without the synopsis so the profile is generated from the model's own knowledge instead.

### Conversational recommendations (`chat` command)

Ask questions like "what should I watch tonight?" or "something like Parasite but lighter" and get back ranked, reasoned recommendations. Under the hood:

- **Query rewriting** — follow-up questions ("what about something shorter?") are rewritten into standalone queries using conversation history, so context carries through multi-turn conversation.
- **Quad retrieval** — four strategies run in parallel and their results are grouped by film and deduplicated:
  - *Direct synopsis retriever*: your query is embedded directly and searched against synopsis embeddings — reliable for plot-specific and meta queries (language, cast, content rating) where thematic vocabulary is less useful.
  - *HyDE retriever*: the LLM generates a dense expert film profile matching your request (subgenre labels, director influences, tone descriptors, cinematic movements), then finds real movies whose enrichment embeddings are closest to that hypothetical profile — surfaces films that match the *critic vocabulary* of your request rather than its surface words.
  - *LLM knowledge retriever*: the LLM uses its film expertise to scan your full movie list and select candidates by director, subgenre, cultural context, tone, etc. — great for queries like "classic Kubrick-esque films." (Scales well up to a few hundred titles in the list.)
  - *Enrichment retriever*: your query is embedded directly and searched against the pre-computed expert profiles — craft, meaning, and context sections — bringing in retrieval signal that doesn't exist in any synopsis, such as cinematographer names, movement labels, thematic keywords, and tone descriptors.
- **Grouped context** — retrieved documents are assembled per film with candidates in randomised order (to avoid position bias), synopsis first and enrichment sections following within each film block. Each film gets a single block in the context window, so the generator sees the full picture for each candidate.
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

# Generate and index expert film profiles (run once after sync, then incrementally)
plex-rag enrich

# Remove all enriched embeddings (preserves synopsis embeddings; re-run enrich to rebuild)
plex-rag clear-enrichments

# Start an interactive recommendation session
plex-rag chat

# Start in spoiler-free mode
plex-rag chat --no-spoilers

# Show retriever source coverage after each response (for debugging bias)
plex-rag chat --verbose
```

### Recommended first-run order

```bash
plex-rag sync       # pull library + scrape synopses
plex-rag enrich     # build expert profiles (takes a few minutes; rate-limited to ~4s between films)
plex-rag chat       # start chatting
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
│   ├── retrievers.py           # DirectSynopsisRetriever, HyDEVectorRetriever, LLMKnowledgeRetriever, LLMEnrichmentRetriever
│   └── generators.py           # GeminiRecommendationGenerator, GeminiQueryRewriter
├── services/
│   ├── enrichment.py           # EnrichmentService: generates and indexes per-film expert profiles
│   ├── recommendation.py       # ConversationalRecommendationService (manages chat history)
│   └── vector_store.py         # Qdrant vector store builder with retry/batching
├── repositories/
│   ├── sql.py                  # SQLAlchemy-backed media item persistence
│   └── json.py                 # JSON-backed alternative repository
└── models/
    └── media_item.py           # MediaItem dataclass + Plex/Document conversion
```

### Embedding types in the vector store

The Qdrant collection stores two types of documents, distinguished by a `metadata.embedding_type` field:

| Type       | Content                                                    | Added by                      |
|------------|------------------------------------------------------------|-------------------------------|
| `synopsis` | Title, year, rating, genres, synopsis text                 | `sync` / `rag.py` on startup  |
| `enriched` | LLM-generated expert profile (one document per section)    | `enrich` command              |

Enriched documents also carry a `metadata.section` field (`craft`, `meaning`, or `context`) that the enrichment retriever and idempotency checks use to filter precisely.