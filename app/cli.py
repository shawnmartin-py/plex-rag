import typer

app = typer.Typer(
    name="plex-rag",
    help="Manage your Plex library and get AI-powered movie recommendations.",
    no_args_is_help=True,
)


@app.command()
def sync() -> None:
    """Sync your Plex library to the local DB and fetch any missing synopses."""
    from app.main import sync_library

    sync_library()


@app.command()
def scrape() -> None:
    """Fetch synopses for any existing library items that are still missing one."""
    from app.scrape_imdb import scrape_missing_synopses

    scrape_missing_synopses()


@app.command()
def enrich() -> None:
    """Generate and index LLM expert profiles for all library movies (run after sync)."""
    from langchain_google_genai import (
        ChatGoogleGenerativeAI,
        GoogleGenerativeAIEmbeddings,
        HarmBlockThreshold,
        HarmCategory,
    )

    from app.config import QDRANT_COLLECTION as COLLECTION_NAME
    from app.config import QDRANT_PATH
    from app.repositories.sql import SqlMediaItems
    from app.services.enrichment import EnrichmentService
    from app.services.vector_store import VectorStoreService

    _safety_off = {
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
    }

    embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
    llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite", temperature=0, safety_settings=_safety_off, timeout=60)

    sql_repo = SqlMediaItems()
    all_items = sql_repo.load()

    vs_service = VectorStoreService(path=QDRANT_PATH, collection_name=COLLECTION_NAME, embeddings=embeddings)
    documents = [item.to_document() for item in all_items if item.synopsis]
    vs_service.load_or_build(documents)

    EnrichmentService(llm, vs_service, COLLECTION_NAME).build(all_items)


@app.command(name="clear-enrichments")
def clear_enrichments() -> None:
    """Delete all enriched embeddings from the vector store (preserves synopsis embeddings)."""
    from qdrant_client import QdrantClient
    from qdrant_client.models import FieldCondition, Filter, MatchValue

    from app.config import QDRANT_COLLECTION as COLLECTION_NAME
    from app.config import QDRANT_PATH

    client = QdrantClient(path=QDRANT_PATH)
    client.delete(
        collection_name=COLLECTION_NAME,
        points_selector=Filter(
            must=[FieldCondition(key="metadata.embedding_type", match=MatchValue(value="enriched"))]
        ),
    )
    print("All enriched embeddings removed. Run 'enrich' to regenerate.")


@app.command()
def chat(
    no_spoilers: bool = typer.Option(
        False, "--no-spoilers", help="Omit plot details and story spoilers from recommendations."
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show which retrievers flagged each candidate movie."),
) -> None:
    """Start an interactive AI movie recommendation session."""
    from app.rag import main

    main(spoiler_free=no_spoilers, verbose=verbose)
