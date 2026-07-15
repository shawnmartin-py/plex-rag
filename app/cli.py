import asyncio

import typer

app = typer.Typer(
    name="plex-rag",
    help="Get AI-powered movie recommendations from your Plex library.",
    no_args_is_help=True,
)


@app.command()
def chat(
    no_spoilers: bool = typer.Option(
        False,
        "--no-spoilers",
        help="Omit plot details and story spoilers from recommendations.",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Show which retrievers flagged each candidate movie.",
    ),
) -> None:
    """Start an interactive AI movie recommendation session."""
    from app.rag import main

    asyncio.run(main(spoiler_free=no_spoilers, verbose=verbose))


@app.command()
def surprise() -> None:
    """Recommend something different from your recent Plex watch history, instead
    of matching a query — the opposite of `chat`'s similarity search."""
    from app.surprise import main as surprise_main

    surprise_main()


@app.command()
def check_imdb(
    imdb_id: str = typer.Argument(..., help="IMDb ID to look up, e.g. tt0111161."),
) -> None:
    """Check whether an IMDb ID is present in the Qdrant media_items collection.

    Prints "true" or "false" and exits 0 if found, 1 if not — so it can be used
    in shell conditionals as well as scripted for its output.
    """
    from app.config import QDRANT_COLLECTION, QDRANT_URL
    from app.repositories.vector_store import QdrantUnavailableError, imdb_id_exists

    try:
        exists = imdb_id_exists(QDRANT_URL, QDRANT_COLLECTION, imdb_id)
    except QdrantUnavailableError as e:
        typer.echo(str(e), err=True)
        raise typer.Exit(2) from None

    typer.echo("true" if exists else "false")
    raise typer.Exit(0 if exists else 1)


@app.command()
def check_tmdb(
    tmdb_id: str = typer.Argument(..., help="TMDB ID to look up, e.g. 603."),
) -> None:
    """Check whether a TMDB ID is present in the Qdrant media_items collection.

    TMDB ids are the collection's primary key (see docs/vector-store-contract.md);
    `check-imdb` still works against the retained imdb_id metadata attribute.
    Prints "true" or "false" and exits 0 if found, 1 if not, 2 if Qdrant is
    unreachable — same convention as `check-imdb`.
    """
    from app.config import QDRANT_COLLECTION, QDRANT_URL
    from app.repositories.vector_store import QdrantUnavailableError, tmdb_id_exists

    try:
        exists = tmdb_id_exists(QDRANT_URL, QDRANT_COLLECTION, tmdb_id)
    except QdrantUnavailableError as e:
        typer.echo(str(e), err=True)
        raise typer.Exit(2) from None

    typer.echo("true" if exists else "false")
    raise typer.Exit(0 if exists else 1)


@app.command()
def clear_history() -> None:
    """Wipe the web UI's recent-conversations history."""
    from app.config import CONVERSATIONS_DB_PATH
    from app.repositories.conversation_store import ConversationStore

    store = ConversationStore(CONVERSATIONS_DB_PATH)
    count = store.clear()
    typer.echo(f"Cleared {count} conversation(s) from history.")
