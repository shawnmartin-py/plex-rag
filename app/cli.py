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
def clear_history() -> None:
    """Wipe the web UI's recent-conversations history."""
    from app.config import CONVERSATIONS_DB_PATH
    from app.repositories.conversation_store import ConversationStore

    store = ConversationStore(CONVERSATIONS_DB_PATH)
    count = store.clear()
    typer.echo(f"Cleared {count} conversation(s) from history.")
