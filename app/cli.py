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
def chat(
    no_spoilers: bool = typer.Option(
        False, "--no-spoilers", help="Omit plot details and story spoilers from recommendations."
    ),
) -> None:
    """Start an interactive AI movie recommendation session."""
    from app.rag import main

    main(spoiler_free=no_spoilers)
