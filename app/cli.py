import typer

app = typer.Typer(
    name="plex-rag",
    help="Get AI-powered movie recommendations from your Plex library.",
    no_args_is_help=True,
)


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
