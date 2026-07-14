def format_runtime(minutes: int | None) -> str | None:
    """ "1h 44m" style, matching how streaming services usually display runtime.
    `None` when unresolved — a streaming-placeholder movie whose OMDb lookup hasn't
    (yet, or ever) resolved (see docs/vector-store-contract.md's "runtime_minutes")."""
    if minutes is None:
        return None
    hours, mins = divmod(minutes, 60)
    return f"{hours}h {mins}m" if hours else f"{mins}m"
