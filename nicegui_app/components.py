from nicegui import ui
from nicegui.element import Element

from app.formatting.media_badge import PlatformBadge, describe_media_badge
from app.formatting.sections import parse_sections, strip_section_heading
from app.models.media_item import MediaItem, StreamingSource

# Streaming-platform logos live under nicegui_app/static/logos (see main.py's
# add_static_files call). A platform without a mapped logo here still gets a plain
# text badge — see the fallback in render_movie_card.
_PLATFORM_LOGOS: dict[StreamingSource, str] = {
    StreamingSource.NETFLIX: "/static/logos/netflix.png",
    StreamingSource.DISNEY_PLUS: "/static/logos/disney_plus.png",
}


def render_movie_card(item: MediaItem, body_md: str, top_pick: bool) -> None:
    """Render one recommendation as a poster-backdrop card.

    The header (title, year, rating, genres) comes from the item's structured
    metadata rather than LLM text — the LLM's own heading line is stripped by
    the caller. The card's backdrop is a blurred copy of the movie's real
    poster behind a left-to-right dark scrim, so each card picks up its own
    artwork's palette while the text column stays high-contrast.

    Must be called inside a `with <container>:` block, same as any other
    NiceGUI element construction — used both for the batch (replay-from-
    Recent) render below and for live per-section rendering as a streamed
    answer completes each recommendation.
    """
    url = item.thumb_url.replace('"', "%22") if item.thumb_url else None
    with ui.element("article").classes("plex-card w-full"):
        if url:
            ui.element("div").classes("plex-card-bg").style(
                f'background-image: url("{url}")'
            )
        ui.element("div").classes("plex-card-scrim")
        with ui.row().classes("plex-card-inner w-full"):
            with ui.column().classes("plex-poster-col"):
                if url:
                    # A plain <img> rather than ui.image: Quasar's QImg manages
                    # its own internal sizing, which fights the stretch-to-card-
                    # height CSS and clips the picture.
                    alt = f"{item.title} poster".replace('"', "'")
                    ui.element("img").classes("plex-poster").props(
                        f'src="{url}" alt="{alt}"'
                    )
            with ui.column().classes("plex-text-col"):
                if top_pick:
                    ui.label("Top pick").classes("plex-card-eyebrow")
                with ui.row().classes("plex-title-row items-baseline"):
                    ui.label(item.title).classes("plex-card-title")
                    ui.label(str(item.year)).classes("plex-card-year")
                with ui.row().classes("plex-card-meta items-center"):
                    if item.imdb_rating:
                        ui.link(
                            f"★ {item.imdb_rating}",
                            f"https://www.imdb.com/title/{item.imdb_id}/",
                            new_tab=True,
                        ).classes("plex-badge plex-badge-link")
                    badge = describe_media_badge(item)
                    if isinstance(badge, PlatformBadge):
                        logo_url = _PLATFORM_LOGOS.get(badge.platform)
                        if logo_url:
                            alt = f"{badge.platform.value} logo"
                            ui.element("img").classes("plex-platform-badge").props(
                                f'src="{logo_url}" alt="{alt}"'
                            )
                        else:
                            ui.label(badge.platform.value).classes("plex-badge")
                    elif badge is not None:
                        ui.label(badge.label).classes("plex-badge")
                    if item.genres:
                        ui.label(" · ".join(item.genres[:3])).classes("plex-genres")
                if body_md:
                    ui.markdown(body_md).classes("plex-card-body")


def render_recommendations(
    container: Element, response: str, items: list[MediaItem]
) -> None:
    """Render each numbered recommendation as a poster-backdrop card.

    Items are paired to numbered sections positionally — no title
    text-matching. The first card gets a "Top pick" eyebrow, since the
    generator is instructed to rank best match first.
    """
    sections = parse_sections(response)

    any_numbered = any(is_numbered for is_numbered, _ in sections)
    if not any_numbered or not items:
        with container:
            ui.markdown(response).classes("plex-msg-prose")
        return

    item_idx = 0
    with container:
        for is_numbered, text in sections:
            if is_numbered and item_idx < len(items):
                item = items[item_idx]
                render_movie_card(
                    item, strip_section_heading(text), top_pick=item_idx == 0
                )
                item_idx += 1
            else:
                ui.markdown(text).classes("plex-msg-prose")


def render_chat_row(container: Element, role: str, content: str) -> Element:
    """Render a chat row: a right-aligned bubble for the user, plain
    secondary-tier prose for the assistant. No avatars.

    Returns the row body so callers (e.g. a pending assistant turn) can
    append content into it once available.
    """
    with container:
        row = ui.row().classes("plex-msg-row w-full")
        if role == "user":
            row.classes("justify-end")
            with row:
                with ui.element("div").classes("plex-msg-user") as body:
                    if content:
                        ui.markdown(content)
        else:
            with row:
                with ui.column().classes("plex-msg-body w-full") as body:
                    if content:
                        ui.markdown(content).classes("plex-msg-prose")
    return body
