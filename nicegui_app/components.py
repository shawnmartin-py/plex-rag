import html
from collections.abc import Mapping, Sequence

from nicegui import ui
from nicegui.element import Element

from app.formatting.media_badge import PlatformBadge, describe_media_badge
from app.formatting.runtime import format_runtime
from app.formatting.sections import (
    break_out_run_in_labels,
    parse_sections,
    strip_section_heading,
)
from app.models.media_item import HdrFormat, MediaItem, StreamingSource

# Streaming-platform logos live under nicegui_app/static/logos (see main.py's
# add_static_files call). A platform without a mapped logo here still gets a plain
# text badge — see the fallback in render_movie_card.
_PLATFORM_LOGOS: dict[StreamingSource, str] = {
    StreamingSource.NETFLIX: "/static/logos/netflix.png",
    StreamingSource.DISNEY_PLUS: "/static/logos/disney_plus.png",
}

# HDR-format marks, keyed like the platform logos above but with no text
# fallback: unlike the platform badge (which replaces the resolution tag
# entirely), these are garnish next to it — an unmapped format just doesn't
# show. (url, alt text) per format.
_HDR_FORMAT_ICONS: dict[HdrFormat, tuple[str, str]] = {
    HdrFormat.HDR: ("/static/icons/hdr.png", "HDR"),
    HdrFormat.DV: ("/static/icons/dv.png", "Dolby Vision"),
}


def _key_light_gradient(colors: Sequence[str]) -> str:
    """The key-light line as a CSS gradient: the poster's middle-band colors
    left to right, fading out at both ends. Adjacent stops interpolate
    linearly, standing in for the blur of the old sampled-poster strip."""
    if len(colors) == 1:
        stops = f"{colors[0]} 30%, {colors[0]} 70%"
    else:
        start, end = 25.0, 75.0
        step = (end - start) / (len(colors) - 1)
        stops = ", ".join(
            f"{color} {start + i * step:g}%" for i, color in enumerate(colors)
        )
    return f"linear-gradient(90deg, transparent, {stops}, transparent)"


def render_movie_card(
    item: MediaItem,
    body_md: str,
    top_pick: bool,
    accent: Sequence[str] | None = None,
) -> None:
    """Render one recommendation as a poster-backdrop card.

    The header (title, year, rating, genres) comes from the item's structured
    metadata rather than LLM text — the LLM's own heading line is stripped by
    the caller. The card's backdrop is a blurred copy of the movie's real
    poster behind a left-to-right dark scrim, so each card picks up its own
    artwork's palette while the text column stays high-contrast.

    `accent` is the poster's server-extracted key-light palette (middle-band
    colors, left to right — see app/adapters/poster_accent.py), rendered as
    a plain gradient line rather than a sampled-poster CSS construct because
    Safari painted the latter unreliably. None means no key light.

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
        if accent:
            ui.element("div").classes("plex-card-key").style(
                f"background: {_key_light_gradient(accent)}"
            )
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
                        # A raw anchor rather than ui.link so the star glyph
                        # can be a styled span (tungsten) while the rating
                        # number stays ink-colored.
                        href = html.escape(
                            f"https://www.imdb.com/title/{item.imdb_id}/"
                        )
                        ui.html(
                            f'<a class="plex-badge plex-badge-link" '
                            f'href="{href}" target="_blank" rel="noopener">'
                            f'<span class="plex-star">★</span> '
                            f"{item.imdb_rating}</a>"
                        )
                    runtime = format_runtime(item.runtime_minutes)
                    if runtime:
                        ui.label(runtime).classes("plex-runtime").mark("plex-runtime")
                    if item.content_rating:
                        ui.label(item.content_rating).classes("plex-cert").mark(
                            "plex-cert"
                        )
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
                    for fmt in item.hdr_formats:
                        if fmt not in _HDR_FORMAT_ICONS:
                            continue
                        icon_url, icon_alt = _HDR_FORMAT_ICONS[fmt]
                        # Per-format height class: the HDR plaque and the wide
                        # Dolby Vision wordmark have very different aspect
                        # ratios, so one shared height would misbalance them.
                        ui.element("img").classes(
                            f"plex-format-badge plex-format-{fmt.name.lower()}"
                        ).props(f'src="{icon_url}" alt="{icon_alt}"').mark(
                            f"hdr-badge-{fmt.name.lower()}"
                        )
                    if item.genres:
                        ui.label(" · ".join(item.genres[:3])).classes("plex-genres")
                if body_md:
                    ui.markdown(break_out_run_in_labels(body_md)).classes(
                        "plex-card-body"
                    )


def render_recommendations(
    container: Element,
    response: str,
    items: list[MediaItem],
    accents: Mapping[str, Sequence[str] | None] | None = None,
) -> None:
    """Render each numbered recommendation as a poster-backdrop card.

    Items are paired to numbered sections positionally — no title
    text-matching. The first card gets a "Top pick" eyebrow, since the
    generator is instructed to rank best match first. `accents` maps
    tmdb_id -> key-light color for cards whose poster accent is known.
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
                    item,
                    strip_section_heading(text),
                    top_pick=item_idx == 0,
                    accent=(accents or {}).get(item.tmdb_id),
                )
                item_idx += 1
            else:
                ui.markdown(text).classes("plex-msg-prose")


def render_surprise_results(
    container: Element,
    response: str,
    items: list[MediaItem],
    accents: Mapping[str, Sequence[str] | None] | None = None,
) -> None:
    """Render a "Surprise me" turn: intro text followed by one card per
    diversity pick.

    Unlike `render_recommendations`, items here are never paired to numbered
    sections in the text — the diversity recommender's answer is always plain
    prose, so every item gets its own card regardless, sourced from the
    item's own description rather than LLM commentary. Used for both the live
    turn and Recent replay so the two can't drift apart.
    """
    with container:
        ui.markdown(response).classes("plex-msg-prose")
        for idx, item in enumerate(items):
            render_movie_card(
                item,
                item.description or "",
                top_pick=idx == 0,
                accent=(accents or {}).get(item.tmdb_id),
            )


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
