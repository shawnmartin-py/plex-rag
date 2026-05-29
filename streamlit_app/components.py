import re

import streamlit as st

from app.models.media_item import MediaItem
from app.repositories.sql import SqlMediaItems

# Patterns that signal a trailing summary/notes block within a numbered section
_NOTES_RE = re.compile(
    r"\n+(?=(?:Recommendation Summary|A Note on|Note:|In Summary|Final Note|Summary:|To summarize"
    r"|Honorable Mention|Additional|Other Option|Other Candidate|In Closing|Overall"
    r"|\*\*(?:A Note|Note|Summary|Recommendation|Honorable|Additional|Other)))",
    re.IGNORECASE,
)


def _split_trailing_notes(text: str) -> tuple[str, str | None]:
    """Peel off a trailing summary/notes block from a movie section, if present."""
    m = _NOTES_RE.search(text)
    if m:
        return text[: m.start()].strip(), text[m.start() :].strip()
    return text, None


def _parse_sections(response: str, sql_repo: SqlMediaItems) -> list[tuple[MediaItem | None, str]]:
    """Split LLM response into (item, text) pairs by numbered section.

    Matches each section to a MediaItem by looking for a known title in the
    first two lines only (where the movie title always appears), so that
    cross-references in later prose don't steal the match.
    """
    # Match numbered items whether bare ("1."), bold ("**1.**"), or a markdown header ("### 1.")
    parts = re.split(r"(?=\n(?:#{1,4} *|\*{1,2})?(?:\d+)\b[.)])", "\n" + response.strip())
    items = list(sql_repo._cached_items.values())
    used: set[str] = set()

    results: list[tuple[MediaItem | None, str]] = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        matched: MediaItem | None = None
        # Strip markdown markers so "### 1. Julieta" and "Juror #2" both normalise cleanly
        raw_header = "\n".join(part.split("\n")[:2])
        header = re.sub(r"[#*_`]", "", raw_header).lower()
        for item in items:
            normalised_title = re.sub(r"[#*_`]", "", item.title).lower()
            if item.imdb_id not in used and normalised_title in header:
                matched = item
                used.add(item.imdb_id)
                break

        if matched is not None:
            # Split out any trailing notes/summary that got bundled into this section
            movie_text, notes = _split_trailing_notes(part)
            results.append((matched, movie_text))
            if notes:
                results.append((None, notes))
        else:
            results.append((None, part))

    return results


def render_recommendations(response: str, sql_repo: SqlMediaItems) -> None:
    """Render each numbered recommendation as poster + text, in sequence."""
    sections = _parse_sections(response, sql_repo)

    any_matched = any(item is not None for item, _ in sections)
    if not any_matched:
        st.markdown(response)
        return

    for item, text in sections:
        if item is None:
            st.markdown(text)
        else:
            col_img, col_text = st.columns([1, 3])
            with col_img:
                if item.thumb_url:
                    st.image(item.thumb_url, use_container_width=True)
                else:
                    st.markdown("🎬")
                if item.imdb_rating:
                    st.markdown(
                        f"<p style='color:#8E8E93;font-size:13px;margin-top:6px;'>⭐ {item.imdb_rating} IMDb</p>",
                        unsafe_allow_html=True,
                    )
            with col_text:
                st.markdown(text)
            st.divider()
