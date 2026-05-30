import re

import streamlit as st

from app.models.media_item import MediaItem

# Patterns that signal a trailing summary/notes block within a numbered section
_NOTES_RE = re.compile(
    r"\n+(?=(?:Recommendation Summary|A Note on|Note:|In Summary|Final Note|Summary:|To summarize"
    r"|Honorable Mention|Additional|Other Option|Other Candidate|In Closing|Overall"
    r"|\*\*(?:A Note|Note|Summary|Recommendation|Honorable|Additional|Other)))",
    re.IGNORECASE,
)

_NUMBERED_RE = re.compile(r"^(?:#{1,4} *|\*{1,2})?(?:\d+)\b[.)]")


def _split_trailing_notes(text: str) -> tuple[str, str | None]:
    """Peel off a trailing summary/notes block from a movie section, if present."""
    m = _NOTES_RE.search(text)
    if m:
        return text[: m.start()].strip(), text[m.start() :].strip()
    return text, None


def _parse_sections(response: str) -> list[tuple[bool, str]]:
    """Split LLM response into (is_numbered_section, text) pairs."""
    parts = re.split(r"(?=\n(?:#{1,4} *|\*{1,2})?(?:\d+)\b[.)])", "\n" + response.strip())
    results: list[tuple[bool, str]] = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        is_numbered = bool(_NUMBERED_RE.match(part))
        movie_text, notes = _split_trailing_notes(part)
        results.append((is_numbered, movie_text))
        if notes:
            results.append((False, notes))
    return results


def render_recommendations(response: str, items: list[MediaItem]) -> None:
    """Render each numbered recommendation as poster + text.

    Items are paired to numbered sections positionally — no title text-matching.
    """
    sections = _parse_sections(response)

    any_numbered = any(is_numbered for is_numbered, _ in sections)
    if not any_numbered or not items:
        st.markdown(response)
        return

    item_idx = 0
    for is_numbered, text in sections:
        if is_numbered and item_idx < len(items):
            item = items[item_idx]
            item_idx += 1
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
        else:
            st.markdown(text)
