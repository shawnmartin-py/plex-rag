import re

# Patterns that signal a trailing summary/notes block within a numbered section
_NOTES_RE = re.compile(
    r"\n+(?=(?:Recommendation Summary|A Note on|Note:|In Summary|Final Note|Summary:"
    r"|To summarize|Honorable Mention|Additional|Other Option|Other Candidate"
    r"|In Closing|Overall|\*\*(?:A Note|Note|Summary|Recommendation|Honorable"
    r"|Additional|Other)))",
    re.IGNORECASE,
)

_NUMBERED_RE = re.compile(r"^(?:#{1,4} *|\*{1,2})?(?:\d+)\b[.)]")


def split_trailing_notes(text: str) -> tuple[str, str | None]:
    """Peel off a trailing summary/notes block from a movie section, if present."""
    m = _NOTES_RE.search(text)
    if m:
        return text[: m.start()].strip(), text[m.start() :].strip()
    return text, None


def strip_section_heading(text: str) -> str:
    """Drop the leading numbered-heading line (e.g. ``1. **Title** (2019)``).

    The web UI builds each card's header (title, year, rating) from the
    matched item's structured metadata, so the LLM's own heading line would
    duplicate it. Returns the section unchanged when its first line is not a
    numbered heading; a heading-only section becomes an empty string.
    """
    first_line, _, rest = text.partition("\n")
    if _NUMBERED_RE.match(first_line):
        return rest.strip()
    return text


# A bold run-in label glued onto the preceding sentence, e.g.
# "...heavy entry. **Tone & Pacing:** Slow and mournful". Both colon
# placements occur in real output: **Label:** and **Label**:.
# The lookbehind excludes list markers (- * +) so a label that already
# starts its own bullet line is left alone.
_RUN_IN_LABEL_RE = re.compile(r"(?<=[^\s*+-])[ \t]+(?=\*\*[^*\n]{1,60}(?::\*\*|\*\*:))")


def break_out_run_in_labels(text: str) -> str:
    """Give every bold run-in label (``**Tone & Pacing:** ...``) its own
    paragraph.

    The generator usually starts a new paragraph per labeled block, but
    sometimes glues a second label onto the end of the previous sentence.
    Mid-paragraph, the card stylesheet can't lift it into a block label —
    it only targets ``strong:first-child`` — so the label renders inline.
    A deterministic markdown fix beats prompting the model about layout.
    """
    return _RUN_IN_LABEL_RE.sub("\n\n", text)


def parse_sections(response: str) -> list[tuple[bool, str]]:
    """Split LLM response into (is_numbered_section, text) pairs."""
    parts = re.split(
        r"(?=\n(?:#{1,4} *|\*{1,2})?(?:\d+)\b[.)])", "\n" + response.strip()
    )
    results: list[tuple[bool, str]] = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        is_numbered = bool(_NUMBERED_RE.match(part))
        movie_text, notes = split_trailing_notes(part)
        results.append((is_numbered, movie_text))
        if notes:
            results.append((False, notes))
    return results
