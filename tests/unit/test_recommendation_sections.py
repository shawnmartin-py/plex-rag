from app.formatting.sections import (
    break_out_run_in_labels,
    parse_sections,
    split_trailing_notes,
    strip_section_heading,
)


def test_split_trailing_notes_peels_off_recommendation_summary() -> None:
    text = "1. Parasite (2019)\nGreat pick.\n\nRecommendation Summary: enjoy!"
    movie_text, notes = split_trailing_notes(text)
    assert movie_text == "1. Parasite (2019)\nGreat pick."
    assert notes == "Recommendation Summary: enjoy!"


def test_split_trailing_notes_returns_none_when_no_notes_block() -> None:
    text = "1. Parasite (2019)\nGreat pick."
    movie_text, notes = split_trailing_notes(text)
    assert movie_text == text
    assert notes is None


def test_split_trailing_notes_matches_case_insensitively() -> None:
    text = "1. Parasite (2019)\nGreat pick.\n\nin summary, watch it."
    movie_text, notes = split_trailing_notes(text)
    assert movie_text == "1. Parasite (2019)\nGreat pick."
    assert notes == "in summary, watch it."


def test_parse_sections_splits_numbered_recommendations() -> None:
    response = (
        "Here are my picks:\n\n"
        "1. **Parasite** (2019)\nA masterpiece.\n\n"
        "2. **Thelma** (2017)\nAn icy thriller."
    )
    sections = parse_sections(response)
    numbered = [text for is_numbered, text in sections if is_numbered]
    assert len(numbered) == 2
    assert numbered[0].startswith("1. **Parasite**")
    assert numbered[1].startswith("2. **Thelma**")


def test_parse_sections_marks_intro_text_as_not_numbered() -> None:
    response = "Here are my picks:\n\n1. **Parasite** (2019)\nA masterpiece."
    sections = parse_sections(response)
    assert sections[0] == (False, "Here are my picks:")
    assert sections[1][0] is True


def test_parse_sections_separates_trailing_notes_from_the_prior_section() -> None:
    response = (
        "1. **Parasite** (2019)\nA masterpiece.\n\n"
        "Recommendation Summary: great choice overall."
    )
    sections = parse_sections(response)
    assert sections[-1] == (False, "Recommendation Summary: great choice overall.")
    assert sections[-2] == (True, "1. **Parasite** (2019)\nA masterpiece.")


def test_parse_sections_handles_response_with_no_numbered_items() -> None:
    response = "I don't have a great match for that request."
    sections = parse_sections(response)
    assert sections == [(False, response)]


def test_parse_sections_ignores_blank_input() -> None:
    assert parse_sections("") == []


def test_strip_section_heading_drops_numbered_heading_line() -> None:
    text = "1. **Parasite** (2019)\nA masterpiece of class tension."
    assert strip_section_heading(text) == "A masterpiece of class tension."


def test_strip_section_heading_handles_markdown_heading_prefix() -> None:
    text = "### 2. *Thelma* (2017)\nAn icy thriller."
    assert strip_section_heading(text) == "An icy thriller."


def test_strip_section_heading_keeps_text_without_heading() -> None:
    text = "A masterpiece of class tension."
    assert strip_section_heading(text) == text


def test_strip_section_heading_returns_empty_for_heading_only_section() -> None:
    assert strip_section_heading("1. **Parasite** (2019)") == ""


def test_break_out_run_in_labels_splits_label_glued_to_sentence() -> None:
    text = "A heavy entry. **Tone & Pacing:** Slow and mournful."
    assert (
        break_out_run_in_labels(text)
        == "A heavy entry.\n\n**Tone & Pacing:** Slow and mournful."
    )


def test_break_out_run_in_labels_handles_colon_outside_bold() -> None:
    text = "Not an easy watch. **Content note**: Depicted twice."
    assert (
        break_out_run_in_labels(text)
        == "Not an easy watch.\n\n**Content note**: Depicted twice."
    )


def test_break_out_run_in_labels_keeps_paragraph_leading_label() -> None:
    text = "**Why it fits:** A war epic.\n\n**Heads-up:** Runs 2:20."
    assert break_out_run_in_labels(text) == text


def test_break_out_run_in_labels_keeps_bullet_labels() -> None:
    text = "- **Why it fits:** A war epic.\n- **Heads-up:** Runs 2:20."
    assert break_out_run_in_labels(text) == text


def test_break_out_run_in_labels_ignores_bold_without_colon() -> None:
    text = "The **quietest** and heaviest entry."
    assert break_out_run_in_labels(text) == text
