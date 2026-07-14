import pytest
from nicegui import ui
from nicegui.testing import user_simulation

from app.models.media_item import HdrFormat, MediaItem
from nicegui_app.components import (
    render_chat_row,
    render_recommendations,
    render_surprise_results,
)


def make_item(
    imdb_id: str,
    title: str,
    hdr_formats: list[HdrFormat] | None = None,
    runtime_minutes: int | None = None,
) -> MediaItem:
    return MediaItem(
        imdb_id=imdb_id,
        type="movie",
        title=title,
        year=2020,
        imdb_rating=8.0,
        content_rating="R",
        genres=["Drama"],
        hdr_formats=hdr_formats or [],
        runtime_minutes=runtime_minutes,
    )


# --- render_recommendations ---


@pytest.mark.anyio
async def test_render_recommendations_pairs_items_positionally_not_by_title() -> None:
    """The generator's numbered headings can name the wrong film (sequels/
    reboots sharing a title) — pairing must be purely positional against the
    `items` list, never by matching the heading text. See the "fix wrong
    movie posters" commit."""
    response = (
        "1. **Parasite** (2019)\nFirst body.\n\n2. **Old Boy** (2003)\nSecond body."
    )
    items = [
        make_item("tt1", "Totally Different Title"),
        make_item("tt2", "Another Unrelated Title"),
    ]

    async def root() -> None:
        container = ui.column()
        render_recommendations(container, response, items)

    async with user_simulation(root=root) as user:
        await user.open("/")
        await user.should_see(content="Totally Different Title")
        await user.should_see(content="Another Unrelated Title")
        # The LLM's own (mismatched) heading text must never render — the
        # heading is stripped and replaced by the paired item's own title.
        await user.should_not_see(content="Parasite")
        await user.should_not_see(content="Old Boy")


@pytest.mark.anyio
async def test_render_recommendations_marks_only_first_card_as_top_pick() -> None:
    response = (
        "1. **A** (2019)\nBody A.\n\n"
        "2. **B** (2020)\nBody B.\n\n"
        "3. **C** (2021)\nBody C."
    )
    items = [make_item("tt1", "A"), make_item("tt2", "B"), make_item("tt3", "C")]

    async def root() -> None:
        container = ui.column()
        render_recommendations(container, response, items)

    async with user_simulation(root=root) as user:
        await user.open("/")
        await user.should_see(content="Top pick")
        interaction = user.find(content="Top pick")
        assert len(interaction.elements) == 1


@pytest.mark.anyio
async def test_render_recommendations_falls_back_without_numbered_sections() -> None:
    response = "I don't have a great match in your library for that request."
    items = [make_item("tt1", "Some Movie")]

    async def root() -> None:
        container = ui.column()
        render_recommendations(container, response, items)

    async with user_simulation(root=root) as user:
        await user.open("/")
        await user.should_see(content=response)
        await user.should_not_see(content="Top pick")
        await user.should_not_see(content="Some Movie")


@pytest.mark.anyio
async def test_render_recommendations_falls_back_to_markdown_when_no_items() -> None:
    response = "1. **Parasite** (2019)\nGreat pick.\n\n2. **Thelma** (2017)\nAlso good."

    async def root() -> None:
        container = ui.column()
        render_recommendations(container, response, [])

    async with user_simulation(root=root) as user:
        await user.open("/")
        # Whole response rendered verbatim as plain markdown, headings intact.
        await user.should_see(content="Parasite")
        await user.should_not_see(content="Top pick")


@pytest.mark.anyio
async def test_render_recommendations_extra_sections_render_as_plain_markdown() -> None:
    """More numbered sections than items: the sections beyond the item list
    fall back to plain markdown (raw, heading included), only the sections
    with a paired item become cards."""
    response = (
        "1. **A** (2019)\nBody A.\n\n"
        "2. **B** (2020)\nBody B.\n\n"
        "3. **Mulholland Drive** (2001)\nBody C."
    )
    items = [make_item("tt1", "A"), make_item("tt2", "B")]

    async def root() -> None:
        container = ui.column()
        render_recommendations(container, response, items)

    async with user_simulation(root=root) as user:
        await user.open("/")
        await user.should_see(content="A")
        await user.should_see(content="B")
        # The third section's heading survives verbatim since it wasn't
        # stripped by _render_movie_card's strip_section_heading call.
        await user.should_see(content="Mulholland Drive")
        interaction = user.find(content="Top pick")
        assert len(interaction.elements) == 1


# --- render_surprise_results ---


@pytest.mark.anyio
async def test_render_surprise_results_renders_every_item_without_numbered_text() -> (
    None
):
    """Surprise-me answers are plain prose with no numbered sections — this
    is the replay path `render_recommendations` can't handle (its "no
    numbered sections" fallback intentionally drops items, since that also
    covers a regular chat turn's declined-to-recommend response)."""
    response = "Something different, based on your recent watches:"
    items = [make_item("tt1", "Paprika"), make_item("tt2", "Perfect Blue")]

    async def root() -> None:
        container = ui.column()
        render_surprise_results(container, response, items)

    async with user_simulation(root=root) as user:
        await user.open("/")
        await user.should_see(content=response)
        await user.should_see(content="Paprika")
        await user.should_see(content="Perfect Blue")
        await user.should_see(content="Top pick")
        interaction = user.find(content="Top pick")
        assert len(interaction.elements) == 1


@pytest.mark.anyio
async def test_render_surprise_results_with_no_items_renders_text_only() -> None:
    response = "Nothing left to recommend right now — try again later."

    async def root() -> None:
        container = ui.column()
        render_surprise_results(container, response, [])

    async with user_simulation(root=root) as user:
        await user.open("/")
        await user.should_see(content=response)
        await user.should_not_see(content="Top pick")


@pytest.mark.anyio
async def test_render_recommendations_renders_imdb_rating_badge() -> None:
    """The rating badge is a raw anchor (so the star can be a tungsten span)
    — it must still carry the rating and the IMDb link."""
    response = "1. **A** (2019)\nBody A."
    items = [make_item("tt1", "A")]

    async def root() -> None:
        container = ui.column()
        render_recommendations(container, response, items)

    async with user_simulation(root=root) as user:
        await user.open("/")
        await user.should_see(content="8.0")
        await user.should_see(content="https://www.imdb.com/title/tt1/")


@pytest.mark.anyio
async def test_render_recommendations_renders_hdr_format_badges() -> None:
    """A dual-layer Dolby Vision file carries both formats at once — every
    listed format gets its own mark, in list order."""
    response = "1. **A** (2019)\nBody A."
    items = [make_item("tt1", "A", hdr_formats=[HdrFormat.HDR, HdrFormat.DV])]

    async def root() -> None:
        container = ui.column()
        render_recommendations(container, response, items)

    async with user_simulation(root=root) as user:
        await user.open("/")
        await user.should_see(marker="hdr-badge-hdr")
        await user.should_see(marker="hdr-badge-dv")


@pytest.mark.anyio
async def test_render_recommendations_renders_runtime() -> None:
    response = "1. **A** (2019)\nBody A."
    items = [make_item("tt1", "A", runtime_minutes=104)]

    async def root() -> None:
        container = ui.column()
        render_recommendations(container, response, items)

    async with user_simulation(root=root) as user:
        await user.open("/")
        await user.should_see(content="1h 44m")


@pytest.mark.anyio
async def test_render_recommendations_omits_runtime_when_unresolved() -> None:
    """A streaming-placeholder movie whose OMDb lookup hasn't resolved carries
    `runtime_minutes=None` — the card must not show a blank/garbled runtime."""
    response = "1. **A** (2019)\nBody A."
    items = [make_item("tt1", "A", runtime_minutes=None)]

    async def root() -> None:
        container = ui.column()
        render_recommendations(container, response, items)

    async with user_simulation(root=root) as user:
        await user.open("/")
        await user.should_not_see(marker="plex-runtime")


@pytest.mark.anyio
async def test_render_recommendations_renders_content_rating_cert() -> None:
    response = "1. **A** (2019)\nBody A."
    items = [make_item("tt1", "A")]

    async def root() -> None:
        container = ui.column()
        render_recommendations(container, response, items)

    async with user_simulation(root=root) as user:
        await user.open("/")
        await user.should_see(content="R")


@pytest.mark.anyio
async def test_render_recommendations_omits_hdr_badges_for_sdr_item() -> None:
    response = "1. **A** (2019)\nBody A."
    items = [make_item("tt1", "A")]

    async def root() -> None:
        container = ui.column()
        render_recommendations(container, response, items)

    async with user_simulation(root=root) as user:
        await user.open("/")
        await user.should_not_see(marker="hdr-badge-hdr")
        await user.should_not_see(marker="hdr-badge-dv")


# --- render_chat_row ---


@pytest.mark.anyio
async def test_render_chat_row_user_message_renders_content() -> None:
    async def root() -> None:
        container = ui.column()
        render_chat_row(container, "user", "Recommend a heist thriller")

    async with user_simulation(root=root) as user:
        await user.open("/")
        await user.should_see(content="Recommend a heist thriller")


@pytest.mark.anyio
async def test_render_chat_row_assistant_empty_content_renders_nothing_yet() -> None:
    async def root() -> None:
        container = ui.column()
        render_chat_row(container, "assistant", "")

    async with user_simulation(root=root) as user:
        await user.open("/")
        await user.should_not_see(kind=ui.markdown)


@pytest.mark.anyio
async def test_render_chat_row_returns_body_that_can_be_populated_later() -> None:
    """`render_chat_row` returns the row body so a caller can stream content
    into a pending assistant turn once the answer is ready."""

    async def root() -> None:
        container = ui.column()
        body = render_chat_row(container, "assistant", "")
        with body:
            ui.markdown("The answer arrives later")

    async with user_simulation(root=root) as user:
        await user.open("/")
        await user.should_see(content="The answer arrives later")
