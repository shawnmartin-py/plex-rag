from unittest.mock import MagicMock, patch

import pytest

from app.models.media_item import MediaItem
from app.services.enrichment import SECTIONS, EnrichmentService


def make_item(imdb_id: str = "tt001", synopsis: str | None = "A great film.") -> MediaItem:
    return MediaItem(
        imdb_id=imdb_id,
        type="movie",
        title="Test Film",
        year=2020,
        imdb_rating=7.5,
        content_rating="PG-13",
        genres=["Drama", "Sci-Fi"],
        synopsis=synopsis,
    )


def make_service(scroll_results=None):
    """Create an EnrichmentService with a fully mocked VectorStoreService."""
    mock_vs = MagicMock()
    mock_vs.client.scroll.return_value = (scroll_results or [], None)
    service = EnrichmentService(MagicMock(), mock_vs, "test_collection")
    for section in SECTIONS:
        service._chains[section] = MagicMock()
        service._chains[section].invoke.return_value = f"Expert {section} text."
    return service, mock_vs


# --- _already_enriched ---


def test_already_enriched_returns_false_when_scroll_is_empty():
    service, _ = make_service(scroll_results=[])
    assert service._already_enriched("tt001", "craft") is False


def test_already_enriched_returns_true_when_scroll_finds_a_result():
    service, _ = make_service(scroll_results=[MagicMock()])
    assert service._already_enriched("tt001", "craft") is True


def test_already_enriched_passes_correct_collection_name():
    service, mock_vs = make_service()
    service._already_enriched("tt001", "craft")
    assert mock_vs.client.scroll.call_args.kwargs["collection_name"] == "test_collection"


def test_already_enriched_filters_by_correct_imdb_id():
    service, mock_vs = make_service()
    service._already_enriched("tt999", "craft")
    conditions = mock_vs.client.scroll.call_args.kwargs["scroll_filter"].must
    imdb_condition = next(c for c in conditions if "imdb_id" in c.key)
    assert imdb_condition.match.value == "tt999"


def test_already_enriched_filters_by_enriched_embedding_type():
    service, mock_vs = make_service()
    service._already_enriched("tt001", "craft")
    conditions = mock_vs.client.scroll.call_args.kwargs["scroll_filter"].must
    type_condition = next(c for c in conditions if "embedding_type" in c.key)
    assert type_condition.match.value == "enriched"


def test_already_enriched_filters_by_section():
    service, mock_vs = make_service()
    service._already_enriched("tt001", "meaning")
    conditions = mock_vs.client.scroll.call_args.kwargs["scroll_filter"].must
    section_condition = next(c for c in conditions if "section" in c.key)
    assert section_condition.match.value == "meaning"


# --- _generate_section ---


def test_generate_section_returns_chain_output():
    service, _ = make_service()
    service._chains["craft"].invoke.return_value = "Craft profile text."
    assert service._generate_section(make_item(), "craft") == "Craft profile text."


def test_generate_section_invokes_correct_chain():
    service, _ = make_service()
    service._generate_section(make_item(), "meaning")
    service._chains["meaning"].invoke.assert_called_once()
    service._chains["craft"].invoke.assert_not_called()


def test_generate_section_passes_title_and_year():
    item = make_item()
    item.title = "My Film"
    item.year = 2021
    service, _ = make_service()
    service._generate_section(item, "craft")
    args = service._chains["craft"].invoke.call_args[0][0]
    assert args["title"] == "My Film"
    assert args["year"] == 2021


def test_generate_section_joins_genres_as_string():
    item = make_item()
    item.genres = ["Drama", "Sci-Fi"]
    service, _ = make_service()
    service._generate_section(item, "craft")
    args = service._chains["craft"].invoke.call_args[0][0]
    assert args["genres"] == "Drama, Sci-Fi"


@patch("app.services.enrichment.time.sleep")
def test_generate_section_retries_on_429(mock_sleep):
    service, _ = make_service()
    service._chains["craft"].invoke.side_effect = [Exception("429: quota exceeded"), "Success"]
    result = service._generate_section(make_item(), "craft")
    assert result == "Success"
    mock_sleep.assert_called_once()


@patch("app.services.enrichment.time.sleep")
def test_generate_section_retries_on_resource_exhausted(mock_sleep):
    service, _ = make_service()
    service._chains["craft"].invoke.side_effect = [Exception("RESOURCE_EXHAUSTED"), "Success"]
    result = service._generate_section(make_item(), "craft")
    assert result == "Success"
    mock_sleep.assert_called_once()


def test_generate_section_retries_without_synopsis_on_first_empty_response():
    service, _ = make_service()
    service._chains["craft"].invoke.side_effect = ["", "Craft profile text."]
    result = service._generate_section(make_item(), "craft")
    assert result == "Craft profile text."
    assert service._chains["craft"].invoke.call_count == 2


def test_generate_section_second_call_passes_placeholder_synopsis():
    service, _ = make_service()
    service._chains["craft"].invoke.side_effect = ["", "Craft profile text."]
    service._generate_section(make_item(), "craft")
    second_call_input = service._chains["craft"].invoke.call_args_list[1][0][0]
    assert second_call_input["synopsis"] == "(synopsis unavailable)"


def test_generate_section_returns_none_when_both_attempts_empty():
    service, _ = make_service()
    service._chains["craft"].invoke.return_value = ""
    assert service._generate_section(make_item(), "craft") is None
    assert service._chains["craft"].invoke.call_count == 2


def test_generate_section_returns_none_on_whitespace_only_response():
    service, _ = make_service()
    service._chains["craft"].invoke.side_effect = ["   \n  ", ""]
    assert service._generate_section(make_item(), "craft") is None


def test_generate_section_raises_immediately_on_other_errors():
    service, _ = make_service()
    service._chains["craft"].invoke.side_effect = ValueError("Something broke")
    with pytest.raises(ValueError):
        service._generate_section(make_item(), "craft")


@patch("app.services.enrichment.time.sleep")
def test_generate_section_doubles_delay_on_successive_failures(mock_sleep):
    service, _ = make_service()
    service._chains["craft"].invoke.side_effect = [Exception("429"), Exception("429"), "Success"]
    service._generate_section(make_item(), "craft")
    delays = [call.args[0] for call in mock_sleep.call_args_list]
    assert delays[1] > delays[0]


# --- build ---


@patch("app.services.enrichment.time.sleep")
def test_build_skips_items_without_synopsis(mock_sleep):
    service, mock_vs = make_service()
    service.build([make_item(synopsis=None)])
    for chain in service._chains.values():
        chain.invoke.assert_not_called()
    mock_vs.add_documents_with_retry.assert_not_called()


@patch("app.services.enrichment.time.sleep")
def test_build_generates_all_three_sections_per_movie(mock_sleep):
    service, mock_vs = make_service(scroll_results=[])
    service.build([make_item()])
    for section, chain in service._chains.items():
        chain.invoke.assert_called_once()


@patch("app.services.enrichment.time.sleep")
def test_build_skips_already_enriched_sections(mock_sleep):
    service, mock_vs = make_service(scroll_results=[MagicMock()])
    service.build([make_item()])
    for chain in service._chains.values():
        chain.invoke.assert_not_called()
    mock_vs.add_documents_with_retry.assert_not_called()


@patch("app.services.enrichment.time.sleep")
def test_build_embeds_all_sections_together_per_movie(mock_sleep):
    service, mock_vs = make_service(scroll_results=[])
    service.build([make_item()])
    mock_vs.add_documents_with_retry.assert_called_once()
    docs = mock_vs.add_documents_with_retry.call_args[0][0]
    assert len(docs) == len(SECTIONS)


@patch("app.services.enrichment.time.sleep")
def test_build_documents_have_enriched_embedding_type(mock_sleep):
    service, mock_vs = make_service(scroll_results=[])
    service.build([make_item()])
    docs = mock_vs.add_documents_with_retry.call_args[0][0]
    assert all(doc.metadata["embedding_type"] == "enriched" for doc in docs)


@patch("app.services.enrichment.time.sleep")
def test_build_documents_have_correct_sections(mock_sleep):
    service, mock_vs = make_service(scroll_results=[])
    service.build([make_item()])
    docs = mock_vs.add_documents_with_retry.call_args[0][0]
    assert {doc.metadata["section"] for doc in docs} == set(SECTIONS)


@patch("app.services.enrichment.time.sleep")
def test_build_preserves_imdb_id_in_all_section_docs(mock_sleep):
    service, mock_vs = make_service(scroll_results=[])
    service.build([make_item(imdb_id="tt999")])
    docs = mock_vs.add_documents_with_retry.call_args[0][0]
    assert all(doc.metadata["imdb_id"] == "tt999" for doc in docs)


@patch("app.services.enrichment.time.sleep")
def test_build_embeds_once_per_movie(mock_sleep):
    service, mock_vs = make_service(scroll_results=[])
    service.build([make_item(imdb_id="tt001"), make_item(imdb_id="tt002")])
    assert mock_vs.add_documents_with_retry.call_count == 2


@patch("app.services.enrichment.time.sleep")
def test_build_sleeps_between_movies_not_after_last(mock_sleep):
    service, _ = make_service(scroll_results=[])
    service.build([make_item(imdb_id="tt001"), make_item(imdb_id="tt002")])
    assert mock_sleep.call_count == 1


@patch("app.services.enrichment.time.sleep")
def test_build_no_sleep_for_single_movie(mock_sleep):
    service, _ = make_service(scroll_results=[])
    service.build([make_item()])
    mock_sleep.assert_not_called()


@patch("app.services.enrichment.time.sleep")
def test_build_skips_blocked_sections_without_crashing(mock_sleep):
    service, mock_vs = make_service(scroll_results=[])
    # craft returns empty (safety blocked), meaning and context succeed
    service._chains["craft"].invoke.return_value = ""
    service._chains["meaning"].invoke.return_value = "Meaning text."
    service._chains["context"].invoke.return_value = "Context text."
    service.build([make_item()])
    docs = mock_vs.add_documents_with_retry.call_args[0][0]
    assert len(docs) == 2
    assert {doc.metadata["section"] for doc in docs} == {"meaning", "context"}


@patch("app.services.enrichment.time.sleep")
def test_build_does_not_embed_when_all_sections_blocked(mock_sleep):
    service, mock_vs = make_service(scroll_results=[])
    for section in SECTIONS:
        service._chains[section].invoke.return_value = ""
    service.build([make_item()])
    mock_vs.add_documents_with_retry.assert_not_called()


@patch("app.services.enrichment.time.sleep")
def test_build_partially_skips_already_enriched_sections(mock_sleep):
    service, mock_vs = make_service()
    # craft exists, meaning and context do not
    service._vs_service.client.scroll.side_effect = [
        ([MagicMock()], None),  # craft → already enriched
        ([], None),  # meaning → pending
        ([], None),  # context → pending
    ]
    service.build([make_item()])
    docs = mock_vs.add_documents_with_retry.call_args[0][0]
    assert len(docs) == 2
    assert {doc.metadata["section"] for doc in docs} == {"meaning", "context"}
