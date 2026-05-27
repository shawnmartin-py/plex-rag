from app.models.media_item import MediaItem


def make_item(**overrides) -> MediaItem:
    defaults = dict(
        imdb_id="tt6751668",
        type="movie",
        title="Parasite",
        year=2019,
        imdb_rating=8.5,
        content_rating="R",
        genres=["Drama", "Thriller"],
        synopsis="A family schemes their way into a rich household.",
    )
    return MediaItem(**{**defaults, **overrides})


# --- to_document ---


def test_to_document_has_synopsis_embedding_type():
    doc = make_item().to_document()
    assert doc.metadata["embedding_type"] == "synopsis"


def test_to_document_preserves_imdb_id():
    doc = make_item().to_document()
    assert doc.metadata["imdb_id"] == "tt6751668"


def test_to_document_contains_synopsis_in_page_content():
    doc = make_item().to_document()
    assert "A family schemes" in doc.page_content


def test_to_document_does_not_put_synopsis_in_metadata():
    doc = make_item().to_document()
    assert "synopsis" not in doc.metadata


# --- to_enriched_document ---


def test_to_enriched_document_has_enriched_embedding_type():
    doc = make_item().to_enriched_document("Expert analysis.", "craft")
    assert doc.metadata["embedding_type"] == "enriched"


def test_to_enriched_document_stores_section_in_metadata():
    doc = make_item().to_enriched_document("Expert analysis.", "craft")
    assert doc.metadata["section"] == "craft"


def test_to_enriched_document_stores_meaning_section():
    doc = make_item().to_enriched_document("text", "meaning")
    assert doc.metadata["section"] == "meaning"


def test_to_enriched_document_stores_context_section():
    doc = make_item().to_enriched_document("text", "context")
    assert doc.metadata["section"] == "context"


def test_to_enriched_document_uses_enrichment_text_as_page_content():
    doc = make_item().to_enriched_document("Expert analysis.", "craft")
    assert doc.page_content == "Expert analysis."


def test_to_enriched_document_does_not_contain_synopsis_in_page_content():
    doc = make_item().to_enriched_document("Expert analysis.", "craft")
    assert "A family schemes" not in doc.page_content


def test_to_enriched_document_preserves_imdb_id():
    doc = make_item().to_enriched_document("text", "craft")
    assert doc.metadata["imdb_id"] == "tt6751668"


def test_to_enriched_document_preserves_title():
    doc = make_item().to_enriched_document("text", "craft")
    assert doc.metadata["title"] == "Parasite"


def test_to_enriched_document_preserves_year():
    doc = make_item().to_enriched_document("text", "craft")
    assert doc.metadata["year"] == 2019


def test_to_enriched_document_does_not_put_synopsis_in_metadata():
    doc = make_item().to_enriched_document("text", "craft")
    assert "synopsis" not in doc.metadata


def test_embedding_types_differ_between_document_and_enriched_document():
    item = make_item()
    assert (
        item.to_document().metadata["embedding_type"]
        != item.to_enriched_document("text", "craft").metadata["embedding_type"]
    )


def test_sections_produce_distinct_documents():
    item = make_item()
    craft = item.to_enriched_document("craft text", "craft")
    meaning = item.to_enriched_document("meaning text", "meaning")
    assert craft.metadata["section"] != meaning.metadata["section"]
    assert craft.page_content != meaning.page_content
