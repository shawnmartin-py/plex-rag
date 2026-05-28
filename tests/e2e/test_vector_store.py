import tempfile

from langchain_core.documents import Document
from qdrant_client.models import FieldCondition, Filter, MatchValue

from app.services.vector_store import VectorStoreService
from tests.e2e.conftest import TEST_DOCS, StubEmbeddings


def test_load_or_build_creates_collection(stub_embeddings):
    with tempfile.TemporaryDirectory() as tmpdir:
        service = VectorStoreService(path=tmpdir, collection_name="movies", embeddings=stub_embeddings)
        store = service.load_or_build(TEST_DOCS)
        assert store is not None


def test_load_or_build_skips_rebuild_on_second_call(stub_embeddings):
    with tempfile.TemporaryDirectory() as tmpdir:
        service = VectorStoreService(path=tmpdir, collection_name="movies", embeddings=stub_embeddings)
        service.load_or_build(TEST_DOCS)
        service.close()  # release file lock before opening a second client on the same path

        # Second load should only call embed_documents for validation ("dummy_text"), not for all docs
        fresh_embeddings = StubEmbeddings()
        call_log = []
        original = fresh_embeddings.embed_documents
        fresh_embeddings.embed_documents = lambda texts: (
            call_log.append(texts),
            original(texts),
        )[1]

        service2 = VectorStoreService(path=tmpdir, collection_name="movies", embeddings=fresh_embeddings)
        service2.load_or_build(TEST_DOCS)

        assert all(texts == ["dummy_text"] for texts in call_log)


def test_built_store_returns_documents_on_search(qdrant_store, stub_embeddings):
    query_vector = stub_embeddings.embed_documents(["a thriller"])[0]
    results = qdrant_store.similarity_search_by_vector(query_vector, k=3)
    assert len(results) > 0


def test_built_store_contains_all_test_documents(qdrant_store, stub_embeddings):
    query_vector = stub_embeddings.embed_documents(["query"])[0]
    results = qdrant_store.similarity_search_by_vector(query_vector, k=10)
    returned_titles = {doc.metadata["title"] for doc in results}
    expected_titles = {doc.metadata["title"] for doc in TEST_DOCS}
    assert returned_titles == expected_titles


def test_returned_documents_have_correct_metadata(qdrant_store, stub_embeddings):
    query_vector = stub_embeddings.embed_documents(["query"])[0]
    results = qdrant_store.similarity_search_by_vector(query_vector, k=1)
    doc = results[0]
    assert "imdb_id" in doc.metadata
    assert "title" in doc.metadata
    assert "year" in doc.metadata


# --- delete_by_imdb_ids ---


def test_delete_by_imdb_ids_removes_matching_document(stub_embeddings):
    with tempfile.TemporaryDirectory() as tmpdir:
        service = VectorStoreService(path=tmpdir, collection_name="movies", embeddings=stub_embeddings)
        service.load_or_build(TEST_DOCS)

        service.delete_by_imdb_ids({"tt6751668"})  # Parasite

        results = service.store.similarity_search_by_vector(stub_embeddings.embed_query("query"), k=10)
        returned_ids = {doc.metadata["imdb_id"] for doc in results}
        assert "tt6751668" not in returned_ids


def test_delete_by_imdb_ids_preserves_undeleted_documents(stub_embeddings):
    with tempfile.TemporaryDirectory() as tmpdir:
        service = VectorStoreService(path=tmpdir, collection_name="movies", embeddings=stub_embeddings)
        service.load_or_build(TEST_DOCS)

        service.delete_by_imdb_ids({"tt6751668"})  # remove Parasite only

        results = service.store.similarity_search_by_vector(stub_embeddings.embed_query("query"), k=10)
        returned_ids = {doc.metadata["imdb_id"] for doc in results}
        assert "tt0364569" in returned_ids  # Oldboy
        assert "tt4016934" in returned_ids  # The Handmaiden


def test_delete_by_imdb_ids_removes_multiple_at_once(stub_embeddings):
    with tempfile.TemporaryDirectory() as tmpdir:
        service = VectorStoreService(path=tmpdir, collection_name="movies", embeddings=stub_embeddings)
        service.load_or_build(TEST_DOCS)

        service.delete_by_imdb_ids({"tt6751668", "tt0364569"})  # Parasite + Oldboy

        results = service.store.similarity_search_by_vector(stub_embeddings.embed_query("query"), k=10)
        returned_ids = {doc.metadata["imdb_id"] for doc in results}
        assert "tt6751668" not in returned_ids
        assert "tt0364569" not in returned_ids
        assert "tt4016934" in returned_ids  # The Handmaiden untouched


def test_delete_by_imdb_ids_empty_set_is_noop(stub_embeddings):
    with tempfile.TemporaryDirectory() as tmpdir:
        service = VectorStoreService(path=tmpdir, collection_name="movies", embeddings=stub_embeddings)
        service.load_or_build(TEST_DOCS)

        service.delete_by_imdb_ids(set())

        results = service.store.similarity_search_by_vector(stub_embeddings.embed_query("query"), k=10)
        assert len(results) == len(TEST_DOCS)


def test_delete_by_imdb_ids_removes_both_synopsis_and_enriched_docs(stub_embeddings):
    enriched_doc = Document(
        page_content="Craft profile for Parasite.",
        metadata={
            "imdb_id": "tt6751668",
            "title": "Parasite",
            "year": 2019,
            "embedding_type": "enriched",
            "section": "craft",
        },
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        service = VectorStoreService(path=tmpdir, collection_name="movies", embeddings=stub_embeddings)
        service.load_or_build(TEST_DOCS)
        service.add_documents_with_retry([enriched_doc])

        service.delete_by_imdb_ids({"tt6751668"})

        results, _ = service.client.scroll(
            collection_name="movies",
            scroll_filter=Filter(must=[FieldCondition(key="metadata.imdb_id", match=MatchValue(value="tt6751668"))]),
            limit=10,
            with_payload=True,
            with_vectors=False,
        )
        assert len(results) == 0


# --- load_or_build adds missing synopsis docs on reload ---


def test_load_or_build_adds_synopsis_doc_for_new_movie(stub_embeddings):
    new_doc = Document(
        page_content="Title: Drive\nYear: 2011\nIMDb Rating: 7.8\nGenres: Crime, Drama\nSynopsis: A driver.",
        metadata={"imdb_id": "tt0780504", "title": "Drive", "year": 2011, "embedding_type": "synopsis"},
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        service1 = VectorStoreService(path=tmpdir, collection_name="movies", embeddings=stub_embeddings)
        service1.load_or_build(TEST_DOCS)
        service1.close()

        service2 = VectorStoreService(path=tmpdir, collection_name="movies", embeddings=stub_embeddings)
        service2.load_or_build(TEST_DOCS + [new_doc])

        results = service2.store.similarity_search_by_vector(stub_embeddings.embed_query("query"), k=10)
        returned_ids = {doc.metadata["imdb_id"] for doc in results}
        assert "tt0780504" in returned_ids


def test_load_or_build_does_not_duplicate_existing_synopsis_docs(stub_embeddings):
    with tempfile.TemporaryDirectory() as tmpdir:
        service1 = VectorStoreService(path=tmpdir, collection_name="movies", embeddings=stub_embeddings)
        service1.load_or_build(TEST_DOCS)
        service1.close()

        service2 = VectorStoreService(path=tmpdir, collection_name="movies", embeddings=stub_embeddings)
        service2.load_or_build(TEST_DOCS)

        results, _ = service2.client.scroll(
            collection_name="movies",
            scroll_filter=Filter(
                must=[FieldCondition(key="metadata.embedding_type", match=MatchValue(value="synopsis"))]
            ),
            limit=100,
            with_payload=False,
            with_vectors=False,
        )
        assert len(results) == len(TEST_DOCS)
