import tempfile

from app.services.vector_store import VectorStoreService
from tests.e2e.conftest import TEST_DOCS, StubEmbeddings


def test_load_or_build_creates_collection(stub_embeddings):
    with tempfile.TemporaryDirectory() as tmpdir:
        service = VectorStoreService(stub_embeddings, path=tmpdir, collection_name="movies")
        store = service.load_or_build(TEST_DOCS)
        assert store is not None


def test_load_or_build_skips_rebuild_on_second_call(stub_embeddings):
    with tempfile.TemporaryDirectory() as tmpdir:
        service = VectorStoreService(stub_embeddings, path=tmpdir, collection_name="movies")
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

        service2 = VectorStoreService(fresh_embeddings, path=tmpdir, collection_name="movies")
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
