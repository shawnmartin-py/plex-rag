from unittest.mock import MagicMock, patch

from app.main import _sync_removals_to_vector_store, sync_library

# --- _sync_removals_to_vector_store ---


def test_sync_removals_skips_delete_when_collection_absent():
    mock_client = MagicMock()
    mock_client.collection_exists.return_value = False
    with patch("qdrant_client.QdrantClient", return_value=mock_client):
        _sync_removals_to_vector_store({"tt001", "tt002"})
    mock_client.delete.assert_not_called()


def test_sync_removals_deletes_when_collection_exists():
    mock_client = MagicMock()
    mock_client.collection_exists.return_value = True
    with patch("qdrant_client.QdrantClient", return_value=mock_client):
        _sync_removals_to_vector_store({"tt001", "tt002"})
    mock_client.delete.assert_called_once()


def test_sync_removals_filter_contains_all_removed_ids():
    mock_client = MagicMock()
    mock_client.collection_exists.return_value = True
    with patch("qdrant_client.QdrantClient", return_value=mock_client):
        _sync_removals_to_vector_store({"tt001", "tt002"})
    points_selector = mock_client.delete.call_args.kwargs["points_selector"]
    id_condition = next(c for c in points_selector.must if "imdb_id" in c.key)
    assert set(id_condition.match.any) == {"tt001", "tt002"}


def test_sync_removals_closes_client_when_collection_absent():
    mock_client = MagicMock()
    mock_client.collection_exists.return_value = False
    with patch("qdrant_client.QdrantClient", return_value=mock_client):
        _sync_removals_to_vector_store({"tt001"})
    mock_client.close.assert_called_once()


def test_sync_removals_closes_client_after_delete():
    mock_client = MagicMock()
    mock_client.collection_exists.return_value = True
    with patch("qdrant_client.QdrantClient", return_value=mock_client):
        _sync_removals_to_vector_store({"tt001"})
    mock_client.close.assert_called_once()


# --- sync_library: vector store cleanup is wired in ---


def _make_plex_item(imdb_id: str) -> MagicMock:
    item = MagicMock()
    item.imdb_id = imdb_id
    return item


def test_sync_library_calls_vector_cleanup_with_removed_ids():
    plex_item = _make_plex_item("tt001")
    mock_sql = MagicMock()
    mock_sql.loaded_ids = {"tt001", "tt999"}  # tt999 absent from Plex
    mock_sql.__contains__ = MagicMock(return_value=True)  # all plex items already in sql

    with (
        patch("app.main.Plex") as MockPlex,
        patch("app.main.SqlMediaItems", return_value=mock_sql),
        patch("app.main._sync_removals_to_vector_store") as mock_cleanup,
    ):
        MockPlex.return_value.get_media_items.return_value = [plex_item]
        sync_library()

    mock_cleanup.assert_called_once_with({"tt999"})


def test_sync_library_skips_vector_cleanup_when_nothing_removed():
    plex_item = _make_plex_item("tt001")
    mock_sql = MagicMock()
    mock_sql.loaded_ids = {"tt001"}  # exact match with Plex
    mock_sql.__contains__ = MagicMock(return_value=True)

    with (
        patch("app.main.Plex") as MockPlex,
        patch("app.main.SqlMediaItems", return_value=mock_sql),
        patch("app.main._sync_removals_to_vector_store") as mock_cleanup,
    ):
        MockPlex.return_value.get_media_items.return_value = [plex_item]
        sync_library()

    mock_cleanup.assert_not_called()


def test_sync_library_deletes_removed_ids_from_sql():
    plex_item = _make_plex_item("tt001")
    mock_sql = MagicMock()
    mock_sql.loaded_ids = {"tt001", "tt999"}
    mock_sql.__contains__ = MagicMock(return_value=True)

    with (
        patch("app.main.Plex") as MockPlex,
        patch("app.main.SqlMediaItems", return_value=mock_sql),
        patch("app.main._sync_removals_to_vector_store"),
    ):
        MockPlex.return_value.get_media_items.return_value = [plex_item]
        sync_library()

    mock_sql.delete.assert_called_once_with({"tt999"})
