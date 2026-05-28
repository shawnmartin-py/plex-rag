from app.browser import browser_context
from app.config import PLEX_MOVIE_LIBRARY, QDRANT_COLLECTION, QDRANT_PATH
from app.plex import Plex
from app.repositories.sql import SqlMediaItems
from app.synopsis import fetch_synopsis


def sync_library() -> None:
    plex = Plex()
    sql_repo = SqlMediaItems()
    sql_repo.load()

    plex_media_items = plex.get_media_items(libraries={PLEX_MOVIE_LIBRARY})

    new_items = []
    for plex_item in plex_media_items:
        if plex_item in sql_repo:
            continue
        print(f"Adding: {plex_item.title}")
        new_items.append(plex_item)

    if new_items:
        sql_repo.save(new_items)
        with browser_context() as context:
            page = context.new_page()
            for item in new_items:
                synopsis = fetch_synopsis(page, item.imdb_id, item.title, item.year)
                if synopsis:
                    item.synopsis = synopsis
                    sql_repo.save([item])
    else:
        print("No new items to add.")

    plex_ids = {item.imdb_id for item in plex_media_items}
    removed_ids = sql_repo.loaded_ids - plex_ids
    if removed_ids:
        print(f"Removing {len(removed_ids)} items no longer in Plex")
        sql_repo.delete(removed_ids)
        _sync_removals_to_vector_store(removed_ids)


def _sync_removals_to_vector_store(imdb_ids: set[str]) -> None:
    from qdrant_client import QdrantClient
    from qdrant_client.models import FieldCondition, Filter, MatchAny

    client = QdrantClient(path=QDRANT_PATH)
    try:
        if not client.collection_exists(QDRANT_COLLECTION):
            return
        client.delete(
            collection_name=QDRANT_COLLECTION,
            points_selector=Filter(must=[FieldCondition(key="metadata.imdb_id", match=MatchAny(any=list(imdb_ids)))]),
        )
        print(f"Removed {len(imdb_ids)} item(s) from vector store")
    finally:
        client.close()
