from app.browser import browser_context
from app.repositories.sql import SqlMediaItems
from app.synopsis import fetch_synopsis

MOVIES_TO_SKIP = ["tt4943998"]


def scrape_missing_synopses() -> None:
    sql_repo = SqlMediaItems()
    media_items = sql_repo.load()

    items_to_fill = [item for item in media_items if not item.synopsis]
    print(f"{len(items_to_fill)} items without a synopsis")

    if not items_to_fill:
        return

    with browser_context() as context:
        page = context.new_page()
        for item in items_to_fill:
            if item.imdb_id in MOVIES_TO_SKIP:
                continue
            synopsis = fetch_synopsis(page, item.imdb_id, item.title, item.year)
            if synopsis:
                item.synopsis = synopsis
                sql_repo.save([item])
