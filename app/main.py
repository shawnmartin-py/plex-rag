from playwright.sync_api import sync_playwright

from app.plex import Plex
from app.repositories.sql import SqlMediaItems
from app.synopsis import fetch_synopsis

plex = Plex()
sql_repo = SqlMediaItems()
sql_repo.load()

plex_media_items = plex.get_media_items(media_types={Plex.MediaType.MOVIE})

new_items = []
for plex_item in plex_media_items:
    if plex_item in sql_repo:
        continue
    print(f"Adding: {plex_item.title}")
    new_items.append(plex_item)

if new_items:
    sql_repo.save(new_items)
    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=True,
            args=["--disable-blink-features=AutomationControlled"],
        )
        context = browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            )
        )
        page = context.new_page()
        page.add_init_script(
            "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"
        )

        for item in new_items:
            synopsis = fetch_synopsis(page, item.imdb_id, item.title, item.year)
            if synopsis:
                item.synopsis = synopsis
                sql_repo.save([item])

        context.close()
        browser.close()

plex_ids = {item.imdb_id for item in plex_media_items}
removed_ids = set(sql_repo._cached_items.keys()) - plex_ids
if removed_ids:
    print(f"Removing {len(removed_ids)} items no longer in Plex")
    sql_repo.delete(removed_ids)
