from playwright.sync_api import sync_playwright

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
        page.add_init_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")

        for item in items_to_fill:
            if item.imdb_id in MOVIES_TO_SKIP:
                continue
            synopsis = fetch_synopsis(page, item.imdb_id, item.title, item.year)
            if synopsis:
                item.synopsis = synopsis
                sql_repo.save([item])

        context.close()
        browser.close()
