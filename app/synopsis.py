import random
import time

import requests
from bs4 import BeautifulSoup

IMDB_PLOTSUMMARY_URL = "https://www.imdb.com/title/{imdb_id}/plotsummary"
IMDB_TITLE_URL = "https://www.imdb.com/title/{imdb_id}/"
WIKIPEDIA_API = "https://en.wikipedia.org/w/api.php"
WIKIPEDIA_HEADERS = {"User-Agent": "plex-synopsis-bot/1.0"}


def _fetch_imdb_synopsis(page, imdb_id: str) -> str | None:
    url = IMDB_PLOTSUMMARY_URL.format(imdb_id=imdb_id)
    time.sleep(random.uniform(2, 4))  # noqa: S311
    page.goto(url, wait_until="networkidle")
    page.wait_for_timeout(2000)

    soup = BeautifulSoup(page.content(), "html.parser")
    synopsis_section = soup.find(attrs={"data-testid": "sub-section-synopsis"})
    if not synopsis_section:
        return None

    divs = synopsis_section.find_all("div", class_="ipc-html-content-inner-div")
    if not divs:
        return None

    longest = max(divs, key=lambda d: len(d.get_text(strip=True)))
    text = longest.get_text(strip=True)
    return text or None


def _fetch_wikipedia(title: str, year: int) -> str | None:
    try:
        search = requests.get(
            WIKIPEDIA_API,
            headers=WIKIPEDIA_HEADERS,
            params={
                "action": "query",
                "list": "search",
                "srsearch": f"{title} {year} film",
                "format": "json",
                "srlimit": 3,
            },
            timeout=10,
        ).json()

        results = search.get("query", {}).get("search", [])
        if not results:
            return None

        page_title = results[0]["title"]

        extract = requests.get(
            WIKIPEDIA_API,
            headers=WIKIPEDIA_HEADERS,
            params={
                "action": "query",
                "titles": page_title,
                "prop": "extracts",
                "explaintext": True,
                "format": "json",
            },
            timeout=10,
        ).json()

        pages = extract.get("query", {}).get("pages", {})
        content = next(iter(pages.values())).get("extract", "")

        if "== Plot ==" not in content:
            return None

        plot_start = content.index("== Plot ==") + len("== Plot ==")
        rest = content[plot_start:].strip()
        next_section = rest.find("\n==")
        return rest[:next_section].strip() if next_section > 0 else rest.strip()

    except Exception as e:
        print(f"  !! Wikipedia error: {e}")
        return None


def _fetch_imdb_description(page, imdb_id: str) -> str | None:
    url = IMDB_TITLE_URL.format(imdb_id=imdb_id)
    time.sleep(random.uniform(2, 4))  # noqa: S311
    page.goto(url, wait_until="domcontentloaded")
    page.wait_for_timeout(2000)

    soup = BeautifulSoup(page.content(), "html.parser")
    el = (
        soup.select_one("[data-testid='plot-xl']")
        or soup.select_one("[data-testid='plot-l']")
        or soup.select_one("[data-testid='plot']")
    )
    if el:
        text = el.get_text(strip=True)
        return text or None
    return None


def fetch_synopsis(page, imdb_id: str, title: str, year: int) -> str | None:
    print(f"  Fetching synopsis for: {title}")

    synopsis = _fetch_imdb_synopsis(page, imdb_id)
    if synopsis:
        print(f"  -> IMDB synopsis ({len(synopsis)} chars)")
        return synopsis

    synopsis = _fetch_wikipedia(title, year)
    if synopsis:
        print(f"  -> Wikipedia ({len(synopsis)} chars)")
        return synopsis

    synopsis = _fetch_imdb_description(page, imdb_id)
    if synopsis:
        print(f"  -> IMDB description ({len(synopsis)} chars)")
        return synopsis

    print("  -> No synopsis found")
    return None
