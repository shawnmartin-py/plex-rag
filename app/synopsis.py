import random
import re
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


def _titles_match(movie_title: str, wiki_title: str) -> bool:
    """Return True if wiki_title plausibly refers to the same film as movie_title."""

    def _normalize(s: str) -> str:
        s = re.sub(r"\([^)]*\)", "", s)  # drop "(film)", "(2019 film)", etc.
        return re.sub(r"[^a-z0-9 ]", "", s.lower()).strip()

    movie_norm = _normalize(movie_title)
    wiki_norm = _normalize(wiki_title)
    return movie_norm in wiki_norm or wiki_norm in movie_norm


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

        page_title = next(
            (r["title"] for r in results if _titles_match(title, r["title"])),
            None,
        )
        if page_title is None:
            return None

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

    except (requests.RequestException, KeyError, StopIteration, ValueError) as e:
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
