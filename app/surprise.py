from app.bootstrap import build_diversity_service
from app.domain.diversity import NoWatchHistoryError


def main() -> None:
    service = build_diversity_service()
    if service is None:
        print(
            "\nDiversity mode isn't available yet — the watch_history collection "
            "hasn't been populated. Run plex-ingest's watch_history pipeline "
            "first.\n"
        )
        return

    print(
        "\nSomething different, based on your recent watches. "
        "Press Enter for more, or 'q' to quit.\n"
    )
    while True:
        try:
            items = service.recommend()
        except NoWatchHistoryError:
            print("No recent watch history found — watch something on Plex first!")
            return
        if not items:
            print("Nothing left to recommend from the current pool.\n")
            return
        for item in items:
            rating = f" — ★ {item.imdb_rating}" if item.imdb_rating else ""
            genres = f" ({', '.join(item.genres)})" if item.genres else ""
            print(f"  {item.title} ({item.year}){rating}{genres}")
        again = input("\nShow me more? [Enter/q] ").strip().lower()
        if again == "q":
            break


if __name__ == "__main__":
    main()
