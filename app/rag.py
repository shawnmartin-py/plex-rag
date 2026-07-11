import asyncio

from app.bootstrap import build_recommender_service
from app.domain.recommender import CoverageReport


def _print_coverage(coverage: CoverageReport) -> None:
    col = 44
    print("\n[Source coverage]")

    if coverage.recommended:
        print(f"  {'RECOMMENDED':<{col}}  source(s)")
        print(f"  {'─' * col}  {'─' * 22}")
        for entry in coverage.recommended:
            label = f"{entry.title} ({entry.year})"
            print(f"  {label:<{col}}  {', '.join(sorted(entry.sources))}")

    if coverage.dropped:
        print(f"\n  {'DROPPED (in context, not recommended)':<{col}}  source(s)")
        print(f"  {'─' * col}  {'─' * 22}")
        for entry in coverage.dropped:
            label = f"{entry.title} ({entry.year})"
            print(f"  {label:<{col}}  {', '.join(sorted(entry.sources))}")

    counts = {name: 0 for name in coverage.retriever_names}
    for entry in coverage.recommended:
        for name in entry.sources:
            if name in counts:
                counts[name] += 1
    total = len(coverage.recommended)
    summary = "  · ".join(
        f"{name} {counts[name]}/{total}" for name in coverage.retriever_names
    )
    print(f"\n  Coverage: {summary}\n")


async def main(spoiler_free: bool = False, verbose: bool = False) -> None:
    service, _, _ = build_recommender_service(
        spoiler_free=spoiler_free, include_knowledge_retriever=True
    )

    mode = " (spoiler-free mode)" if spoiler_free else ""
    print(
        f"\nMovie recommendation bot ready{mode}. "
        "Type your request, or 'quit' to exit.\n"
    )
    while True:
        question = input("You: ").strip()
        if question.lower() in {"quit", "exit", "q"}:
            break
        if not question:
            continue
        answer, coverage = await service.chat(question, verbose=verbose)
        print(f"\nBot: {answer}\n")
        if coverage is not None:
            _print_coverage(coverage)


if __name__ == "__main__":
    asyncio.run(main())
