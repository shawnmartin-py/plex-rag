from app.bootstrap import build_recommender_service


def main(spoiler_free: bool = False, verbose: bool = False) -> None:
    service, _ = build_recommender_service(spoiler_free=spoiler_free, include_knowledge_retriever=True)

    mode = " (spoiler-free mode)" if spoiler_free else ""
    print(f"\nMovie recommendation bot ready{mode}. Type your request, or 'quit' to exit.\n")
    while True:
        question = input("You: ").strip()
        if question.lower() in {"quit", "exit", "q"}:
            break
        if not question:
            continue
        print(f"\nBot: {service.chat(question, verbose=verbose)}\n")


if __name__ == "__main__":
    main()
