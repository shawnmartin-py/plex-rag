from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

from app.adapters.generators import GeminiQueryRewriter, GeminiRecommendationGenerator
from app.adapters.retrievers import HyDEVectorRetriever, LLMKnowledgeRetriever
from app.domain.recommender import MovieRecommender
from app.repositories.sql import SqlMediaItems
from app.services.recommendation import ConversationalRecommendationService
from app.services.vector_store import VectorStoreService

QDRANT_PATH = "./media_items_qdrant_db"
COLLECTION_NAME = "media_items"


def main() -> None:
    embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
    llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite", temperature=0)

    sql_repo = SqlMediaItems()
    all_items = sql_repo.load()

    vs_service = VectorStoreService(
        embeddings, path=QDRANT_PATH, collection_name=COLLECTION_NAME
    )
    documents = [item.to_document() for item in all_items if item.synopsis]
    vector_store = vs_service.load_or_build(documents)

    doc_by_title = {
        item.title.lower(): item.to_document() for item in all_items if item.synopsis
    }
    movie_list_str = "\n".join(
        f"- {item.title} ({item.year})"
        for item in sorted(all_items, key=lambda x: x.title)
    )

    recommender = MovieRecommender(
        retrievers=[
            HyDEVectorRetriever(vector_store, embeddings, llm),
            LLMKnowledgeRetriever(llm, movie_list_str, doc_by_title),
        ],
        generator=GeminiRecommendationGenerator(llm),
        rewriter=GeminiQueryRewriter(llm),
    )
    service = ConversationalRecommendationService(recommender)

    print("\nMovie recommendation bot ready. Type your request, or 'quit' to exit.\n")
    while True:
        question = input("You: ").strip()
        if question.lower() in {"quit", "exit", "q"}:
            break
        if not question:
            continue
        print(f"\nBot: {service.chat(question)}\n")


if __name__ == "__main__":
    main()
