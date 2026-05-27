from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings,
    HarmBlockThreshold,
    HarmCategory,
)

from app.adapters.generators import GeminiQueryRewriter, GeminiRecommendationGenerator
from app.adapters.retrievers import (
    DirectSynopsisRetriever,
    HyDEVectorRetriever,
    LLMEnrichmentRetriever,
    LLMKnowledgeRetriever,
)
from app.domain.recommender import MovieRecommender
from app.repositories.sql import SqlMediaItems
from app.services.recommendation import ConversationalRecommendationService
from app.services.vector_store import VectorStoreService

QDRANT_PATH = "./media_items_qdrant_db"
COLLECTION_NAME = "media_items"


def main(spoiler_free: bool = False, verbose: bool = False) -> None:
    embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
    _safety_off = {
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
    }
    llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite", temperature=0, safety_settings=_safety_off)

    sql_repo = SqlMediaItems()
    all_items = sql_repo.load()

    vs_service = VectorStoreService(embeddings, path=QDRANT_PATH, collection_name=COLLECTION_NAME)
    documents = [item.to_document() for item in all_items if item.synopsis]
    vector_store = vs_service.load_or_build(documents)

    doc_by_title = {item.title.lower(): item.to_document() for item in all_items if item.synopsis}
    movie_list_str = "\n".join(f"- {item.title} ({item.year})" for item in sorted(all_items, key=lambda x: x.title))

    recommender = MovieRecommender(
        retrievers=[
            DirectSynopsisRetriever(vector_store, embeddings),
            HyDEVectorRetriever(vector_store, embeddings, llm),
            LLMKnowledgeRetriever(llm, movie_list_str, doc_by_title),
            LLMEnrichmentRetriever(vector_store, embeddings),
        ],
        generator=GeminiRecommendationGenerator(llm, spoiler_free=spoiler_free),
        rewriter=GeminiQueryRewriter(llm),
    )
    service = ConversationalRecommendationService(recommender)

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
