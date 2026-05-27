import json
import re

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client.models import FieldCondition, Filter, MatchValue

from app.domain.ports import CandidateRetriever


class DirectSynopsisRetriever(CandidateRetriever):
    name = "synopsis"

    def __init__(
        self,
        vector_store: QdrantVectorStore,
        embeddings: GoogleGenerativeAIEmbeddings,
        k: int = 8,
    ) -> None:
        self._vector_store = vector_store
        self._embeddings = embeddings
        self._k = k
        self._filter = Filter(must=[FieldCondition(key="metadata.embedding_type", match=MatchValue(value="synopsis"))])

    def retrieve(self, query: str) -> list[Document]:
        vector = self._embeddings.embed_documents([query])[0]
        return self._vector_store.similarity_search_by_vector(vector, k=self._k, filter=self._filter)


class HyDEVectorRetriever(CandidateRetriever):
    name = "hyde"

    _prompt = ChatPromptTemplate.from_messages(
        [
            (
                "human",
                (
                    "Write a dense expert film profile (4-6 sentences) describing the ideal film for "
                    "this request. Use precise critical vocabulary: exact subgenre labels, cinematic "
                    "movements, director names and influences, tone and mood descriptors, thematic "
                    "keywords, narrative structure, emotional register, visual style, and pacing. "
                    "Every word should serve as a retrieval signal for finding real films that match. "
                    "Output only the profile, nothing else.\n\nRequest: {question}"
                ),
            ),
        ]
    )

    def __init__(
        self,
        vector_store: QdrantVectorStore,
        embeddings: GoogleGenerativeAIEmbeddings,
        llm: ChatGoogleGenerativeAI,
        k: int = 20,
    ) -> None:
        self._vector_store = vector_store
        self._embeddings = embeddings
        self._chain = self._prompt | llm | StrOutputParser()
        self._k = k
        self._filter = Filter(must=[FieldCondition(key="metadata.embedding_type", match=MatchValue(value="enriched"))])

    def retrieve(self, query: str) -> list[Document]:
        hypothetical = self._chain.invoke({"question": query})
        vector = self._embeddings.embed_documents([hypothetical])[0]
        return self._vector_store.similarity_search_by_vector(vector, k=self._k, filter=self._filter)


class LLMKnowledgeRetriever(CandidateRetriever):
    name = "llm-knowledge"

    _prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                (
                    "You are a film expert with deep knowledge of cinema. From the list of movies below, "
                    "select up to 8 that are most relevant to the user's request.\n\n"
                    "Use your knowledge of each film's themes, tone, director, cultural significance, "
                    "subgenre, and critical reception — not just the title. Return ONLY a JSON array of "
                    "the selected titles exactly as they appear in the list (without years), "
                    'e.g.: ["Movie Title", "Another Film"]. No explanation, no markdown, just the JSON array.'
                ),
            ),
            ("human", "Request: {question}\n\nAvailable movies:\n{movie_list}"),
        ]
    )

    def __init__(
        self,
        llm: ChatGoogleGenerativeAI,
        movie_list: str,
        doc_by_title: dict[str, Document],
    ) -> None:
        self._chain = self._prompt | llm | StrOutputParser()
        self._movie_list = movie_list
        self._doc_by_title = doc_by_title

    def retrieve(self, query: str) -> list[Document]:
        response = self._chain.invoke({"question": query, "movie_list": self._movie_list})
        clean = re.sub(r"```(?:json)?|```", "", response).strip()
        try:
            titles: list[str] = json.loads(clean)
        except json.JSONDecodeError:
            titles = []
        return [self._doc_by_title[t.lower()] for t in titles if t.lower() in self._doc_by_title]


class LLMEnrichmentRetriever(CandidateRetriever):
    name = "enricher"

    def __init__(
        self,
        vector_store: QdrantVectorStore,
        embeddings: GoogleGenerativeAIEmbeddings,
        k: int = 8,
        filter_by_type: bool = True,
    ) -> None:
        self._vector_store = vector_store
        self._embeddings = embeddings
        self._k = k
        self._filter = (
            Filter(must=[FieldCondition(key="metadata.embedding_type", match=MatchValue(value="enriched"))])
            if filter_by_type
            else None
        )

    def retrieve(self, query: str) -> list[Document]:
        vector = self._embeddings.embed_documents([query])[0]
        return self._vector_store.similarity_search_by_vector(vector, k=self._k, filter=self._filter)
