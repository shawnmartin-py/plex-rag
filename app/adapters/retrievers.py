import json
import re

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_qdrant import QdrantVectorStore

from app.domain.ports import CandidateRetriever


class HyDEVectorRetriever(CandidateRetriever):
    _prompt = ChatPromptTemplate.from_messages(
        [
            (
                "human",
                (
                    "Write a brief fictional movie synopsis (3-4 sentences) that would perfectly match "
                    "this request. Focus on themes, tone, and style. Output only the synopsis, nothing else."
                    "\n\nRequest: {question}"
                ),
            ),
        ]
    )

    def __init__(
        self,
        vector_store: QdrantVectorStore,
        embeddings: GoogleGenerativeAIEmbeddings,
        llm: ChatGoogleGenerativeAI,
        k: int = 8,
    ) -> None:
        self._vector_store = vector_store
        self._embeddings = embeddings
        self._chain = self._prompt | llm | StrOutputParser()
        self._k = k

    def retrieve(self, query: str) -> list[Document]:
        hypothetical = self._chain.invoke({"question": query})
        vector = self._embeddings.embed_documents([hypothetical])[0]
        return self._vector_store.similarity_search_by_vector(vector, k=self._k)


class LLMKnowledgeRetriever(CandidateRetriever):
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
