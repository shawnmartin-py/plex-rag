from abc import ABC, abstractmethod

from langchain_core.documents import Document
from langchain_core.messages import BaseMessage


class CandidateRetriever(ABC):
    name: str

    @abstractmethod
    def retrieve(self, query: str) -> list[Document]: ...


class QueryRewriter(ABC):
    @abstractmethod
    def rewrite(self, question: str, history: list[BaseMessage]) -> str: ...


class RecommendationGenerator(ABC):
    @abstractmethod
    def generate(self, question: str, context: str, history: list[BaseMessage]) -> str: ...
