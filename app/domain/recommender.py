from langchain_core.documents import Document
from langchain_core.messages import BaseMessage

from app.domain.ports import CandidateRetriever, QueryRewriter, RecommendationGenerator


def _format_docs(docs: list[Document]) -> str:
    return "\n\n---\n\n".join(doc.page_content for doc in docs)


def _merge_docs(candidate_sets: list[list[Document]]) -> list[Document]:
    seen: set[str] = set()
    merged: list[Document] = []
    for docs in candidate_sets:
        for doc in docs:
            doc_id = doc.metadata.get("imdb_id")
            if doc_id not in seen:
                seen.add(doc_id)
                merged.append(doc)
    return merged


class MovieRecommender:
    def __init__(
        self,
        retrievers: list[CandidateRetriever],
        generator: RecommendationGenerator,
        rewriter: QueryRewriter,
    ) -> None:
        self._retrievers = retrievers
        self._generator = generator
        self._rewriter = rewriter

    def recommend(self, question: str, history: list[BaseMessage]) -> str:
        standalone = self._rewriter.rewrite(question, history) if history else question
        candidate_sets = [retriever.retrieve(standalone) for retriever in self._retrievers]
        context = _format_docs(_merge_docs(candidate_sets))
        return self._generator.generate(question, context, history)
