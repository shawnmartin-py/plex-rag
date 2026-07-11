from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from app.domain.ports import MediaItemLookup
from app.domain.recommender import CoverageReport, MovieRecommender
from app.models.media_item import MediaItem


class ConversationalRecommendationService:
    def __init__(self, recommender: MovieRecommender) -> None:
        self._recommender = recommender
        self._history: list[BaseMessage] = []

    def chat(
        self, question: str, verbose: bool = False
    ) -> tuple[str, CoverageReport | None]:
        answer, _, coverage = self._recommender.recommend(
            question, self._history, verbose=verbose
        )
        self._history.append(HumanMessage(content=question))
        self._history.append(AIMessage(content=answer))
        return answer, coverage

    def chat_with_items(
        self, question: str, media_repo: MediaItemLookup
    ) -> tuple[str, list[MediaItem]]:
        answer, imdb_ids, _ = self._recommender.recommend(question, self._history)
        self._history.append(HumanMessage(content=question))
        self._history.append(AIMessage(content=answer))
        items = [media_repo.get_by_id(imdb_id) for imdb_id in imdb_ids]
        return answer, [i for i in items if i is not None]

    def reset_history(self) -> None:
        self._history = []
