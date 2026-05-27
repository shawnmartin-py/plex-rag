from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from app.domain.recommender import MovieRecommender


class ConversationalRecommendationService:
    def __init__(self, recommender: MovieRecommender) -> None:
        self._recommender = recommender
        self._history: list[BaseMessage] = []

    def chat(self, question: str) -> str:
        answer = self._recommender.recommend(question, self._history)
        self._history.append(HumanMessage(content=question))
        self._history.append(AIMessage(content=answer))
        return answer
