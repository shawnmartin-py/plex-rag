from collections.abc import Callable
from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from app.domain.recommender import CoverageReport
from app.services.recommendation import ConversationalRecommendationService


def _capture_history_side_effect(
    snapshots: list[list[BaseMessage]],
) -> Callable[..., tuple[str, list[str], CoverageReport | None]]:
    """Records a snapshot of the history seen on each call, then returns a
    canned answer — lets tests assert what history the recommender was
    actually invoked with on each turn."""

    def side_effect(
        question: str, history: list[BaseMessage], **_: object
    ) -> tuple[str, list[str], CoverageReport | None]:
        snapshots.append(list(history))
        return "here are some films", [], None

    return side_effect


@pytest.fixture
def recommender() -> MagicMock:
    mock = MagicMock()
    mock.recommend.return_value = ("here are some films", [], None)
    return mock


@pytest.fixture
def service(recommender: MagicMock) -> ConversationalRecommendationService:
    return ConversationalRecommendationService(recommender)


def test_first_chat_passes_empty_history(recommender: MagicMock) -> None:
    snapshots: list[list[BaseMessage]] = []
    recommender.recommend.side_effect = _capture_history_side_effect(snapshots)
    service = ConversationalRecommendationService(recommender)
    service.chat("recommend a thriller")
    assert snapshots[0] == []


def test_first_chat_returns_answer(
    service: ConversationalRecommendationService,
) -> None:
    answer, coverage = service.chat("recommend a thriller")
    assert answer == "here are some films"
    assert coverage is None


def test_second_chat_includes_first_exchange_in_history(
    recommender: MagicMock,
) -> None:
    snapshots: list[list[BaseMessage]] = []
    recommender.recommend.side_effect = _capture_history_side_effect(snapshots)
    service = ConversationalRecommendationService(recommender)
    service.chat("recommend a thriller")
    service.chat("what about something slower?")

    second_call_history = snapshots[1]
    assert len(second_call_history) == 2
    assert isinstance(second_call_history[0], HumanMessage)
    assert second_call_history[0].content == "recommend a thriller"
    assert isinstance(second_call_history[1], AIMessage)
    assert second_call_history[1].content == "here are some films"


def test_history_grows_with_each_turn(recommender: MagicMock) -> None:
    snapshots: list[list[BaseMessage]] = []
    recommender.recommend.side_effect = _capture_history_side_effect(snapshots)
    service = ConversationalRecommendationService(recommender)
    service.chat("first question")
    service.chat("second question")
    service.chat("third question")

    assert len(snapshots[0]) == 0  # no history before first call
    assert len(snapshots[1]) == 2  # one exchange before second call
    assert len(snapshots[2]) == 4  # two exchanges before third call


def test_each_chat_passes_correct_question(
    service: ConversationalRecommendationService, recommender: MagicMock
) -> None:
    service.chat("recommend a comedy")
    question, _ = recommender.recommend.call_args[0]
    assert question == "recommend a comedy"


def test_history_contains_ai_response_from_recommender(
    service: ConversationalRecommendationService, recommender: MagicMock
) -> None:
    recommender.recommend.return_value = ("my custom answer", [], None)
    service.chat("question one")
    service.chat("question two")

    _, history = recommender.recommend.call_args[0]
    ai_messages = [m for m in history if isinstance(m, AIMessage)]
    assert ai_messages[0].content == "my custom answer"
