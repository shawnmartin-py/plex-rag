from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from app.services.recommendation import ConversationalRecommendationService


@pytest.fixture
def recommender():
    mock = MagicMock()
    mock.recommend.return_value = "here are some films"
    return mock


@pytest.fixture
def service(recommender):
    return ConversationalRecommendationService(recommender)


def test_first_chat_passes_empty_history(recommender):
    # History is a mutable list passed by reference, so we capture a snapshot at call time
    snapshots = []
    recommender.recommend.side_effect = lambda q, h: (
        snapshots.append(list(h)),
        "here are some films",
    )[1]
    service = ConversationalRecommendationService(recommender)
    service.chat("recommend a thriller")
    assert snapshots[0] == []


def test_first_chat_returns_answer(service):
    result = service.chat("recommend a thriller")
    assert result == "here are some films"


def test_second_chat_includes_first_exchange_in_history(recommender):
    snapshots = []
    recommender.recommend.side_effect = lambda q, h: (
        snapshots.append(list(h)),
        "here are some films",
    )[1]
    service = ConversationalRecommendationService(recommender)
    service.chat("recommend a thriller")
    service.chat("what about something slower?")

    second_call_history = snapshots[1]
    assert len(second_call_history) == 2
    assert isinstance(second_call_history[0], HumanMessage)
    assert second_call_history[0].content == "recommend a thriller"
    assert isinstance(second_call_history[1], AIMessage)
    assert second_call_history[1].content == "here are some films"


def test_history_grows_with_each_turn(recommender):
    snapshots = []
    recommender.recommend.side_effect = lambda q, h: (
        snapshots.append(list(h)),
        "here are some films",
    )[1]
    service = ConversationalRecommendationService(recommender)
    service.chat("first question")
    service.chat("second question")
    service.chat("third question")

    assert len(snapshots[0]) == 0  # no history before first call
    assert len(snapshots[1]) == 2  # one exchange before second call
    assert len(snapshots[2]) == 4  # two exchanges before third call


def test_each_chat_passes_correct_question(service, recommender):
    service.chat("recommend a comedy")
    question, _ = recommender.recommend.call_args[0]
    assert question == "recommend a comedy"


def test_history_contains_ai_response_from_recommender(service, recommender):
    recommender.recommend.return_value = "my custom answer"
    service.chat("question one")
    service.chat("question two")

    _, history = recommender.recommend.call_args[0]
    ai_messages = [m for m in history if isinstance(m, AIMessage)]
    assert ai_messages[0].content == "my custom answer"
