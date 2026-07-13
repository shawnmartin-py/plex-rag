from collections.abc import AsyncIterator, Callable
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.domain.ports import ChatMessage, SectionReady, StreamEvent, TextDelta
from app.domain.recommender import CoverageReport, StreamedAnswer
from app.models.media_item import MediaItem
from app.services.recommendation import CardReady, ConversationalRecommendationService


def _capture_history_side_effect(
    snapshots: list[list[ChatMessage]],
) -> Callable[..., tuple[str, list[str], CoverageReport | None]]:
    """Records a snapshot of the history seen on each call, then returns a
    canned answer — lets tests assert what history the recommender was
    actually invoked with on each turn."""

    def side_effect(
        question: str, history: list[ChatMessage], **_: object
    ) -> tuple[str, list[str], CoverageReport | None]:
        snapshots.append(list(history))
        return "here are some films", [], None

    return side_effect


@pytest.fixture
def recommender() -> MagicMock:
    mock = MagicMock()
    mock.recommend = AsyncMock(return_value=("here are some films", [], None))
    return mock


@pytest.fixture
def service(recommender: MagicMock) -> ConversationalRecommendationService:
    return ConversationalRecommendationService(recommender)


async def test_first_chat_passes_empty_history(recommender: MagicMock) -> None:
    snapshots: list[list[ChatMessage]] = []
    recommender.recommend.side_effect = _capture_history_side_effect(snapshots)
    service = ConversationalRecommendationService(recommender)
    await service.chat("recommend a thriller")
    assert snapshots[0] == []


async def test_first_chat_returns_answer(
    service: ConversationalRecommendationService,
) -> None:
    answer, coverage = await service.chat("recommend a thriller")
    assert answer == "here are some films"
    assert coverage is None


async def test_second_chat_includes_first_exchange_in_history(
    recommender: MagicMock,
) -> None:
    snapshots: list[list[ChatMessage]] = []
    recommender.recommend.side_effect = _capture_history_side_effect(snapshots)
    service = ConversationalRecommendationService(recommender)
    await service.chat("recommend a thriller")
    await service.chat("what about something slower?")

    second_call_history = snapshots[1]
    assert len(second_call_history) == 2
    assert second_call_history[0].role == "human"
    assert second_call_history[0].content == "recommend a thriller"
    assert second_call_history[1].role == "ai"
    assert second_call_history[1].content == "here are some films"


async def test_history_grows_with_each_turn(recommender: MagicMock) -> None:
    snapshots: list[list[ChatMessage]] = []
    recommender.recommend.side_effect = _capture_history_side_effect(snapshots)
    service = ConversationalRecommendationService(recommender)
    await service.chat("first question")
    await service.chat("second question")
    await service.chat("third question")

    assert len(snapshots[0]) == 0  # no history before first call
    assert len(snapshots[1]) == 2  # one exchange before second call
    assert len(snapshots[2]) == 4  # two exchanges before third call


async def test_each_chat_passes_correct_question(
    service: ConversationalRecommendationService, recommender: MagicMock
) -> None:
    await service.chat("recommend a comedy")
    question, _ = recommender.recommend.call_args[0]
    assert question == "recommend a comedy"


async def test_history_contains_ai_response_from_recommender(
    service: ConversationalRecommendationService, recommender: MagicMock
) -> None:
    recommender.recommend.return_value = ("my custom answer", [], None)
    await service.chat("question one")
    await service.chat("question two")

    _, history = recommender.recommend.call_args[0]
    ai_messages = [m for m in history if m.role == "ai"]
    assert ai_messages[0].content == "my custom answer"


# --- chat_with_items_stream ---


async def _aevents(events: list[StreamEvent]) -> AsyncIterator[StreamEvent]:
    for event in events:
        yield event


def _make_media_item(imdb_id: str) -> MediaItem:
    return MediaItem(
        imdb_id=imdb_id,
        type="movie",
        title="Parasite",
        year=2019,
        imdb_rating=8.5,
        content_rating="R",
        genres=["Thriller"],
    )


async def test_chat_with_items_stream_passes_through_text_deltas(
    recommender: MagicMock,
) -> None:
    recommender.recommend_stream = AsyncMock(
        return_value=StreamedAnswer(
            events=_aevents([TextDelta(text="Hello there.")]),
            answer="Hello there.",
            imdb_ids=[],
        )
    )
    service = ConversationalRecommendationService(recommender)
    streamed = await service.chat_with_items_stream("recommend a thriller", MagicMock())

    events = [event async for event in streamed.events]
    assert events == [TextDelta(text="Hello there.")]


async def test_chat_with_items_stream_resolves_section_ready_to_card_ready(
    recommender: MagicMock,
) -> None:
    item = _make_media_item("tt001")
    recommender.recommend_stream = AsyncMock(
        return_value=StreamedAnswer(
            events=_aevents([SectionReady(imdb_id="tt001", body_md="Great pick.")]),
            answer="1. Parasite\nGreat pick.",
            imdb_ids=["tt001"],
        )
    )
    service = ConversationalRecommendationService(recommender)
    media_repo = MagicMock()
    media_repo.get_by_id.return_value = item
    streamed = await service.chat_with_items_stream("recommend a thriller", media_repo)

    events = [event async for event in streamed.events]

    assert events == [CardReady(item=item, body_md="Great pick.")]
    media_repo.get_by_id.assert_called_once_with("tt001")


async def test_chat_with_items_stream_skips_unresolved_section(
    recommender: MagicMock,
) -> None:
    recommender.recommend_stream = AsyncMock(
        return_value=StreamedAnswer(
            events=_aevents([SectionReady(imdb_id="tt999", body_md="Unknown film.")]),
            answer="Unknown film.",
            imdb_ids=["tt999"],
        )
    )
    service = ConversationalRecommendationService(recommender)
    media_repo = MagicMock()
    media_repo.get_by_id.return_value = None
    streamed = await service.chat_with_items_stream("recommend a thriller", media_repo)

    events = [event async for event in streamed.events]

    assert events == [CardReady(item=None, body_md="Unknown film.")]
    assert streamed.items == []


async def test_chat_with_items_stream_resolves_items_after_completion(
    recommender: MagicMock,
) -> None:
    item = _make_media_item("tt001")
    recommender.recommend_stream = AsyncMock(
        return_value=StreamedAnswer(
            events=_aevents([SectionReady(imdb_id="tt001", body_md="Great pick.")]),
            answer="Parasite is great.",
            imdb_ids=["tt001"],
        )
    )
    service = ConversationalRecommendationService(recommender)
    media_repo = MagicMock()
    media_repo.get_by_id.return_value = item
    streamed = await service.chat_with_items_stream("recommend a thriller", media_repo)

    async for _event in streamed.events:
        pass

    assert streamed.answer == "Parasite is great."
    assert streamed.items == [item]


async def test_chat_with_items_stream_appends_to_history_after_completion(
    recommender: MagicMock,
) -> None:
    recommender.recommend_stream = AsyncMock(
        return_value=StreamedAnswer(
            events=_aevents([TextDelta(text="answer")]), answer="answer", imdb_ids=[]
        )
    )
    service = ConversationalRecommendationService(recommender)
    streamed = await service.chat_with_items_stream("my question", MagicMock())

    assert service._history == []  # not yet appended mid-stream

    async for _event in streamed.events:
        pass

    assert len(service._history) == 2
    assert service._history[0].content == "my question"
    assert service._history[1].content == "answer"
