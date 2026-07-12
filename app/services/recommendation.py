from collections.abc import AsyncIterator
from dataclasses import dataclass, field

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from app.domain.ports import MediaItemLookup, SectionReady, TextDelta
from app.domain.recommender import CoverageReport, MovieRecommender
from app.models.media_item import MediaItem


@dataclass(frozen=True)
class CardReady:
    """The UI-facing counterpart to `SectionReady`, with the imdb_id already
    resolved to a full `MediaItem` (poster, rating, etc.) — or `None` if the
    id couldn't be resolved, in which case the card should be skipped."""

    item: MediaItem | None
    body_md: str


ChatStreamEvent = TextDelta | CardReady


@dataclass
class StreamedChatAnswer:
    """`events` yields one event per completed section, in generation order —
    a `CardReady` the moment a numbered recommendation is finished and its
    `MediaItem` resolved. `answer` and `items` are only meaningful once
    `events` has been fully consumed — at that point the turn has also
    already been appended to history."""

    events: AsyncIterator[ChatStreamEvent]
    answer: str = ""
    items: list[MediaItem] = field(default_factory=list)


class ConversationalRecommendationService:
    def __init__(self, recommender: MovieRecommender) -> None:
        self._recommender = recommender
        self._history: list[BaseMessage] = []

    async def chat(
        self, question: str, verbose: bool = False
    ) -> tuple[str, CoverageReport | None]:
        answer, _, coverage = await self._recommender.recommend(
            question, self._history, verbose=verbose
        )
        self._history.append(HumanMessage(content=question))
        self._history.append(AIMessage(content=answer))
        return answer, coverage

    async def chat_with_items(
        self, question: str, media_repo: MediaItemLookup
    ) -> tuple[str, list[MediaItem]]:
        answer, imdb_ids, _ = await self._recommender.recommend(question, self._history)
        self._history.append(HumanMessage(content=question))
        self._history.append(AIMessage(content=answer))
        items = [media_repo.get_by_id(imdb_id) for imdb_id in imdb_ids]
        return answer, [i for i in items if i is not None]

    async def chat_with_items_stream(
        self, question: str, media_repo: MediaItemLookup
    ) -> StreamedChatAnswer:
        streamed = await self._recommender.recommend_stream(question, self._history)

        async def _events() -> AsyncIterator[ChatStreamEvent]:
            items: list[MediaItem] = []
            async for event in streamed.events:
                if isinstance(event, SectionReady):
                    item = (
                        media_repo.get_by_id(event.imdb_id) if event.imdb_id else None
                    )
                    if item is not None:
                        items.append(item)
                    yield CardReady(item=item, body_md=event.body_md)
                else:
                    yield event
            self._history.append(HumanMessage(content=question))
            self._history.append(AIMessage(content=streamed.answer))
            result.answer = streamed.answer
            result.items = items

        result = StreamedChatAnswer(events=_events())
        return result

    def reset_history(self) -> None:
        self._history = []
