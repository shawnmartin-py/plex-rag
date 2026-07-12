from collections.abc import AsyncIterator
from typing import cast

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from app.domain.ports import (
    ConversationTitler,
    QueryRewriter,
    RecommendationGenerator,
    RecommendationResponse,
    SectionReady,
    StreamEvent,
    TextDelta,
)


class GeminiQueryRewriter(QueryRewriter):
    _prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                (
                    "Rewrite the follow-up input as a standalone question or request "
                    "that captures all necessary context from the conversation "
                    "history. Return only the rewritten question, nothing else."
                ),
            ),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    def __init__(self, llm: BaseChatModel) -> None:
        self._chain = self._prompt | llm | StrOutputParser()

    async def rewrite(self, question: str, history: list[BaseMessage]) -> str:
        return await self._chain.ainvoke({"input": question, "chat_history": history})


class GeminiConversationTitler(ConversationTitler):
    _prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                (
                    "Summarize the topic of this movie-recommendation exchange as a "
                    "short, punchy title, in the style of a browser tab title or a "
                    "chat sidebar entry — e.g. 'Heist thrillers with a twist' or "
                    "'Rainy-Sunday comfort films'. 3-6 words. No quotes, no trailing "
                    "punctuation. Return only the title, nothing else."
                ),
            ),
            ("human", "User asked: {question}\n\nAssistant replied: {answer}"),
        ]
    )

    def __init__(self, llm: BaseChatModel) -> None:
        self._chain = self._prompt | llm | StrOutputParser()

    async def title(self, first_question: str, first_answer: str) -> str:
        response = await self._chain.ainvoke(
            {"question": first_question, "answer": first_answer}
        )
        return response.strip()


_RECOMMENDATION_GUIDELINES = (
    "- Recommend only movies from the context above. Never suggest anything outside "
    "it. Each card's imdb_id must be copied exactly from that film's context block "
    "(`[imdb_id: ...]`).\n"
    "- Rank recommendations by how well they match the request — best match first.\n"
    "- For each recommendation, explain specifically why it fits: reference themes, "
    "tone, pacing, director style, or cultural context relevant to the request. Avoid "
    "generic praise.\n"
    "- Write each card's body as 2-3 bullets that each start with a short bold "
    'run-in label. Always lead with "**Why it fits:**"; choose any further labels '
    'to suit the film (e.g. "**Tone & pacing:**", "**The twist:**", "**Content '
    'note:**") — only where they genuinely apply. Keep each bullet to one or two '
    "sentences. Do not restate the film's title or year — the app already displays "
    "them.\n"
    "- If a movie is a weak match, acknowledge it rather than overselling it.\n"
    "- Note content ratings where relevant.\n"
    "- If nothing in the library fits well, leave cards empty and explain why in "
    "closing_note."
)

_SPOILER_FREE_GUIDELINES = (
    "- Recommend only movies from the context above. Never suggest anything outside "
    "it. Each card's imdb_id must be copied exactly from that film's context block "
    "(`[imdb_id: ...]`).\n"
    "- Rank recommendations by how well they match the request — best match first.\n"
    "- For each recommendation, explain why it fits using only genre, tone, pacing, "
    "director style, cast, or cultural context. Avoid generic praise.\n"
    "- Write each card's body as 2-3 bullets that each start with a short bold "
    'run-in label. Always lead with "**Why it fits:**"; choose any further labels '
    'to suit the film (e.g. "**Tone & pacing:**", "**Style:**", "**Content '
    'note:**") — only where they genuinely apply. Keep each bullet to one or two '
    "sentences. Do not restate the film's title or year — the app already displays "
    "them.\n"
    "- IMPORTANT: Do NOT reveal any plot details, story twists, character fates, or "
    "story outcomes. Keep all reasoning completely spoiler-free.\n"
    "- If a movie is a weak match, acknowledge it rather than overselling it.\n"
    "- Note content ratings where relevant.\n"
    "- If nothing in the library fits well, leave cards empty and explain why in "
    "closing_note."
)

_SYSTEM_TEMPLATE = (
    "You are a knowledgeable movie recommendation assistant for a personal Plex "
    "library.\n\n"
    "The following movies have been selected as candidates for the user's request — "
    "some via synopsis similarity, others via broader film knowledge:\n\n"
    "{context}\n\n"
    "Guidelines:\n"
    "{guidelines}"
)


class GeminiRecommendationGenerator(RecommendationGenerator):
    def __init__(self, llm: BaseChatModel, spoiler_free: bool = False) -> None:
        guidelines = (
            _SPOILER_FREE_GUIDELINES if spoiler_free else _RECOMMENDATION_GUIDELINES
        )
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    _SYSTEM_TEMPLATE.format(context="{context}", guidelines=guidelines),
                ),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )
        self._chain = prompt | llm.with_structured_output(RecommendationResponse)

    async def generate(
        self, question: str, context: str, history: list[BaseMessage]
    ) -> RecommendationResponse:
        # with_structured_output's return type is broadened to
        # `dict[str, Any] | BaseModel` to cover non-Pydantic schemas; passing a
        # Pydantic class with the default include_raw=False always yields an
        # instance of that class.
        return cast(
            RecommendationResponse,
            await self._chain.ainvoke(
                {"input": question, "context": context, "chat_history": history}
            ),
        )

    async def stream(
        self, question: str, context: str, history: list[BaseMessage]
    ) -> AsyncIterator[StreamEvent]:
        """`with_structured_output(...).astream()` yields progressively larger,
        fully-validated `RecommendationResponse` instances as Gemini writes the
        answer — not discrete "card N is done" events (confirmed against the
        live API; see docs/plan-structured-recommendation-output.md §3.2-3.3
        and test_example.py). A card at index i is only guaranteed finished
        once `cards` has grown past it: JSON generation is append-only, so
        once the model starts writing card i+1, card i's fields don't change.
        `intro` is flushed once, the first time `cards` becomes non-empty
        (by then its own value can no longer change, for the same reason).
        Whatever's left once the stream ends — the final card, `closing_note`,
        or `intro` alone if the model produced zero cards — is flushed last.
        """
        finalized = 0
        intro_flushed = False
        last: RecommendationResponse | None = None
        async for partial in self._chain.astream(
            {"input": question, "context": context, "chat_history": history}
        ):
            last = cast(RecommendationResponse, partial)
            if not intro_flushed and last.intro and last.cards:
                yield TextDelta(text=last.intro)
                intro_flushed = True
            while finalized < len(last.cards) - 1:
                card = last.cards[finalized]
                yield SectionReady(imdb_id=card.imdb_id, body_md=card.body_md)
                finalized += 1

        if last is None:
            return
        if not intro_flushed and last.intro:
            yield TextDelta(text=last.intro)
        while finalized < len(last.cards):
            card = last.cards[finalized]
            yield SectionReady(imdb_id=card.imdb_id, body_md=card.body_md)
            finalized += 1
        if last.closing_note:
            yield TextDelta(text=last.closing_note)
