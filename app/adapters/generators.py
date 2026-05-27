from langchain_core.messages import BaseMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI

from app.domain.ports import QueryRewriter, RecommendationGenerator


class GeminiQueryRewriter(QueryRewriter):
    _prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                (
                    "Rewrite the follow-up input as a standalone question or request that captures all "
                    "necessary context from the conversation history. Return only the rewritten question, "
                    "nothing else."
                ),
            ),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    def __init__(self, llm: ChatGoogleGenerativeAI) -> None:
        self._chain = self._prompt | llm | StrOutputParser()

    def rewrite(self, question: str, history: list[BaseMessage]) -> str:
        return self._chain.invoke({"input": question, "chat_history": history})


_RECOMMENDATION_GUIDELINES = (
    "- Recommend only movies from the context above. Never suggest anything outside it.\n"
    "- Rank recommendations by how well they match the request — best match first.\n"
    "- For each recommendation, explain specifically why it fits: reference themes, tone, "
    "pacing, director style, or cultural context relevant to the request. Avoid generic praise.\n"
    "- If a movie is a weak match, acknowledge it rather than overselling it.\n"
    "- Note content ratings where relevant.\n"
    "- If nothing in the library fits well, say so directly and briefly explain why."
)

_SPOILER_FREE_GUIDELINES = (
    "- Recommend only movies from the context above. Never suggest anything outside it.\n"
    "- Rank recommendations by how well they match the request — best match first.\n"
    "- For each recommendation, explain why it fits using only genre, tone, pacing, director style, "
    "cast, or cultural context. Avoid generic praise.\n"
    "- IMPORTANT: Do NOT reveal any plot details, story twists, character fates, or story outcomes. "
    "Keep all reasoning completely spoiler-free.\n"
    "- If a movie is a weak match, acknowledge it rather than overselling it.\n"
    "- Note content ratings where relevant.\n"
    "- If nothing in the library fits well, say so directly and briefly explain why."
)

_SYSTEM_TEMPLATE = (
    "You are a knowledgeable movie recommendation assistant for a personal Plex library.\n\n"
    "The following movies have been selected as candidates for the user's request — "
    "some via synopsis similarity, others via broader film knowledge:\n\n"
    "{context}\n\n"
    "Guidelines:\n"
    "{guidelines}"
)


class GeminiRecommendationGenerator(RecommendationGenerator):
    def __init__(self, llm: ChatGoogleGenerativeAI, spoiler_free: bool = False) -> None:
        guidelines = _SPOILER_FREE_GUIDELINES if spoiler_free else _RECOMMENDATION_GUIDELINES
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", _SYSTEM_TEMPLATE.format(context="{context}", guidelines=guidelines)),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )
        self._chain = prompt | llm | StrOutputParser()

    def generate(self, question: str, context: str, history: list[BaseMessage]) -> str:
        return self._chain.invoke({"input": question, "context": context, "chat_history": history})
