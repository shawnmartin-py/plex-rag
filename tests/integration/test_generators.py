from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableSequence

from app.adapters.generators import (
    GeminiConversationTitler,
    GeminiQueryRewriter,
    GeminiRecommendationGenerator,
)


def _system_template(generator: GeminiRecommendationGenerator) -> str:
    chain = cast("RunnableSequence[Any, Any]", generator._chain)
    prompt = cast(Any, chain.steps[0])
    return cast(str, prompt.messages[0].prompt.template)


@pytest.fixture
def rewriter() -> tuple[GeminiQueryRewriter, MagicMock]:
    instance = GeminiQueryRewriter(MagicMock())
    mock_chain = MagicMock()
    instance._chain = mock_chain
    mock_chain.ainvoke = AsyncMock(return_value="standalone rewritten question")
    return instance, mock_chain


@pytest.fixture
def generator() -> tuple[GeminiRecommendationGenerator, MagicMock]:
    instance = GeminiRecommendationGenerator(MagicMock())
    mock_chain = MagicMock()
    instance._chain = mock_chain
    mock_chain.ainvoke = AsyncMock(return_value="here are my recommendations")
    return instance, mock_chain


@pytest.fixture
def spoiler_free_generator() -> tuple[GeminiRecommendationGenerator, MagicMock]:
    instance = GeminiRecommendationGenerator(MagicMock(), spoiler_free=True)
    mock_chain = MagicMock()
    instance._chain = mock_chain
    mock_chain.ainvoke = AsyncMock(return_value="here are my recommendations")
    return instance, mock_chain


@pytest.fixture
def titler() -> tuple[GeminiConversationTitler, MagicMock]:
    instance = GeminiConversationTitler(MagicMock())
    mock_chain = MagicMock()
    instance._chain = mock_chain
    mock_chain.ainvoke = AsyncMock(return_value="  Heist thrillers with a twist  ")
    return instance, mock_chain


# --- GeminiQueryRewriter ---


async def test_rewriter_returns_rewritten_question(
    rewriter: tuple[GeminiQueryRewriter, MagicMock],
) -> None:
    instance, _ = rewriter
    result = await instance.rewrite("something like the last one", history=[])
    assert result == "standalone rewritten question"


async def test_rewriter_passes_question_as_input(
    rewriter: tuple[GeminiQueryRewriter, MagicMock],
) -> None:
    instance, mock_chain = rewriter
    await instance.rewrite("follow-up question", history=[])
    call_args = mock_chain.ainvoke.call_args[0][0]
    assert call_args["input"] == "follow-up question"


async def test_rewriter_passes_history(
    rewriter: tuple[GeminiQueryRewriter, MagicMock],
) -> None:
    instance, mock_chain = rewriter
    history = [HumanMessage(content="first"), AIMessage(content="response")]
    await instance.rewrite("follow-up", history=history)
    call_args = mock_chain.ainvoke.call_args[0][0]
    assert call_args["chat_history"] == history


async def test_rewriter_passes_empty_history(
    rewriter: tuple[GeminiQueryRewriter, MagicMock],
) -> None:
    instance, mock_chain = rewriter
    await instance.rewrite("standalone question", history=[])
    call_args = mock_chain.ainvoke.call_args[0][0]
    assert call_args["chat_history"] == []


# --- GeminiRecommendationGenerator ---


async def test_generator_returns_answer(
    generator: tuple[GeminiRecommendationGenerator, MagicMock],
) -> None:
    instance, _ = generator
    result = await instance.generate("recommend a thriller", "some context", history=[])
    assert result == "here are my recommendations"


async def test_generator_passes_question(
    generator: tuple[GeminiRecommendationGenerator, MagicMock],
) -> None:
    instance, mock_chain = generator
    await instance.generate("recommend a thriller", "context", history=[])
    call_args = mock_chain.ainvoke.call_args[0][0]
    assert call_args["input"] == "recommend a thriller"


async def test_generator_passes_context(
    generator: tuple[GeminiRecommendationGenerator, MagicMock],
) -> None:
    instance, mock_chain = generator
    await instance.generate(
        "question", "Title: Parasite\n---\nTitle: Oldboy", history=[]
    )
    call_args = mock_chain.ainvoke.call_args[0][0]
    assert call_args["context"] == "Title: Parasite\n---\nTitle: Oldboy"


async def test_generator_passes_history(
    generator: tuple[GeminiRecommendationGenerator, MagicMock],
) -> None:
    instance, mock_chain = generator
    history = [HumanMessage(content="hi"), AIMessage(content="hello")]
    await instance.generate("question", "context", history=history)
    call_args = mock_chain.ainvoke.call_args[0][0]
    assert call_args["chat_history"] == history


# --- GeminiRecommendationGenerator spoiler_free flag ---


def test_generator_default_prompt_allows_plot_details() -> None:
    instance = GeminiRecommendationGenerator(MagicMock())
    assert "Do NOT reveal" not in _system_template(instance)


def test_generator_spoiler_free_prompt_prohibits_spoilers() -> None:
    instance = GeminiRecommendationGenerator(MagicMock(), spoiler_free=True)
    assert "Do NOT reveal" in _system_template(instance)


def test_generator_spoiler_free_false_matches_default() -> None:
    default = GeminiRecommendationGenerator(MagicMock())
    explicit_false = GeminiRecommendationGenerator(MagicMock(), spoiler_free=False)
    assert _system_template(default) == _system_template(explicit_false)


def test_generator_spoiler_free_default_differs_from_spoiler_free() -> None:
    default = GeminiRecommendationGenerator(MagicMock())
    sf = GeminiRecommendationGenerator(MagicMock(), spoiler_free=True)
    assert _system_template(default) != _system_template(sf)


async def test_spoiler_free_generator_returns_answer(
    spoiler_free_generator: tuple[GeminiRecommendationGenerator, MagicMock],
) -> None:
    instance, _ = spoiler_free_generator
    result = await instance.generate("recommend a thriller", "some context", history=[])
    assert result == "here are my recommendations"


async def test_spoiler_free_generator_passes_question(
    spoiler_free_generator: tuple[GeminiRecommendationGenerator, MagicMock],
) -> None:
    instance, mock_chain = spoiler_free_generator
    await instance.generate("recommend a thriller", "context", history=[])
    call_args = mock_chain.ainvoke.call_args[0][0]
    assert call_args["input"] == "recommend a thriller"


async def test_spoiler_free_generator_passes_context(
    spoiler_free_generator: tuple[GeminiRecommendationGenerator, MagicMock],
) -> None:
    instance, mock_chain = spoiler_free_generator
    await instance.generate("question", "Title: Parasite", history=[])
    call_args = mock_chain.ainvoke.call_args[0][0]
    assert call_args["context"] == "Title: Parasite"


# --- GeminiConversationTitler ---


async def test_titler_returns_stripped_title(
    titler: tuple[GeminiConversationTitler, MagicMock],
) -> None:
    instance, _ = titler
    result = await instance.title("recommend a heist movie", "Here's Heat (1995)...")
    assert result == "Heist thrillers with a twist"


async def test_titler_passes_question_and_answer_to_chain(
    titler: tuple[GeminiConversationTitler, MagicMock],
) -> None:
    instance, mock_chain = titler
    await instance.title("recommend a heist movie", "Here's Heat (1995)...")
    call_args = mock_chain.ainvoke.call_args[0][0]
    assert call_args["question"] == "recommend a heist movie"
    assert call_args["answer"] == "Here's Heat (1995)..."
