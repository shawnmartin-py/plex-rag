from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from app.adapters.generators import GeminiQueryRewriter, GeminiRecommendationGenerator


@pytest.fixture
def rewriter():
    instance = GeminiQueryRewriter(MagicMock())
    instance._chain = MagicMock()
    instance._chain.invoke.return_value = "standalone rewritten question"
    return instance


@pytest.fixture
def generator():
    instance = GeminiRecommendationGenerator(MagicMock())
    instance._chain = MagicMock()
    instance._chain.invoke.return_value = "here are my recommendations"
    return instance


# --- GeminiQueryRewriter ---


def test_rewriter_returns_rewritten_question(rewriter):
    result = rewriter.rewrite("something like the last one", history=[])
    assert result == "standalone rewritten question"


def test_rewriter_passes_question_as_input(rewriter):
    rewriter.rewrite("follow-up question", history=[])
    call_args = rewriter._chain.invoke.call_args[0][0]
    assert call_args["input"] == "follow-up question"


def test_rewriter_passes_history(rewriter):
    history = [HumanMessage(content="first"), AIMessage(content="response")]
    rewriter.rewrite("follow-up", history=history)
    call_args = rewriter._chain.invoke.call_args[0][0]
    assert call_args["chat_history"] == history


def test_rewriter_passes_empty_history(rewriter):
    rewriter.rewrite("standalone question", history=[])
    call_args = rewriter._chain.invoke.call_args[0][0]
    assert call_args["chat_history"] == []


# --- GeminiRecommendationGenerator ---


def test_generator_returns_answer(generator):
    result = generator.generate("recommend a thriller", "some context", history=[])
    assert result == "here are my recommendations"


def test_generator_passes_question(generator):
    generator.generate("recommend a thriller", "context", history=[])
    call_args = generator._chain.invoke.call_args[0][0]
    assert call_args["input"] == "recommend a thriller"


def test_generator_passes_context(generator):
    generator.generate("question", "Title: Parasite\n---\nTitle: Oldboy", history=[])
    call_args = generator._chain.invoke.call_args[0][0]
    assert call_args["context"] == "Title: Parasite\n---\nTitle: Oldboy"


def test_generator_passes_history(generator):
    history = [HumanMessage(content="hi"), AIMessage(content="hello")]
    generator.generate("question", "context", history=history)
    call_args = generator._chain.invoke.call_args[0][0]
    assert call_args["chat_history"] == history
