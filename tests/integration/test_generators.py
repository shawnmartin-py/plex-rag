from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from app.adapters.generators import GeminiQueryRewriter, GeminiRecommendationGenerator


def _system_template(generator: GeminiRecommendationGenerator) -> str:
    return generator._chain.steps[0].messages[0].prompt.template


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


@pytest.fixture
def spoiler_free_generator():
    instance = GeminiRecommendationGenerator(MagicMock(), spoiler_free=True)
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


# --- GeminiRecommendationGenerator spoiler_free flag ---


def test_generator_default_prompt_allows_plot_details():
    instance = GeminiRecommendationGenerator(MagicMock())
    assert "Do NOT reveal" not in _system_template(instance)


def test_generator_spoiler_free_prompt_prohibits_spoilers():
    instance = GeminiRecommendationGenerator(MagicMock(), spoiler_free=True)
    assert "Do NOT reveal" in _system_template(instance)


def test_generator_spoiler_free_false_matches_default():
    default = GeminiRecommendationGenerator(MagicMock())
    explicit_false = GeminiRecommendationGenerator(MagicMock(), spoiler_free=False)
    assert _system_template(default) == _system_template(explicit_false)


def test_generator_spoiler_free_default_differs_from_spoiler_free():
    default = GeminiRecommendationGenerator(MagicMock())
    sf = GeminiRecommendationGenerator(MagicMock(), spoiler_free=True)
    assert _system_template(default) != _system_template(sf)


def test_spoiler_free_generator_returns_answer(spoiler_free_generator):
    result = spoiler_free_generator.generate("recommend a thriller", "some context", history=[])
    assert result == "here are my recommendations"


def test_spoiler_free_generator_passes_question(spoiler_free_generator):
    spoiler_free_generator.generate("recommend a thriller", "context", history=[])
    call_args = spoiler_free_generator._chain.invoke.call_args[0][0]
    assert call_args["input"] == "recommend a thriller"


def test_spoiler_free_generator_passes_context(spoiler_free_generator):
    spoiler_free_generator.generate("question", "Title: Parasite", history=[])
    call_args = spoiler_free_generator._chain.invoke.call_args[0][0]
    assert call_args["context"] == "Title: Parasite"
