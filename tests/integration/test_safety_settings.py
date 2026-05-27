"""Live tests that verify Gemini safety settings allow mature film content through.

Run with: uv run pytest tests/integration/test_safety_settings.py -m live -v
"""

import pytest
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI, HarmBlockThreshold, HarmCategory

from app.services.enrichment import _PROMPTS

_MODEL = "gemini-3.1-flash-lite"

_SAFETY_OFF = {
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
}

_AMERICAN_HONEY_INPUT = {
    "title": "American Honey",
    "year": 2016,
    "genres": "Adventure, Drama",
    "imdb_rating": 7.1,
    "content_rating": "R",
    "synopsis": (
        "A teenage girl joins a traveling magazine sales crew and embarks on a journey across the American Midwest."
    ),
}


@pytest.fixture(scope="module")
def llm_safety_off():
    return ChatGoogleGenerativeAI(model=_MODEL, temperature=0, safety_settings=_SAFETY_OFF)


@pytest.mark.live
@pytest.mark.parametrize("section", list(_PROMPTS.keys()))
def test_safety_off_produces_content_for_american_honey(llm_safety_off, section):
    chain = _PROMPTS[section] | llm_safety_off | StrOutputParser()
    content = chain.invoke(_AMERICAN_HONEY_INPUT).strip()
    assert content, f"Section '{section}' was blocked even with BLOCK_NONE safety settings"
    assert len(content) > 50, f"Section '{section}' returned suspiciously short content: {content!r}"
