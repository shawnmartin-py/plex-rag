from unittest.mock import MagicMock

import pytest
import tenacity
from langchain_core.embeddings import Embeddings
from langchain_google_genai._common import GoogleGenerativeAIError

from app.bootstrap import _aembed_documents_with_retry, _DedupingEmbeddings


@pytest.fixture(autouse=True)
def _no_retry_delay() -> None:
    """Retries are real (asserted via call count) but instant — the
    production `wait_exponential_jitter` would otherwise add real seconds of
    sleep to every test run."""
    _aembed_documents_with_retry.retry.wait = tenacity.wait_none()  # type: ignore[attr-defined]


def _make_inner(**kwargs: object) -> MagicMock:
    inner = MagicMock(spec=Embeddings)
    for key, value in kwargs.items():
        getattr(inner, key).side_effect = value
    return inner


async def test_aembed_documents_with_retry_retries_on_transient_error() -> None:
    inner = _make_inner(aembed_documents=[GoogleGenerativeAIError("429"), [[0.1, 0.2]]])

    result = await _aembed_documents_with_retry(inner, ["some text"])

    assert result == [[0.1, 0.2]]
    assert inner.aembed_documents.await_count == 2


async def test_aembed_documents_with_retry_gives_up_after_three_attempts() -> None:
    inner = _make_inner(aembed_documents=GoogleGenerativeAIError("still failing"))

    with pytest.raises(GoogleGenerativeAIError):
        await _aembed_documents_with_retry(inner, ["some text"])

    assert inner.aembed_documents.await_count == 3


async def test_aembed_documents_with_retry_does_not_retry_other_errors() -> None:
    inner = _make_inner(aembed_documents=ValueError("not a Gemini error"))

    with pytest.raises(ValueError, match="not a Gemini error"):
        await _aembed_documents_with_retry(inner, ["some text"])

    assert inner.aembed_documents.await_count == 1


async def test_deduping_embeddings_embed_one_retries_through_wrapper() -> None:
    inner = _make_inner(aembed_documents=[GoogleGenerativeAIError("429"), [[0.3, 0.4]]])
    embeddings = _DedupingEmbeddings(inner)

    result = await embeddings.aembed_documents(["some text"])

    assert result == [[0.3, 0.4]]
    assert inner.aembed_documents.await_count == 2


async def test_deduping_embeddings_multi_text_batch_retries_through_wrapper() -> None:
    inner = _make_inner(
        aembed_documents=[GoogleGenerativeAIError("429"), [[0.1], [0.2]]]
    )
    embeddings = _DedupingEmbeddings(inner)

    result = await embeddings.aembed_documents(["a", "b"])

    assert result == [[0.1], [0.2]]
    assert inner.aembed_documents.await_count == 2
