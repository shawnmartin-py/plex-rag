"""Builds the LLM RAGAS uses to *score* recommendations — a separate concern
from the LLM the app itself uses to *generate* them (`app/bootstrap.py`),
even though both currently point at the same Gemini model. See
evals/README.md ("Why LangchainLLMWrapper, not ragas's llm_factory") for why
this goes through LangChain rather than ragas's newer native Google adapter.
"""

import os
import warnings
from typing import cast

from langchain_google_genai import ChatGoogleGenerativeAI

# ragas.llms.LangchainLLMWrapper and the classic ragas.metrics.* classes are
# deprecated in favor of ragas.metrics.collections + llm_factory (see
# evals/README.md) but are what actually works reliably with Gemini today —
# the deprecation warning is expected noise, not a signal something's wrong.
# The public `LangchainLLMWrapper` name is a `ragas.utils.DeprecationHelper`
# proxy (not a real class mypy can use as a type or return an accurate type
# from), so `BaseRagasLLM` — the actual base class it constructs an instance
# of at runtime — is imported separately, straight from its stable module,
# purely for typing.
with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    from ragas.llms import LangchainLLMWrapper
from ragas.llms.base import BaseRagasLLM

# Deliberately independent of app/config.py's GOOGLE_API_KEY usage — evals is
# a standalone tool, not a runtime component of the app it's evaluating.
JUDGE_MODEL = os.environ.get("RAGAS_JUDGE_MODEL", "gemini-3.1-flash-lite")


def build_judge_llm() -> BaseRagasLLM:
    """`gemini-3.1-flash-lite` by default, matching the app's own generation
    model — cheap and fast enough for frequent local runs while the eval
    harness itself is still being built out. A judge model that's *stronger*
    than the generator is usually better practice (reduces the risk of a
    model rating its own kind of mistake as acceptable), so once this harness
    is trusted, revisit via `RAGAS_JUDGE_MODEL` rather than hardcoding a
    second model name here."""
    llm = ChatGoogleGenerativeAI(model=JUDGE_MODEL, temperature=0)
    return cast(BaseRagasLLM, LangchainLLMWrapper(llm))
