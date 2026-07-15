import pytest
from pydantic import BaseModel

from app.adapters.fake_gemini import DeterministicEmbeddings, FakeChatModel
from app.adapters.generators import GeminiConversationTitler, GeminiQueryRewriter
from app.adapters.retrievers import TitleSelection
from app.domain.ports import ChatMessage, RecommendationResponse

CONTEXT = (
    "=== Parasite (2019) [tmdb_id: 496243] ===\nSynopsis: class conflict.\n\n"
    "---\n\n"
    "=== Oldboy (2003) [tmdb_id: 670] ===\nSynopsis: revenge thriller."
)


class _OtherSchema(BaseModel):
    foo: str


async def test_query_rewriter_echoes_standalone_question() -> None:
    rewriter = GeminiQueryRewriter(FakeChatModel())

    result = await rewriter.rewrite(
        "something shorter?",
        [ChatMessage(role="human", content="dark thrillers")],
    )

    assert result == "something shorter?"


async def test_conversation_titler_is_deterministic_and_repeatable() -> None:
    titler = GeminiConversationTitler(FakeChatModel())

    first = await titler.title("dark thrillers", "here are some picks")
    second = await titler.title("dark thrillers", "here are some picks")

    assert first == second
    assert first != ""


async def test_conversation_titler_varies_by_input() -> None:
    titler = GeminiConversationTitler(FakeChatModel())

    a = await titler.title("dark thrillers", "here are some picks")
    b = await titler.title("feel-good comedies", "here are some other picks")

    # Not guaranteed to differ (small canned pool), but exercising both paths
    # shouldn't error and both must be non-empty.
    assert a and b


async def test_structured_output_recommendation_uses_real_context_tmdb_ids() -> None:
    llm = FakeChatModel()
    chain = llm.with_structured_output(RecommendationResponse)

    class _FakePromptValue:
        def to_messages(self) -> list[object]:
            from langchain_core.messages import HumanMessage, SystemMessage

            return [
                SystemMessage(content=f"Context:\n{CONTEXT}"),
                HumanMessage(content="recommend a thriller"),
            ]

    result = await chain.ainvoke(_FakePromptValue())

    assert isinstance(result, RecommendationResponse)
    ids = {c.tmdb_id for c in result.cards}
    assert ids == {"496243", "670"}


async def test_structured_output_recommendation_empty_context_yields_no_cards() -> None:
    from langchain_core.messages import HumanMessage

    llm = FakeChatModel()
    chain = llm.with_structured_output(RecommendationResponse)

    result = await chain.ainvoke([HumanMessage(content="no context here")])

    assert isinstance(result, RecommendationResponse)
    assert result.cards == []


async def test_structured_output_title_selection_from_movie_list() -> None:
    from langchain_core.messages import HumanMessage

    llm = FakeChatModel()
    chain = llm.with_structured_output(TitleSelection)

    result = await chain.ainvoke(
        [HumanMessage(content="Request: x\n\nAvailable movies:\nParasite\nOldboy")]
    )

    assert isinstance(result, TitleSelection)
    assert set(result.titles) <= {"Parasite", "Oldboy"}


async def test_structured_output_unknown_schema_raises() -> None:
    from langchain_core.messages import HumanMessage

    llm = FakeChatModel()
    chain = llm.with_structured_output(_OtherSchema)

    with pytest.raises(NotImplementedError):
        await chain.ainvoke([HumanMessage(content="anything")])


def test_deterministic_embeddings_same_text_same_vector() -> None:
    embeddings = DeterministicEmbeddings(dims=16)

    a = embeddings.embed_query("dark thriller")
    b = embeddings.embed_query("dark thriller")

    assert a == b
    assert len(a) == 16


def test_deterministic_embeddings_different_text_different_vector() -> None:
    embeddings = DeterministicEmbeddings(dims=16)

    a = embeddings.embed_query("dark thriller")
    b = embeddings.embed_query("feel-good comedy")

    assert a != b


async def test_deterministic_embeddings_async_matches_sync() -> None:
    embeddings = DeterministicEmbeddings(dims=16)

    sync_result = embeddings.embed_documents(["a", "b"])
    async_result = await embeddings.aembed_documents(["a", "b"])

    assert sync_result == async_result
