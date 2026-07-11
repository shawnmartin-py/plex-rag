from pathlib import Path

from app.models.conversation import Conversation, ConversationMessage, MessageRole
from app.repositories.conversation_store import ConversationStore


def make_conversation(
    conversation_id: str = "conv-1",
    title: str | None = "Heist thrillers with a twist",
    created_at: str = "2026-01-01T00:00:00+00:00",
    updated_at: str = "2026-01-01T00:00:00+00:00",
    messages: list[ConversationMessage] | None = None,
) -> Conversation:
    if messages is None:
        messages = [
            ConversationMessage(
                role=MessageRole.USER, content="recommend a heist movie"
            ),
            ConversationMessage(
                role=MessageRole.ASSISTANT, content="Here's Heat (1995)..."
            ),
        ]
    return Conversation(
        id=conversation_id,
        title=title,
        created_at=created_at,
        updated_at=updated_at,
        messages=messages,
    )


def make_store(tmp_path: Path) -> ConversationStore:
    return ConversationStore(str(tmp_path / "conversations.duckdb"))


def test_save_then_get_roundtrips_a_conversation(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    store.save(make_conversation())
    result = store.get("conv-1")
    assert result is not None
    assert result.id == "conv-1"
    assert result.title == "Heist thrillers with a twist"
    assert result.messages[0].role is MessageRole.USER
    assert result.messages[0].content == "recommend a heist movie"
    assert result.messages[1].role is MessageRole.ASSISTANT


def test_get_returns_none_for_unknown_id(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    assert store.get("does-not-exist") is None


def test_save_upserts_by_id_rather_than_duplicating(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    store.save(make_conversation(updated_at="2026-01-01T00:00:00+00:00"))
    store.save(
        make_conversation(
            updated_at="2026-01-01T00:05:00+00:00",
            messages=[
                ConversationMessage(role=MessageRole.USER, content="first"),
                ConversationMessage(role=MessageRole.ASSISTANT, content="a"),
                ConversationMessage(role=MessageRole.USER, content="second"),
                ConversationMessage(role=MessageRole.ASSISTANT, content="b"),
            ],
        )
    )
    assert len(store.list_recent()) == 1
    result = store.get("conv-1")
    assert result is not None
    assert len(result.messages) == 4


def test_list_recent_orders_by_updated_at_descending(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    store.save(make_conversation("conv-1", updated_at="2026-01-01T00:00:00+00:00"))
    store.save(make_conversation("conv-2", updated_at="2026-01-02T00:00:00+00:00"))
    store.save(make_conversation("conv-3", updated_at="2026-01-01T12:00:00+00:00"))
    ids = [c.id for c in store.list_recent()]
    assert ids == ["conv-2", "conv-3", "conv-1"]


def test_list_recent_respects_limit_argument(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    for i in range(5):
        store.save(
            make_conversation(f"conv-{i}", updated_at=f"2026-01-0{i + 1}T00:00:00")
        )
    assert len(store.list_recent(limit=2)) == 2


def test_save_prunes_beyond_ten_most_recently_updated_conversations(
    tmp_path: Path,
) -> None:
    store = make_store(tmp_path)
    for i in range(12):
        store.save(
            make_conversation(f"conv-{i}", updated_at=f"2026-01-{i + 1:02d}T00:00:00")
        )
    recent = store.list_recent(limit=100)
    assert len(recent) == 10
    ids = {c.id for c in recent}
    assert "conv-11" in ids  # most recently updated
    assert "conv-0" not in ids  # pruned
    assert "conv-1" not in ids  # pruned


def test_messages_with_items_roundtrip_losslessly(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    item_dict = {
        "imdb_id": "tt0000001",
        "type": "movie",
        "title": "Heat",
        "year": 1995,
        "imdb_rating": 8.2,
        "content_rating": "R",
        "genres": ["Crime", "Drama"],
        "thumb_url": None,
        "video_resolution": None,
        "source_platform": None,
    }
    conversation = make_conversation(
        messages=[
            ConversationMessage(
                role=MessageRole.USER, content="recommend a heist movie"
            ),
            ConversationMessage(
                role=MessageRole.ASSISTANT,
                content="1. **Heat** (1995)",
                items=[item_dict],
            ),
        ]
    )
    store.save(conversation)
    result = store.get("conv-1")
    assert result is not None
    assert result.messages[1].items == [item_dict]
