import json
from pathlib import Path

import duckdb

from app.models.conversation import Conversation, ConversationMessage, MessageRole

_MAX_RECENT = 10

_SCHEMA = """
CREATE TABLE IF NOT EXISTS conversations (
    id VARCHAR PRIMARY KEY,
    title VARCHAR,
    created_at VARCHAR NOT NULL,
    updated_at VARCHAR NOT NULL,
    messages_json VARCHAR NOT NULL
)
"""


def _row_to_conversation(row: tuple[str, str | None, str, str, str]) -> Conversation:
    id_, title, created_at, updated_at, messages_json = row
    raw_messages = json.loads(messages_json)
    return Conversation(
        id=id_,
        title=title,
        created_at=created_at,
        updated_at=updated_at,
        messages=[
            ConversationMessage(
                role=MessageRole(m["role"]), content=m["content"], items=m["items"]
            )
            for m in raw_messages
        ],
    )


class ConversationStore:
    """DuckDB-backed persistence for the web UI's Recent-conversations sidebar
    list — the first read-write repository in plex-rag (everything else under
    app/repositories/ is a read-only Qdrant view). `messages` is stored as a
    plain VARCHAR of `json.dumps` output rather than DuckDB's native JSON
    column type, which depends on the `json` extension and can attempt a
    network fetch to autoload it on first use — unacceptable for an
    offline-capable desktop app.

    Opens a short-lived connection per call rather than holding one open for
    the process lifetime: DuckDB connections aren't documented as safe for
    concurrent use from multiple threads, and NiceGUI runs blocking calls
    (`run.io_bound`) on a shared worker-thread pool, so a single long-lived
    connection could be entered concurrently by two different tabs' turns.
    """

    def __init__(self, db_path: str) -> None:
        self._db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        with duckdb.connect(self._db_path) as conn:
            conn.execute(_SCHEMA)

    def save(self, conversation: Conversation) -> None:
        """Upsert by id, then prune to the 10 most-recently-updated rows so the
        file stays bounded — matches "kept" in the retention requirement
        rather than just limiting what's read."""
        messages_json = json.dumps(
            [
                {"role": m.role.value, "content": m.content, "items": m.items}
                for m in conversation.messages
            ]
        )
        with duckdb.connect(self._db_path) as conn:
            conn.execute(
                """
                INSERT INTO conversations
                    (id, title, created_at, updated_at, messages_json)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT (id) DO UPDATE SET
                    title = excluded.title,
                    updated_at = excluded.updated_at,
                    messages_json = excluded.messages_json
                """,
                [
                    conversation.id,
                    conversation.title,
                    conversation.created_at,
                    conversation.updated_at,
                    messages_json,
                ],
            )
            conn.execute(
                """
                DELETE FROM conversations WHERE id NOT IN (
                    SELECT id FROM conversations
                    ORDER BY updated_at DESC LIMIT ?
                )
                """,
                [_MAX_RECENT],
            )

    def list_recent(self, limit: int = _MAX_RECENT) -> list[Conversation]:
        """Most-recently-updated first, so a conversation you keep chatting in
        stays near the top of the sidebar."""
        with duckdb.connect(self._db_path) as conn:
            rows = conn.execute(
                """
                SELECT id, title, created_at, updated_at, messages_json
                FROM conversations ORDER BY updated_at DESC LIMIT ?
                """,
                [limit],
            ).fetchall()
        return [_row_to_conversation(row) for row in rows]

    def get(self, conversation_id: str) -> Conversation | None:
        with duckdb.connect(self._db_path) as conn:
            row = conn.execute(
                """
                SELECT id, title, created_at, updated_at, messages_json
                FROM conversations WHERE id = ?
                """,
                [conversation_id],
            ).fetchone()
        return _row_to_conversation(row) if row else None
