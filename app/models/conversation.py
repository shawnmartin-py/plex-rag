import enum
from dataclasses import dataclass, field


class MessageRole(enum.Enum):
    USER = "user"
    ASSISTANT = "assistant"


@dataclass
class ConversationMessage:
    role: MessageRole
    content: str
    # Serialized MediaItem dicts (nicegui_app/main.py's _item_to_dict/_dict_to_item),
    # only ever populated for role=ASSISTANT — kept as dicts rather than MediaItem so
    # this model stays a pure storage/replay shape, not a rendering dependency.
    items: list[dict[str, object]] = field(default_factory=list)
    # True for a "Surprise me" turn's assistant message: its items are the
    # diversity recommender's plain picks, not paired to numbered sections in
    # the text, so replay must render them differently than a regular chat
    # turn (see render_surprise_results in nicegui_app/components.py).
    is_surprise: bool = False


@dataclass
class Conversation:
    id: str
    title: str | None
    created_at: str
    updated_at: str
    messages: list[ConversationMessage]
