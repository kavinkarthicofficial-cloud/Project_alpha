"""Provider-neutral message format.

Everything outside `core/llm/*_provider.py` speaks these types. Each adapter converts
them to and from its provider's wire format, which is what lets one conversation
(and one memory) move between Claude, OpenAI and local models.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Annotated, Any, Literal, Protocol

from pydantic import BaseModel, Field


class TextBlock(BaseModel):
    type: Literal["text"] = "text"
    text: str


class ToolCallBlock(BaseModel):
    type: Literal["tool_call"] = "tool_call"
    id: str = Field(default_factory=lambda: f"call_{uuid.uuid4().hex[:24]}")
    name: str
    arguments: dict[str, Any]


class ToolResultBlock(BaseModel):
    type: Literal["tool_result"] = "tool_result"
    tool_call_id: str
    tool_name: str
    content: str
    is_error: bool = False


Block = Annotated[TextBlock | ToolCallBlock | ToolResultBlock, Field(discriminator="type")]

Role = Literal["system", "user", "assistant"]


class Message(BaseModel):
    """One entry in a conversation.

    - A `system` message at index 0 is the frozen session prompt. Later `system`
      messages are mid-conversation context (e.g. retrieved memory) and are never
      edited or deleted once appended, so provider prompt caches stay valid.
    - Tool results travel in a `user` message, as ToolResultBlocks.
    - `model_alias` records which configured model produced an assistant turn.
    - `provider_state` keeps the provider's native content (e.g. Claude thinking
      blocks) so the same provider can be replayed byte-for-byte.
    """

    role: Role
    content: list[Block]
    model_alias: str | None = None
    provider_state: dict[str, Any] | None = None

    @classmethod
    def text(cls, role: Role, text: str) -> Message:
        return cls(role=role, content=[TextBlock(text=text)])

    def text_content(self) -> str:
        return "".join(b.text for b in self.content if isinstance(b, TextBlock))

    def tool_calls(self) -> list[ToolCallBlock]:
        return [b for b in self.content if isinstance(b, ToolCallBlock)]


class ToolSpec(BaseModel):
    name: str
    description: str
    input_schema: dict[str, Any]


class Usage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0


StopReason = Literal["end", "tool_use", "max_tokens", "refusal", "other"]


@dataclass
class TextDelta:
    text: str


@dataclass
class Completed:
    message: Message
    stop_reason: StopReason
    usage: Usage = field(default_factory=Usage)
    provider_model: str = ""


StreamEvent = TextDelta | Completed


class ModelProfile(BaseModel):
    """A configured model, as declared in config/models.yaml."""

    alias: str
    provider: str
    model: str
    context_window: int
    max_output_tokens: int = 8192
    supports_tools: bool = True
    supports_mid_system: bool = False
    can_write_memory: bool = False
    local: bool = False
    options: dict[str, Any] = Field(default_factory=dict)


class ProviderError(Exception):
    """A request failed in a way the caller should surface, not retry elsewhere."""


class ProviderUnavailable(ProviderError):
    """The provider can't serve this request right now (no credentials, network,
    rate limit, outage). The router may fall back to another model."""


class InvalidToolInput(ProviderError):
    """The model produced tool-call JSON that could not be parsed at all."""


class LLMProvider(Protocol):
    name: str

    def stream(
        self,
        profile: ModelProfile,
        messages: list[Message],
        tools: list[ToolSpec] | None = None,
    ) -> AsyncIterator[StreamEvent]: ...
