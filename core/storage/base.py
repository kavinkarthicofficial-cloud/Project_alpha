from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Protocol

from pydantic import BaseModel, Field

from core.llm.types import Message, Usage


class Session(BaseModel):
    id: uuid.UUID = Field(default_factory=uuid.uuid4)
    # The model this session uses unless a single message asks for another one.
    model_alias: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class SessionNotFound(KeyError):
    pass


class ConversationStore(Protocol):
    async def create_session(self, model_alias: str | None = None) -> Session: ...

    async def get_session(self, session_id: uuid.UUID) -> Session: ...

    async def set_model(self, session_id: uuid.UUID, model_alias: str | None) -> Session: ...

    async def append(
        self,
        session_id: uuid.UUID,
        message: Message,
        *,
        usage: Usage | None = None,
        provider_model: str | None = None,
    ) -> None: ...

    async def list_messages(self, session_id: uuid.UUID) -> list[Message]: ...


class InMemoryStore:
    """Process-local store for development and tests."""

    def __init__(self) -> None:
        self._sessions: dict[uuid.UUID, Session] = {}
        self._messages: dict[uuid.UUID, list[Message]] = {}

    async def create_session(self, model_alias: str | None = None) -> Session:
        session = Session(model_alias=model_alias)
        self._sessions[session.id] = session
        self._messages[session.id] = []
        return session

    async def get_session(self, session_id: uuid.UUID) -> Session:
        try:
            return self._sessions[session_id]
        except KeyError:
            raise SessionNotFound(session_id) from None

    async def set_model(self, session_id: uuid.UUID, model_alias: str | None) -> Session:
        session = await self.get_session(session_id)
        session.model_alias = model_alias
        return session

    async def append(
        self,
        session_id: uuid.UUID,
        message: Message,
        *,
        usage: Usage | None = None,
        provider_model: str | None = None,
    ) -> None:
        await self.get_session(session_id)
        self._messages[session_id].append(message.model_copy(deep=True))

    async def list_messages(self, session_id: uuid.UUID) -> list[Message]:
        await self.get_session(session_id)
        return [m.model_copy(deep=True) for m in self._messages[session_id]]
