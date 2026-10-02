from __future__ import annotations

import uuid

import sqlalchemy as sa
from pydantic import TypeAdapter
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from core.llm.types import Block, Message, Usage
from core.storage import schema
from core.storage.base import Session, SessionNotFound

_blocks = TypeAdapter(list[Block])


class PostgresStore:
    def __init__(self, engine: AsyncEngine) -> None:
        self.engine = engine

    @classmethod
    def from_url(cls, url: str) -> PostgresStore:
        return cls(create_async_engine(url, pool_pre_ping=True))

    async def create_session(self, model_alias: str | None = None) -> Session:
        session = Session(model_alias=model_alias)
        async with self.engine.begin() as conn:
            await conn.execute(sa.insert(schema.sessions).values(**session.model_dump()))
        return session

    async def get_session(self, session_id: uuid.UUID) -> Session:
        async with self.engine.connect() as conn:
            row = (
                await conn.execute(sa.select(schema.sessions).where(schema.sessions.c.id == session_id))
            ).mappings().first()
        if row is None:
            raise SessionNotFound(session_id)
        return Session(**row)

    async def set_model(self, session_id: uuid.UUID, model_alias: str | None) -> Session:
        async with self.engine.begin() as conn:
            result = await conn.execute(
                sa.update(schema.sessions)
                .where(schema.sessions.c.id == session_id)
                .values(model_alias=model_alias)
            )
        if result.rowcount == 0:
            raise SessionNotFound(session_id)
        return await self.get_session(session_id)

    async def append(
        self,
        session_id: uuid.UUID,
        message: Message,
        *,
        usage: Usage | None = None,
        provider_model: str | None = None,
    ) -> None:
        m = schema.messages
        async with self.engine.begin() as conn:
            # Serialise appends per session so `seq` has no gaps or duplicates.
            locked = await conn.execute(
                sa.select(schema.sessions.c.id)
                .where(schema.sessions.c.id == session_id)
                .with_for_update()
            )
            if locked.first() is None:
                raise SessionNotFound(session_id)
            next_seq = (
                await conn.execute(
                    sa.select(sa.func.coalesce(sa.func.max(m.c.seq) + 1, 0)).where(
                        m.c.session_id == session_id
                    )
                )
            ).scalar_one()
            await conn.execute(
                sa.insert(m).values(
                    session_id=session_id,
                    seq=next_seq,
                    role=message.role,
                    content=[b.model_dump(mode="json") for b in message.content],
                    model_alias=message.model_alias,
                    provider_model=provider_model,
                    provider_state=message.provider_state,
                    usage=usage.model_dump() if usage else None,
                )
            )

    async def list_messages(self, session_id: uuid.UUID) -> list[Message]:
        await self.get_session(session_id)
        m = schema.messages
        async with self.engine.connect() as conn:
            rows = (
                await conn.execute(
                    sa.select(m.c.role, m.c.content, m.c.model_alias, m.c.provider_state)
                    .where(m.c.session_id == session_id)
                    .order_by(m.c.seq)
                )
            ).mappings().all()
        return [
            Message(
                role=r["role"],
                content=_blocks.validate_python(r["content"]),
                model_alias=r["model_alias"],
                provider_state=r["provider_state"],
            )
            for r in rows
        ]
