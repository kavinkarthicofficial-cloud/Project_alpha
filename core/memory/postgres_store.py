from __future__ import annotations

import uuid

import sqlalchemy as sa
from sqlalchemy.ext.asyncio import AsyncEngine

from core.memory.episodes import Episode, Hit, SearchFilter, keyword_terms
from core.storage import schema

e = schema.episodes

_COLUMNS = (
    e.c.id,
    e.c.session_id,
    e.c.occurred_at,
    e.c.text,
    e.c.reply,
    e.c.private,
    e.c.importance,
    e.c.embedding_model,
)


def _episode(row: sa.RowMapping) -> Episode:
    return Episode(**{c.name: row[c.name] for c in _COLUMNS})


class PostgresEpisodeStore:
    """Episodes in Postgres: pgvector for meaning, full-text search for exact words."""

    def __init__(self, engine: AsyncEngine) -> None:
        self.engine = engine

    async def add(self, episode: Episode) -> None:
        async with self.engine.begin() as conn:
            await conn.execute(
                sa.insert(e).values(
                    **episode.model_dump(exclude={"embedding"}), embedding=episode.embedding
                )
            )

    def _filtered(self, query: sa.Select, flt: SearchFilter) -> sa.Select:
        if flt.exclude_session is not None:
            query = query.where(
                sa.or_(e.c.session_id.is_(None), e.c.session_id != flt.exclude_session)
            )
        if not flt.include_private:
            query = query.where(e.c.private.is_(False))
        return query

    async def search(
        self,
        query: str,
        embedding: list[float] | None,
        embedding_model: str | None,
        flt: SearchFilter,
        limit: int,
    ) -> list[Hit]:
        terms = keyword_terms(query)
        # OR the terms: a question shares only some words with the memory that answers it.
        tsquery = (
            sa.func.to_tsquery("english", " | ".join(t.replace("'", "") for t in terms))
            if terms
            else None
        )
        rank = sa.func.ts_rank(e.c.tsv, tsquery) if tsquery is not None else sa.literal(0.0)
        if embedding is not None:
            similarity = sa.case(
                (e.c.embedding_model == embedding_model, 1 - e.c.embedding.cosine_distance(embedding)),
                else_=0.0,
            )
        else:
            similarity = sa.literal(0.0)

        base = sa.select(*_COLUMNS, similarity.label("similarity"), rank.label("rank"))
        rows: dict[uuid.UUID, sa.RowMapping] = {}
        async with self.engine.connect() as conn:
            if embedding is not None:
                q = self._filtered(base, flt).where(e.c.embedding_model == embedding_model)
                for r in (await conn.execute(q.order_by(sa.desc("similarity")).limit(limit))).mappings():
                    rows[r["id"]] = r
            if tsquery is not None:
                q = self._filtered(base, flt).where(e.c.tsv.op("@@")(tsquery))
                for r in (await conn.execute(q.order_by(sa.desc("rank")).limit(limit))).mappings():
                    rows.setdefault(r["id"], r)

        top_rank = max((r["rank"] for r in rows.values()), default=0.0) or 1.0
        return [
            Hit(episode=_episode(r), similarity=float(r["similarity"]), keyword=float(r["rank"]) / top_rank)
            for r in rows.values()
        ]

    async def recent(self, limit: int) -> list[Episode]:
        async with self.engine.connect() as conn:
            rows = await conn.execute(
                sa.select(*_COLUMNS).order_by(e.c.occurred_at.desc()).limit(limit)
            )
            return [_episode(r) for r in rows.mappings()]

    async def delete(self, episode_id: uuid.UUID) -> bool:
        async with self.engine.begin() as conn:
            result = await conn.execute(sa.delete(e).where(e.c.id == episode_id))
        return result.rowcount > 0
