from __future__ import annotations

import sqlalchemy as sa
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects.postgresql import JSONB, TSVECTOR, UUID

metadata = sa.MetaData()

sessions = sa.Table(
    "sessions",
    metadata,
    sa.Column("id", UUID(as_uuid=True), primary_key=True),
    sa.Column("model_alias", sa.Text, nullable=True),
    sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
)

messages = sa.Table(
    "messages",
    metadata,
    sa.Column("id", sa.BigInteger, primary_key=True, autoincrement=True),
    sa.Column(
        "session_id",
        UUID(as_uuid=True),
        sa.ForeignKey("sessions.id", ondelete="CASCADE"),
        nullable=False,
    ),
    sa.Column("seq", sa.Integer, nullable=False),
    sa.Column("role", sa.Text, nullable=False),
    sa.Column("content", JSONB, nullable=False),
    # Which configured model wrote this turn, and the exact model the provider reported
    # (they differ when a server-side fallback answered).
    sa.Column("model_alias", sa.Text, nullable=True),
    sa.Column("provider_model", sa.Text, nullable=True),
    sa.Column("provider_state", JSONB, nullable=True),
    sa.Column("usage", JSONB, nullable=True),
    sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
    sa.UniqueConstraint("session_id", "seq"),
)

episodes = sa.Table(
    "episodes",
    metadata,
    sa.Column("id", UUID(as_uuid=True), primary_key=True),
    sa.Column(
        "session_id",
        UUID(as_uuid=True),
        sa.ForeignKey("sessions.id", ondelete="SET NULL"),
        nullable=True,
    ),
    sa.Column("occurred_at", sa.DateTime(timezone=True), nullable=False),
    sa.Column("text", sa.Text, nullable=False),
    sa.Column("reply", sa.Text, nullable=True),
    sa.Column("private", sa.Boolean, nullable=False, server_default=sa.false()),
    sa.Column("importance", sa.Float, nullable=False, server_default="0.5"),
    # No fixed dimension: the embedding model is configurable, and embedding_model
    # says which model produced each vector (only same-model vectors are compared).
    sa.Column("embedding", Vector(), nullable=True),
    sa.Column("embedding_model", sa.Text, nullable=True),
    sa.Column(
        "tsv",
        TSVECTOR,
        sa.Computed("to_tsvector('english', text)", persisted=True),
    ),
    sa.Index("ix_episodes_tsv", "tsv", postgresql_using="gin"),
    sa.Index("ix_episodes_occurred_at", "occurred_at"),
)
