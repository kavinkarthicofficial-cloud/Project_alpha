"""episodic memory

Revision ID: 0002
Revises: 0001
"""
import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects.postgresql import TSVECTOR, UUID

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "episodes",
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
        # Exact search is fine for one user's memories; add an HNSW index per
        # embedding model (with a fixed dimension) once there are 100k+ episodes.
        sa.Column("embedding", Vector(), nullable=True),
        sa.Column("embedding_model", sa.Text, nullable=True),
        sa.Column("tsv", TSVECTOR, sa.Computed("to_tsvector('english', text)", persisted=True)),
    )
    op.create_index("ix_episodes_tsv", "episodes", ["tsv"], postgresql_using="gin")
    op.create_index("ix_episodes_occurred_at", "episodes", ["occurred_at"])


def downgrade() -> None:
    op.drop_table("episodes")
