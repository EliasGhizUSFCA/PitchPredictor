"""aging: pitcher_bio.source_url VARCHAR(512) -> TEXT (bulk people URLs carry 100 ids and exceed 512 chars; they
were truncated, which made the stored provenance URL inexact)

Revision ID: aging02_bio_url
Revises: np02_merge
Create Date: 2026-11-13 23:40:00
"""
from alembic import op
import sqlalchemy as sa

revision = 'aging02_bio_url'
down_revision = 'np02_merge'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.alter_column('pitcher_bio', 'source_url', type_=sa.Text(), existing_type=sa.String(length=512),
                    existing_nullable=True)


def downgrade() -> None:
    op.execute("UPDATE pitcher_bio SET source_url = left(source_url, 512)")
    op.alter_column('pitcher_bio', 'source_url', type_=sa.String(length=512), existing_type=sa.Text(),
                    existing_nullable=True)
