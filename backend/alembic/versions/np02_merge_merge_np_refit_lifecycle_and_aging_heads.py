"""merge np-refit lifecycle and aging heads

Revision ID: np02_merge
Revises: aging01_dataset, np01_lifecycle
Create Date: 2026-11-13 22:17:37.268256
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision = 'np02_merge'
down_revision = ('aging01_dataset', 'np01_lifecycle')
branch_labels = None
depends_on = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
