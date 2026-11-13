"""model lifecycle: wider model status, model_lifecycle, model_promotion_events

Revision ID: np01_lifecycle
Revises: 0001
Create Date: 2026-11-13 22:30:00
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = 'np01_lifecycle'
down_revision = '0001'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # "previous_production" (19 chars) does not fit the original String(16).
    op.alter_column('model_registry', 'status', type_=sa.String(length=24), existing_type=sa.String(length=16),
                    existing_nullable=False)
    op.create_table('model_lifecycle',
    sa.Column('model_version', sa.String(length=64), nullable=False),
    sa.Column('pitcher_id', sa.Integer(), nullable=False),
    sa.Column('training_scope', sa.String(length=32), nullable=False),
    sa.Column('data_cutoff', sa.Date(), nullable=False),
    sa.Column('deployment_cutoff', sa.Date(), nullable=False),
    sa.Column('n_train_samples', sa.Integer(), nullable=False),
    sa.Column('n_train_games', sa.Integer(), nullable=False),
    sa.Column('selected_params', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
    sa.Column('dev_metrics', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
    sa.Column('dev_evaluation_run_id', sa.BigInteger(), nullable=True),
    sa.Column('gate_results', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    sa.Column('retrain_policy', sa.String(length=24), nullable=True),
    sa.Column('trigger', sa.Text(), nullable=True),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.ForeignKeyConstraint(['model_version'], ['model_registry.model_version'], ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('model_version')
    )
    op.create_index(op.f('ix_model_lifecycle_pitcher_id'), 'model_lifecycle', ['pitcher_id'], unique=False)
    op.create_table('model_promotion_events',
    sa.Column('event_id', sa.BigInteger(), autoincrement=True, nullable=False),
    sa.Column('pitcher_id', sa.Integer(), nullable=False),
    sa.Column('model_version', sa.String(length=64), nullable=False),
    sa.Column('action', sa.String(length=24), nullable=False),
    sa.Column('from_status', sa.String(length=24), nullable=True),
    sa.Column('to_status', sa.String(length=24), nullable=True),
    sa.Column('replaced_version', sa.String(length=64), nullable=True),
    sa.Column('gate_results', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    sa.Column('source', sa.Text(), nullable=True),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.ForeignKeyConstraint(['model_version'], ['model_registry.model_version'], ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('event_id')
    )
    op.create_index(op.f('ix_model_promotion_events_pitcher_id'), 'model_promotion_events', ['pitcher_id'],
                    unique=False)
    op.create_index(op.f('ix_model_promotion_events_model_version'), 'model_promotion_events', ['model_version'],
                    unique=False)
    op.create_index('ix_pitch_predictions_context_model', 'pitch_predictions', ['context', 'model_version'],
                    unique=False)


def downgrade() -> None:
    op.drop_index('ix_pitch_predictions_context_model', table_name='pitch_predictions')
    op.drop_index(op.f('ix_model_promotion_events_model_version'), table_name='model_promotion_events')
    op.drop_index(op.f('ix_model_promotion_events_pitcher_id'), table_name='model_promotion_events')
    op.drop_table('model_promotion_events')
    op.drop_index(op.f('ix_model_lifecycle_pitcher_id'), table_name='model_lifecycle')
    op.drop_table('model_lifecycle')
    op.execute("UPDATE model_registry SET status='candidate' WHERE status='previous_production'")
    op.alter_column('model_registry', 'status', type_=sa.String(length=16), existing_type=sa.String(length=24),
                    existing_nullable=False)
