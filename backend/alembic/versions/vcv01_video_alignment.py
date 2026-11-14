"""video-cv: automatic alignment runs + detector validation reports

Revision ID: vcv01_video_alignment
Revises: aging02_bio_url
Create Date: 2026-11-14 12:00:00
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = 'vcv01_video_alignment'
down_revision = 'aging02_bio_url'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        'video_alignment_runs',
        sa.Column('alignment_run_id', sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column('video_asset_id', sa.BigInteger(), nullable=False),
        sa.Column('detector_version', sa.String(length=32), nullable=False),
        sa.Column('params', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column('status', sa.String(length=16), nullable=False),
        sa.Column('applied', sa.Boolean(), nullable=False),
        sa.Column('alignment_confidence', sa.Float(), nullable=True),
        sa.Column('clip_start_time', sa.Float(), nullable=True),
        sa.Column('onset_time', sa.Float(), nullable=True),
        sa.Column('release_time', sa.Float(), nullable=True),
        sa.Column('pre_pitch_pause_time', sa.Float(), nullable=True),
        sa.Column('pitch_end_time', sa.Float(), nullable=True),
        sa.Column('clip_duration', sa.Float(), nullable=True),
        sa.Column('fps', sa.Float(), nullable=True),
        sa.Column('n_frames', sa.Integer(), nullable=True),
        sa.Column('file_fingerprint', sa.String(length=64), nullable=True),
        sa.Column('result', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column('error', sa.Text(), nullable=True),
        sa.Column('runtime_ms', sa.Float(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.CheckConstraint("status IN ('auto_aligned','unaligned')", name='ck_valign_status'),
        sa.ForeignKeyConstraint(['video_asset_id'], ['video_assets.video_asset_id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('alignment_run_id'),
    )
    op.create_index('ix_video_alignment_runs_video_asset_id', 'video_alignment_runs', ['video_asset_id'])
    op.create_table(
        'video_alignment_validations',
        sa.Column('validation_id', sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column('detector_version', sa.String(length=32), nullable=False),
        sa.Column('role', sa.String(length=16), nullable=False),
        sa.Column('data_kind', sa.String(length=16), nullable=False),
        sa.Column('generator', sa.String(length=96), nullable=False),
        sa.Column('n_clips', sa.Integer(), nullable=False),
        sa.Column('seeds', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('params', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column('summary', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column('per_clip', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column('mlflow_run_id', sa.String(length=64), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.CheckConstraint("data_kind IN ('synthetic','licensed_real')", name='ck_valid_data_kind'),
        sa.CheckConstraint("role IN ('development','evaluation')", name='ck_valid_role'),
        sa.PrimaryKeyConstraint('validation_id'),
    )


def downgrade() -> None:
    op.drop_table('video_alignment_validations')
    op.drop_index('ix_video_alignment_runs_video_asset_id', table_name='video_alignment_runs')
    op.drop_table('video_alignment_runs')
