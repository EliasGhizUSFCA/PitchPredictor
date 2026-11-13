"""aging dataset: pitcher_bio, pitcher_season_stats, pitcher_season_pitch_stats, aging_observations, aging_cohorts,
aging_etl_runs, and the pitch_ages view (exact age of each pitch from pitcher_bio.birth_date)

Revision ID: aging01_dataset
Revises: 0001
Create Date: 2026-11-13 23:00:00
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = 'aging01_dataset'
down_revision = '0001'
branch_labels = None
depends_on = None

PITCH_AGES_VIEW = """
CREATE VIEW pitch_ages AS
SELECT p.pitch_uid, p.pitcher_id, p.game_pk, p.game_date, p.season, b.birth_date,
       (p.game_date - b.birth_date) / 365.2425 AS age_exact_years,
       date_part('year', age(p.game_date, b.birth_date))::int AS age_completed_years
FROM pitches p JOIN pitcher_bio b ON b.pitcher_id = p.pitcher_id
WHERE b.birth_date IS NOT NULL
"""


def upgrade() -> None:
    op.create_table(
        'aging_etl_runs',
        sa.Column('etl_run_id', sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column('status', sa.String(length=16), nullable=False),
        sa.Column('params', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column('sources', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('coverage', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('validation', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('units', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('started_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('finished_at', sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint('etl_run_id', name='pk_aging_etl_runs'),
    )
    op.create_table(
        'pitcher_bio',
        sa.Column('pitcher_id', sa.Integer(), nullable=False),
        sa.Column('full_name', sa.String(length=120), nullable=True),
        sa.Column('birth_date', sa.Date(), nullable=True),
        sa.Column('throws', sa.String(length=1), nullable=True),
        sa.Column('primary_position', sa.String(length=8), nullable=True),
        sa.Column('birth_country', sa.String(length=64), nullable=True),
        sa.Column('mlb_debut_date', sa.Date(), nullable=True),
        sa.Column('source', sa.String(length=64), nullable=False),
        sa.Column('source_url', sa.String(length=512), nullable=True),
        sa.Column('fetched_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.PrimaryKeyConstraint('pitcher_id', name='pk_pitcher_bio'),
        sa.CheckConstraint("throws IS NULL OR throws IN ('L','R','S')", name='ck_bio_throws'),
    )
    op.create_table(
        'pitcher_season_stats',
        sa.Column('pitcher_id', sa.Integer(), nullable=False),
        sa.Column('season', sa.SmallInteger(), nullable=False),
        sa.Column('player_name', sa.String(length=120), nullable=True),
        sa.Column('pitch_hand', sa.String(length=1), nullable=True),
        sa.Column('age_june30', sa.SmallInteger(), nullable=True),
        sa.Column('age_june30_exact', sa.Float(), nullable=True),
        sa.Column('savant_player_age', sa.SmallInteger(), nullable=True),
        sa.Column('games', sa.SmallInteger(), nullable=True),
        sa.Column('games_started', sa.SmallInteger(), nullable=True),
        sa.Column('outs', sa.Integer(), nullable=True),
        sa.Column('ip', sa.Float(), nullable=True),
        sa.Column('batters_faced', sa.Integer(), nullable=True),
        sa.Column('pitch_count', sa.Integer(), nullable=True),
        sa.Column('k_pct', sa.Float(), nullable=True),
        sa.Column('bb_pct', sa.Float(), nullable=True),
        sa.Column('in_zone_pct', sa.Float(), nullable=True),
        sa.Column('oz_swing_pct', sa.Float(), nullable=True),
        sa.Column('z_swing_pct', sa.Float(), nullable=True),
        sa.Column('swing_pct', sa.Float(), nullable=True),
        sa.Column('whiff_pct', sa.Float(), nullable=True),
        sa.Column('edge_pct', sa.Float(), nullable=True),
        sa.Column('f_strike_pct', sa.Float(), nullable=True),
        sa.Column('meatball_pct', sa.Float(), nullable=True),
        sa.Column('iz_contact_pct', sa.Float(), nullable=True),
        sa.Column('oz_contact_pct', sa.Float(), nullable=True),
        sa.Column('fastball_avg_speed_mph', sa.Float(), nullable=True),
        sa.Column('fastball_avg_spin_rpm', sa.Float(), nullable=True),
        sa.Column('breaking_avg_speed_mph', sa.Float(), nullable=True),
        sa.Column('breaking_avg_spin_rpm', sa.Float(), nullable=True),
        sa.Column('offspeed_avg_speed_mph', sa.Float(), nullable=True),
        sa.Column('offspeed_avg_spin_rpm', sa.Float(), nullable=True),
        sa.Column('extension_ft', sa.Float(), nullable=True),
        sa.Column('release_pos_x_ft', sa.Float(), nullable=True),
        sa.Column('release_pos_z_ft', sa.Float(), nullable=True),
        sa.Column('arm_angle_deg', sa.Float(), nullable=True),
        sa.Column('xwoba', sa.Float(), nullable=True),
        sa.Column('era', sa.Float(), nullable=True),
        sa.Column('measurement_era', sa.String(length=10), nullable=False),
        sa.Column('low_sample', sa.Boolean(), nullable=False),
        sa.Column('quality_flags', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column('field_sources', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column('source_urls', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column('fetched_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('etl_run_id', sa.BigInteger(), nullable=True),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.PrimaryKeyConstraint('pitcher_id', 'season', name='pk_pitcher_season_stats'),
        sa.CheckConstraint("measurement_era IN ('trackman','hawkeye')", name='ck_pss_era'),
    )
    op.create_index('ix_pss_season', 'pitcher_season_stats', ['season'])
    op.create_table(
        'pitcher_season_pitch_stats',
        sa.Column('pitcher_id', sa.Integer(), nullable=False),
        sa.Column('season', sa.SmallInteger(), nullable=False),
        sa.Column('pitch_type', sa.String(length=4), nullable=False),
        sa.Column('pitch_family', sa.String(length=16), nullable=False),
        sa.Column('pitch_group', sa.String(length=10), nullable=False),
        sa.Column('pitch_name', sa.String(length=40), nullable=True),
        sa.Column('pitch_hand', sa.String(length=1), nullable=True),
        sa.Column('pitches', sa.Integer(), nullable=True),
        sa.Column('total_pitches', sa.Integer(), nullable=True),
        sa.Column('usage_pct', sa.Float(), nullable=True),
        sa.Column('avg_speed_mph', sa.Float(), nullable=True),
        sa.Column('avg_spin_rpm', sa.Float(), nullable=True),
        sa.Column('break_z_in', sa.Float(), nullable=True),
        sa.Column('ivb_in', sa.Float(), nullable=True),
        sa.Column('break_x_in', sa.Float(), nullable=True),
        sa.Column('break_z_vs_league_in', sa.Float(), nullable=True),
        sa.Column('break_x_vs_league_in', sa.Float(), nullable=True),
        sa.Column('extension_ft', sa.Float(), nullable=True),
        sa.Column('release_pos_x_ft', sa.Float(), nullable=True),
        sa.Column('release_pos_z_ft', sa.Float(), nullable=True),
        sa.Column('arm_angle_deg', sa.Float(), nullable=True),
        sa.Column('swings', sa.Integer(), nullable=True),
        sa.Column('whiffs', sa.Integer(), nullable=True),
        sa.Column('whiff_pct', sa.Float(), nullable=True),
        sa.Column('k_pct', sa.Float(), nullable=True),
        sa.Column('put_away_pct', sa.Float(), nullable=True),
        sa.Column('run_value', sa.Float(), nullable=True),
        sa.Column('run_value_per_100', sa.Float(), nullable=True),
        sa.Column('batters_faced', sa.Integer(), nullable=True),
        sa.Column('woba', sa.Float(), nullable=True),
        sa.Column('xwoba', sa.Float(), nullable=True),
        sa.Column('hard_hit_pct', sa.Float(), nullable=True),
        sa.Column('measurement_era', sa.String(length=10), nullable=False),
        sa.Column('low_sample', sa.Boolean(), nullable=False),
        sa.Column('quality_flags', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column('values_by_source', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column('field_sources', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column('source_urls', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column('fetched_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('etl_run_id', sa.BigInteger(), nullable=True),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.PrimaryKeyConstraint('pitcher_id', 'season', 'pitch_type', name='pk_pitcher_season_pitch_stats'),
        sa.CheckConstraint("measurement_era IN ('trackman','hawkeye')", name='ck_psps_era'),
    )
    op.create_index('ix_psps_season_family', 'pitcher_season_pitch_stats', ['season', 'pitch_family'])
    op.create_table(
        'aging_observations',
        sa.Column('obs_id', sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column('pitcher_id', sa.Integer(), nullable=False),
        sa.Column('pitch_family', sa.String(length=24), nullable=False),
        sa.Column('metric', sa.String(length=32), nullable=False),
        sa.Column('unit', sa.String(length=16), nullable=True),
        sa.Column('season_from', sa.SmallInteger(), nullable=False),
        sa.Column('season_to', sa.SmallInteger(), nullable=False),
        sa.Column('birth_year', sa.SmallInteger(), nullable=True),
        sa.Column('age_from', sa.Float(), nullable=True),
        sa.Column('age_from_int', sa.SmallInteger(), nullable=True),
        sa.Column('age_to', sa.Float(), nullable=True),
        sa.Column('value_from', sa.Float(), nullable=True),
        sa.Column('value_to', sa.Float(), nullable=True),
        sa.Column('delta', sa.Float(), nullable=True),
        sa.Column('n_from', sa.Integer(), nullable=True),
        sa.Column('n_to', sa.Integer(), nullable=True),
        sa.Column('workload_pitches_from', sa.Integer(), nullable=True),
        sa.Column('workload_pitches_to', sa.Integer(), nullable=True),
        sa.Column('ip_from', sa.Float(), nullable=True),
        sa.Column('ip_to', sa.Float(), nullable=True),
        sa.Column('survived', sa.Boolean(), nullable=False),
        sa.Column('unit_present_next', sa.Boolean(), nullable=False),
        sa.Column('both_qualified', sa.Boolean(), nullable=False),
        sa.Column('returned_later', sa.Boolean(), nullable=False),
        sa.Column('next_qualifying_season', sa.SmallInteger(), nullable=True),
        sa.Column('era_from', sa.String(length=10), nullable=False),
        sa.Column('era_to', sa.String(length=10), nullable=False),
        sa.Column('era_change', sa.Boolean(), nullable=False),
        sa.Column('short_season', sa.Boolean(), nullable=False),
        sa.Column('reclass_flag', sa.Boolean(), nullable=False),
        sa.Column('codes_from', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('codes_to', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('min_pitches_threshold', sa.Integer(), nullable=False),
        sa.Column('min_season_pitches_threshold', sa.Integer(), nullable=False),
        sa.Column('etl_run_id', sa.BigInteger(), nullable=True),
        sa.Column('computed_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.PrimaryKeyConstraint('obs_id', name='pk_aging_observations'),
        sa.UniqueConstraint('pitcher_id', 'pitch_family', 'metric', 'season_from', name='uq_aging_obs'),
    )
    op.create_index('ix_aging_obs_pitcher', 'aging_observations', ['pitcher_id'])
    op.create_index('ix_aging_obs_metric', 'aging_observations', ['pitch_family', 'metric', 'season_from'])
    op.create_table(
        'aging_cohorts',
        sa.Column('cohort_id', sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column('cohort_type', sa.String(length=16), nullable=False),
        sa.Column('cohort_key', sa.String(length=16), nullable=False),
        sa.Column('scope', sa.String(length=8), nullable=False),
        sa.Column('n_pitchers', sa.Integer(), nullable=False),
        sa.Column('n_survived', sa.Integer(), nullable=False),
        sa.Column('n_dropped', sa.Integer(), nullable=False),
        sa.Column('n_returned_later', sa.Integer(), nullable=False),
        sa.Column('n_era_change', sa.Integer(), nullable=False),
        sa.Column('survival_rate', sa.Float(), nullable=True),
        sa.Column('mean_age', sa.Float(), nullable=True),
        sa.Column('metric_summaries', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column('etl_run_id', sa.BigInteger(), nullable=True),
        sa.Column('computed_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.PrimaryKeyConstraint('cohort_id', name='pk_aging_cohorts'),
        sa.UniqueConstraint('cohort_type', 'cohort_key', 'scope', name='uq_aging_cohort'),
    )
    op.execute(PITCH_AGES_VIEW)


def downgrade() -> None:
    op.execute("DROP VIEW IF EXISTS pitch_ages")
    for t in ('aging_cohorts', 'aging_observations', 'pitcher_season_pitch_stats', 'pitcher_season_stats',
              'pitcher_bio', 'aging_etl_runs'):
        op.drop_table(t)
