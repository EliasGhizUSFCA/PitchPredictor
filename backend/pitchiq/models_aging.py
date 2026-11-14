"""Aging dataset tables (MLB-wide pitcher-season longitudinal data). Owned by pitchiq/aging/etl.py.

Column names carry units (_mph, _rpm, _in, _ft, _pct, _deg); `pitchiq.aging.etl.UNITS` documents each one and is
also stored on every `aging_etl_runs` row. Percentages are 0-100 as Baseball Savant reports them.
"""
from __future__ import annotations

from datetime import date, datetime

from sqlalchemy import (BigInteger, Boolean, CheckConstraint, Date, DateTime, Float, Index, Integer, SmallInteger,
                        String, Text, UniqueConstraint, func)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from .models import Base


def _now():
    return mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)


class AgingEtlRun(Base):
    __tablename__ = "aging_etl_runs"
    etl_run_id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    status: Mapped[str] = mapped_column(String(16), default="running")  # running | succeeded | failed
    params: Mapped[dict] = mapped_column(JSONB)
    sources: Mapped[dict | None] = mapped_column(JSONB)  # per endpoint: requests, rows, failures, url templates
    coverage: Mapped[dict | None] = mapped_column(JSONB)
    validation: Mapped[dict | None] = mapped_column(JSONB)
    units: Mapped[dict | None] = mapped_column(JSONB)
    started_at: Mapped[datetime] = _now()
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class PitcherBio(Base):
    """MLB Stats API person record (birth date, throwing hand) for every pitcher in the aging dataset."""

    __tablename__ = "pitcher_bio"
    pitcher_id: Mapped[int] = mapped_column(Integer, primary_key=True)  # MLBAM id (not FK: most are not targets)
    full_name: Mapped[str | None] = mapped_column(String(120))
    birth_date: Mapped[date | None] = mapped_column(Date)
    throws: Mapped[str | None] = mapped_column(String(1))
    primary_position: Mapped[str | None] = mapped_column(String(8))
    birth_country: Mapped[str | None] = mapped_column(String(64))
    mlb_debut_date: Mapped[date | None] = mapped_column(Date)
    source: Mapped[str] = mapped_column(String(64), default="mlb_stats_api.people")
    source_url: Mapped[str | None] = mapped_column(Text)  # exact bulk-people URL (100 ids; can exceed 512 chars)
    fetched_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    updated_at: Mapped[datetime] = _now()
    __table_args__ = (CheckConstraint("throws IS NULL OR throws IN ('L','R','S')", name="ck_bio_throws"),)


class PitcherSeasonStats(Base):
    """One row per (pitcher, regular season): workload and command proxies."""

    __tablename__ = "pitcher_season_stats"
    pitcher_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    season: Mapped[int] = mapped_column(SmallInteger, primary_key=True)
    player_name: Mapped[str | None] = mapped_column(String(120))
    pitch_hand: Mapped[str | None] = mapped_column(String(1))
    age_june30: Mapped[int | None] = mapped_column(SmallInteger)  # completed years on June 30 (from pitcher_bio)
    age_june30_exact: Mapped[float | None] = mapped_column(Float)  # years, days/365.2425
    savant_player_age: Mapped[int | None] = mapped_column(SmallInteger)  # Savant's own age column (cross-check only)
    games: Mapped[int | None] = mapped_column(SmallInteger)
    games_started: Mapped[int | None] = mapped_column(SmallInteger)
    outs: Mapped[int | None] = mapped_column(Integer)
    ip: Mapped[float | None] = mapped_column(Float)  # outs / 3 (true innings, not baseball notation)
    batters_faced: Mapped[int | None] = mapped_column(Integer)
    pitch_count: Mapped[int | None] = mapped_column(Integer)
    k_pct: Mapped[float | None] = mapped_column(Float)
    bb_pct: Mapped[float | None] = mapped_column(Float)
    in_zone_pct: Mapped[float | None] = mapped_column(Float)
    oz_swing_pct: Mapped[float | None] = mapped_column(Float)  # chase rate
    z_swing_pct: Mapped[float | None] = mapped_column(Float)
    swing_pct: Mapped[float | None] = mapped_column(Float)
    whiff_pct: Mapped[float | None] = mapped_column(Float)
    edge_pct: Mapped[float | None] = mapped_column(Float)
    f_strike_pct: Mapped[float | None] = mapped_column(Float)
    meatball_pct: Mapped[float | None] = mapped_column(Float)
    iz_contact_pct: Mapped[float | None] = mapped_column(Float)
    oz_contact_pct: Mapped[float | None] = mapped_column(Float)
    fastball_avg_speed_mph: Mapped[float | None] = mapped_column(Float)
    fastball_avg_spin_rpm: Mapped[float | None] = mapped_column(Float)
    breaking_avg_speed_mph: Mapped[float | None] = mapped_column(Float)
    breaking_avg_spin_rpm: Mapped[float | None] = mapped_column(Float)
    offspeed_avg_speed_mph: Mapped[float | None] = mapped_column(Float)
    offspeed_avg_spin_rpm: Mapped[float | None] = mapped_column(Float)
    extension_ft: Mapped[float | None] = mapped_column(Float)
    release_pos_x_ft: Mapped[float | None] = mapped_column(Float)
    release_pos_z_ft: Mapped[float | None] = mapped_column(Float)
    arm_angle_deg: Mapped[float | None] = mapped_column(Float)
    xwoba: Mapped[float | None] = mapped_column(Float)
    era: Mapped[float | None] = mapped_column(Float)
    measurement_era: Mapped[str] = mapped_column(String(10))  # trackman (2015-2019) | hawkeye (2020+)
    low_sample: Mapped[bool] = mapped_column(Boolean)  # pitch_count < min season pitches
    quality_flags: Mapped[list] = mapped_column(JSONB, default=list)
    field_sources: Mapped[dict] = mapped_column(JSONB, default=dict)  # column -> source key
    source_urls: Mapped[dict] = mapped_column(JSONB, default=dict)  # source key -> exact URL fetched
    fetched_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))  # oldest upstream fetch used
    etl_run_id: Mapped[int | None] = mapped_column(BigInteger)
    updated_at: Mapped[datetime] = _now()
    __table_args__ = (Index("ix_pss_season", "season"),
                      CheckConstraint("measurement_era IN ('trackman','hawkeye')", name="ck_pss_era"))


class PitcherSeasonPitchStats(Base):
    """One row per (pitcher, season, leaderboard pitch type)."""

    __tablename__ = "pitcher_season_pitch_stats"
    pitcher_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    season: Mapped[int] = mapped_column(SmallInteger, primary_key=True)
    pitch_type: Mapped[str] = mapped_column(String(4), primary_key=True)  # official code as Savant leaderboards use it
    pitch_family: Mapped[str] = mapped_column(String(16))
    pitch_group: Mapped[str] = mapped_column(String(10))
    pitch_name: Mapped[str | None] = mapped_column(String(40))
    pitch_hand: Mapped[str | None] = mapped_column(String(1))
    pitches: Mapped[int | None] = mapped_column(Integer)
    total_pitches: Mapped[int | None] = mapped_column(Integer)
    usage_pct: Mapped[float | None] = mapped_column(Float)
    avg_speed_mph: Mapped[float | None] = mapped_column(Float)
    avg_spin_rpm: Mapped[float | None] = mapped_column(Float)
    break_z_in: Mapped[float | None] = mapped_column(Float)  # vertical break incl. gravity
    ivb_in: Mapped[float | None] = mapped_column(Float)  # induced vertical break
    break_x_in: Mapped[float | None] = mapped_column(Float)  # horizontal break (Savant pitcher_break_x convention)
    break_z_vs_league_in: Mapped[float | None] = mapped_column(Float)
    break_x_vs_league_in: Mapped[float | None] = mapped_column(Float)
    extension_ft: Mapped[float | None] = mapped_column(Float)
    release_pos_x_ft: Mapped[float | None] = mapped_column(Float)
    release_pos_z_ft: Mapped[float | None] = mapped_column(Float)
    arm_angle_deg: Mapped[float | None] = mapped_column(Float)
    swings: Mapped[int | None] = mapped_column(Integer)
    whiffs: Mapped[int | None] = mapped_column(Integer)
    whiff_pct: Mapped[float | None] = mapped_column(Float)
    k_pct: Mapped[float | None] = mapped_column(Float)
    put_away_pct: Mapped[float | None] = mapped_column(Float)
    run_value: Mapped[float | None] = mapped_column(Float)
    run_value_per_100: Mapped[float | None] = mapped_column(Float)
    batters_faced: Mapped[int | None] = mapped_column(Integer)
    woba: Mapped[float | None] = mapped_column(Float)
    xwoba: Mapped[float | None] = mapped_column(Float)
    hard_hit_pct: Mapped[float | None] = mapped_column(Float)
    measurement_era: Mapped[str] = mapped_column(String(10))
    low_sample: Mapped[bool] = mapped_column(Boolean)  # pitches < min pitch-type pitches
    quality_flags: Mapped[list] = mapped_column(JSONB, default=list)
    values_by_source: Mapped[dict] = mapped_column(JSONB, default=dict)  # raw per-source values (audit, conflicts)
    field_sources: Mapped[dict] = mapped_column(JSONB, default=dict)
    source_urls: Mapped[dict] = mapped_column(JSONB, default=dict)
    fetched_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    etl_run_id: Mapped[int | None] = mapped_column(BigInteger)
    updated_at: Mapped[datetime] = _now()
    __table_args__ = (Index("ix_psps_season_family", "season", "pitch_family"),
                      CheckConstraint("measurement_era IN ('trackman','hawkeye')", name="ck_psps_era"))


class AgingObservation(Base):
    """Paired consecutive-season (t -> t+1) observation for one pitcher, unit and metric.

    unit (`pitch_family` column) is a pitch family (four_seam, slider, ...), a pitch group ("group:fastball"), or
    "ALL" for pitcher-level metrics. Rows exist for every unit that qualified in season t, including pitchers who did
    not qualify in t+1 (survived = false, value_to/delta null) so survivor bias can be modelled.
    """

    __tablename__ = "aging_observations"
    obs_id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    pitcher_id: Mapped[int] = mapped_column(Integer)
    pitch_family: Mapped[str] = mapped_column(String(24))
    metric: Mapped[str] = mapped_column(String(32))
    unit: Mapped[str | None] = mapped_column(String(16))
    season_from: Mapped[int] = mapped_column(SmallInteger)
    season_to: Mapped[int] = mapped_column(SmallInteger)
    birth_year: Mapped[int | None] = mapped_column(SmallInteger)
    age_from: Mapped[float | None] = mapped_column(Float)  # exact age on June 30 of season_from
    age_from_int: Mapped[int | None] = mapped_column(SmallInteger)
    age_to: Mapped[float | None] = mapped_column(Float)
    value_from: Mapped[float | None] = mapped_column(Float)
    value_to: Mapped[float | None] = mapped_column(Float)
    delta: Mapped[float | None] = mapped_column(Float)
    n_from: Mapped[int | None] = mapped_column(Integer)  # pitches of this unit in season_from
    n_to: Mapped[int | None] = mapped_column(Integer)
    workload_pitches_from: Mapped[int | None] = mapped_column(Integer)
    workload_pitches_to: Mapped[int | None] = mapped_column(Integer)
    ip_from: Mapped[float | None] = mapped_column(Float)
    ip_to: Mapped[float | None] = mapped_column(Float)
    survived: Mapped[bool] = mapped_column(Boolean)  # pitcher qualified (>= min season pitches) in season_to
    unit_present_next: Mapped[bool] = mapped_column(Boolean)  # unit qualified (>= min pitches) in season_to
    both_qualified: Mapped[bool] = mapped_column(Boolean)  # survived and unit qualified in both seasons
    returned_later: Mapped[bool] = mapped_column(Boolean)  # missed season_to but qualified in a later season
    next_qualifying_season: Mapped[int | None] = mapped_column(SmallInteger)
    era_from: Mapped[str] = mapped_column(String(10))
    era_to: Mapped[str] = mapped_column(String(10))
    era_change: Mapped[bool] = mapped_column(Boolean)
    short_season: Mapped[bool] = mapped_column(Boolean)  # either season is the 60-game 2020 season
    reclass_flag: Mapped[bool] = mapped_column(Boolean)  # official codes inside the unit differ between seasons
    codes_from: Mapped[list | None] = mapped_column(JSONB)
    codes_to: Mapped[list | None] = mapped_column(JSONB)
    min_pitches_threshold: Mapped[int] = mapped_column(Integer)
    min_season_pitches_threshold: Mapped[int] = mapped_column(Integer)
    etl_run_id: Mapped[int | None] = mapped_column(BigInteger)
    computed_at: Mapped[datetime] = _now()
    __table_args__ = (
        UniqueConstraint("pitcher_id", "pitch_family", "metric", "season_from", name="uq_aging_obs"),
        Index("ix_aging_obs_metric", "pitch_family", "metric", "season_from"),
        Index("ix_aging_obs_pitcher", "pitcher_id"),
    )


class AgingCohort(Base):
    """Survivor counts and delta summaries by age (June 30) or 5-year birth-year band."""

    __tablename__ = "aging_cohorts"
    cohort_id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    cohort_type: Mapped[str] = mapped_column(String(16))  # age | birth_year_band
    cohort_key: Mapped[str] = mapped_column(String(16))  # "31" or "1985-1989"
    scope: Mapped[str] = mapped_column(String(8))  # "pooled" or the season_from as text
    n_pitchers: Mapped[int] = mapped_column(Integer)  # qualifying pitcher-seasons at t
    n_survived: Mapped[int] = mapped_column(Integer)
    n_dropped: Mapped[int] = mapped_column(Integer)
    n_returned_later: Mapped[int] = mapped_column(Integer)
    n_era_change: Mapped[int] = mapped_column(Integer)
    survival_rate: Mapped[float | None] = mapped_column(Float)
    mean_age: Mapped[float | None] = mapped_column(Float)
    metric_summaries: Mapped[dict] = mapped_column(JSONB)  # unit/metric -> {n, mean, sd, median} of deltas
    etl_run_id: Mapped[int | None] = mapped_column(BigInteger)
    computed_at: Mapped[datetime] = _now()
    __table_args__ = (UniqueConstraint("cohort_type", "cohort_key", "scope", name="uq_aging_cohort"),)
