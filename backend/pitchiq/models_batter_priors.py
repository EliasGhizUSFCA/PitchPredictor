"""MLB-wide batter season leaderboards (Baseball Savant) and the as-of batter priors built from them.

Owned by step np-features (pitchiq/batter_priors.py). Raw leaderboard rows are stored per (batter, season[, pitch
type | pitcher hand]) with the exact source URL and fetch time. `batter_priors_asof` holds the empirical-Bayes
smoothed priors for an as-of season s, computed ONLY from seasons < s; `batter_prior_meta` records the league means,
the estimated shrinkage strengths and the source seasons of each as-of build.
"""
from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, Float, Integer, SmallInteger, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from .models import Base


def _now():
    return mapped_column(DateTime(timezone=True), server_default=func.now())


class BatterPitchTypeSeason(Base):
    """pitch-arsenal-stats?type=batter: a batter's results against one pitch type in one season (all pitchers)."""

    __tablename__ = "batter_lb_pitch_type"
    batter_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    season: Mapped[int] = mapped_column(SmallInteger, primary_key=True)
    pitch_type: Mapped[str] = mapped_column(String(4), primary_key=True)
    player_name: Mapped[str | None] = mapped_column(String(120))
    pitches: Mapped[float | None] = mapped_column(Float)
    pa: Mapped[float | None] = mapped_column(Float)
    whiff_pct: Mapped[float | None] = mapped_column(Float)
    k_pct: Mapped[float | None] = mapped_column(Float)
    put_away_pct: Mapped[float | None] = mapped_column(Float)
    woba: Mapped[float | None] = mapped_column(Float)
    est_woba: Mapped[float | None] = mapped_column(Float)
    run_value: Mapped[float | None] = mapped_column(Float)
    run_value_per_100: Mapped[float | None] = mapped_column(Float)
    hard_hit_pct: Mapped[float | None] = mapped_column(Float)
    source_url: Mapped[str] = mapped_column(Text)
    fetched_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    loaded_at: Mapped[datetime] = _now()


class BatterDisciplineSeason(Base):
    """leaderboard/custom?type=batter: plate-discipline season line (all pitchers)."""

    __tablename__ = "batter_lb_discipline"
    batter_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    season: Mapped[int] = mapped_column(SmallInteger, primary_key=True)
    player_name: Mapped[str | None] = mapped_column(String(120))
    pa: Mapped[float | None] = mapped_column(Float)
    pitch_count: Mapped[float | None] = mapped_column(Float)
    k_pct: Mapped[float | None] = mapped_column(Float)
    bb_pct: Mapped[float | None] = mapped_column(Float)
    swing_pct: Mapped[float | None] = mapped_column(Float)
    o_swing_pct: Mapped[float | None] = mapped_column(Float)
    z_swing_pct: Mapped[float | None] = mapped_column(Float)
    whiff_pct: Mapped[float | None] = mapped_column(Float)
    iz_contact_pct: Mapped[float | None] = mapped_column(Float)
    oz_contact_pct: Mapped[float | None] = mapped_column(Float)
    in_zone_pct: Mapped[float | None] = mapped_column(Float)
    woba: Mapped[float | None] = mapped_column(Float)
    xwoba: Mapped[float | None] = mapped_column(Float)
    source_url: Mapped[str] = mapped_column(Text)
    fetched_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    loaded_at: Mapped[datetime] = _now()


class BatterPlatoonSeason(Base):
    """statcast_search/csv player_type=batter group_by=name pitcher_throws=L|R: a batter's season vs one pitcher hand."""

    __tablename__ = "batter_lb_platoon"
    batter_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    season: Mapped[int] = mapped_column(SmallInteger, primary_key=True)
    pitcher_throws: Mapped[str] = mapped_column(String(1), primary_key=True)
    player_name: Mapped[str | None] = mapped_column(String(120))
    pitches: Mapped[float | None] = mapped_column(Float)
    swings: Mapped[float | None] = mapped_column(Float)
    whiffs: Mapped[float | None] = mapped_column(Float)
    takes: Mapped[float | None] = mapped_column(Float)
    pa: Mapped[float | None] = mapped_column(Float)
    so: Mapped[float | None] = mapped_column(Float)
    bb: Mapped[float | None] = mapped_column(Float)
    woba: Mapped[float | None] = mapped_column(Float)
    xwoba: Mapped[float | None] = mapped_column(Float)
    run_value_per_100: Mapped[float | None] = mapped_column(Float)
    source_url: Mapped[str] = mapped_column(Text)
    fetched_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    loaded_at: Mapped[datetime] = _now()


class BatterPriorAsOf(Base):
    """Empirical-Bayes smoothed batter priors for as-of season `as_of_season`, from seasons < as_of_season only."""

    __tablename__ = "batter_priors_asof"
    batter_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    as_of_season: Mapped[int] = mapped_column(SmallInteger, primary_key=True)
    pitcher_throws: Mapped[str] = mapped_column(String(1), primary_key=True)
    features: Mapped[dict] = mapped_column(JSONB)
    n_prior_pa: Mapped[float] = mapped_column(Float)
    source_seasons: Mapped[list] = mapped_column(JSONB)
    built_at: Mapped[datetime] = _now()


class BatterPriorMeta(Base):
    __tablename__ = "batter_prior_meta"
    as_of_season: Mapped[int] = mapped_column(SmallInteger, primary_key=True)
    pitcher_throws: Mapped[str] = mapped_column(String(1), primary_key=True)
    source_seasons: Mapped[list] = mapped_column(JSONB)
    league_means: Mapped[dict] = mapped_column(JSONB)
    shrinkage: Mapped[dict] = mapped_column(JSONB)
    n_batters: Mapped[int] = mapped_column(Integer)
    method: Mapped[str] = mapped_column(Text)
    built_at: Mapped[datetime] = _now()
