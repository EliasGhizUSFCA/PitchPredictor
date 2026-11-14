"""System B tables: aging-forecast model registry, rolling-origin backtests, survivor / confounding analyses and
pitch characteristic forecasts. Owned by pitchiq/aging_model/*.

`aging_model_registry` and `aging_backtest_runs` mirror the shape of `model_registry` / `evaluation_runs` so the
existing MLflow mirror (pitchiq.ml.tracking.Tracker) can log them unchanged (into its own experiment
"<prefix>-aging-<pitcher_id>"). They are separate tables on purpose: next-pitch code resolves models per pitcher from
`model_registry`, and population aging models must never be picked up there.
"""
from __future__ import annotations

from datetime import datetime

from sqlalchemy import (BigInteger, Boolean, DateTime, Float, ForeignKey, Index, Integer, SmallInteger, String, Text,
                        UniqueConstraint, func)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from .models import Base


def _now():
    return mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)


class AgingBacktest(Base):
    """One run of the engine: rolling-origin backtest for every target + forecast from the historical cutoff."""

    __tablename__ = "aging_backtests"
    backtest_key: Mapped[str] = mapped_column(String(48), primary_key=True)
    pitcher_id: Mapped[int] = mapped_column(Integer)  # forecast pitcher (population fits use every pitcher)
    historical_cutoff: Mapped[int] = mapped_column(SmallInteger)
    status: Mapped[str] = mapped_column(String(16), default="running")  # running | succeeded | failed
    config: Mapped[dict] = mapped_column(JSONB)
    targets: Mapped[list] = mapped_column(JSONB)
    selection: Mapped[dict | None] = mapped_column(JSONB)  # target key -> horizon -> chosen method + ranking
    summary: Mapped[dict | None] = mapped_column(JSONB)
    failures: Mapped[list | None] = mapped_column(JSONB)
    data_fingerprint: Mapped[dict | None] = mapped_column(JSONB)
    started_at: Mapped[datetime] = _now()
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class AgingModelEntry(Base):
    __tablename__ = "aging_model_registry"
    model_version: Mapped[str] = mapped_column(String(80), primary_key=True)
    backtest_key: Mapped[str] = mapped_column(ForeignKey("aging_backtests.backtest_key", ondelete="CASCADE"))
    pitcher_id: Mapped[int] = mapped_column(Integer, index=True)
    model_family: Mapped[str] = mapped_column(String(40))  # aging_<method> | aging_location_own_history | aging_selected
    feature_version: Mapped[str] = mapped_column(String(32))
    feature_schema: Mapped[dict] = mapped_column(JSONB)
    classes: Mapped[list] = mapped_column(JSONB)  # target keys covered ("unit|metric")
    config: Mapped[dict] = mapped_column(JSONB)
    dataset_fingerprint: Mapped[dict] = mapped_column(JSONB)
    artifact_path: Mapped[str] = mapped_column(Text)
    artifact_sha256: Mapped[str] = mapped_column(String(64))
    calibration: Mapped[dict | None] = mapped_column(JSONB)  # interval method
    validation_metrics: Mapped[dict | None] = mapped_column(JSONB)  # development backtest headline metrics
    status: Mapped[str] = mapped_column(String(24), default="candidate")  # candidate | promoted | archived
    historical_cutoff: Mapped[int] = mapped_column(SmallInteger)
    created_at: Mapped[datetime] = _now()


class AgingBacktestRun(Base):
    """Evaluation row (shape of evaluation_runs): one per method and split (development / retrospective)."""

    __tablename__ = "aging_backtest_runs"
    evaluation_run_id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    backtest_key: Mapped[str] = mapped_column(ForeignKey("aging_backtests.backtest_key", ondelete="CASCADE"))
    pitcher_id: Mapped[int] = mapped_column(Integer)
    model_version: Mapped[str] = mapped_column(ForeignKey("aging_model_registry.model_version", ondelete="CASCADE"))
    kind: Mapped[str] = mapped_column(String(32))  # aging_backtest | aging_location_backtest
    split: Mapped[str] = mapped_column(String(48))  # development | retrospective_2026_previously_viewed
    dataset_definition: Mapped[dict] = mapped_column(JSONB)
    n_samples: Mapped[int] = mapped_column(Integer)
    metrics: Mapped[dict] = mapped_column(JSONB)
    status: Mapped[str] = mapped_column(String(16), default="completed")
    created_at: Mapped[datetime] = _now()


class AgingBacktestMetric(Base):
    __tablename__ = "aging_backtest_metrics"
    metric_id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    backtest_key: Mapped[str] = mapped_column(ForeignKey("aging_backtests.backtest_key", ondelete="CASCADE"))
    method: Mapped[str] = mapped_column(String(32))
    unit: Mapped[str] = mapped_column(String(24))
    metric: Mapped[str] = mapped_column(String(32))
    horizon: Mapped[int] = mapped_column(SmallInteger)
    breakdown_type: Mapped[str] = mapped_column(String(32))  # subset | age_band_dev_same_era | target_season | ...
    breakdown_key: Mapped[str] = mapped_column(String(32))
    n: Mapped[int] = mapped_column(Integer)
    mae: Mapped[float] = mapped_column(Float)
    rmse: Mapped[float] = mapped_column(Float)
    bias: Mapped[float] = mapped_column(Float)
    coverage80: Mapped[float] = mapped_column(Float)
    width80: Mapped[float] = mapped_column(Float)
    pinball_q10: Mapped[float] = mapped_column(Float)
    pinball_q50: Mapped[float] = mapped_column(Float)
    pinball_q90: Mapped[float] = mapped_column(Float)
    pinball_mean: Mapped[float] = mapped_column(Float)
    __table_args__ = (Index("ix_abm_key_target", "backtest_key", "unit", "metric", "horizon"),)


class AgingBacktestPrediction(Base):
    """The forecast pitcher's own backtest forecasts (every method, origin and horizon), with the realized value."""

    __tablename__ = "aging_backtest_predictions"
    prediction_id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    backtest_key: Mapped[str] = mapped_column(ForeignKey("aging_backtests.backtest_key", ondelete="CASCADE"))
    pitcher_id: Mapped[int] = mapped_column(Integer)
    method: Mapped[str] = mapped_column(String(32))
    unit: Mapped[str] = mapped_column(String(24))
    metric: Mapped[str] = mapped_column(String(32))
    origin_season: Mapped[int] = mapped_column(SmallInteger)
    target_season: Mapped[int] = mapped_column(SmallInteger)
    horizon: Mapped[int] = mapped_column(SmallInteger)
    origin_value: Mapped[float | None] = mapped_column(Float)
    origin_age: Mapped[float | None] = mapped_column(Float)
    point: Mapped[float] = mapped_column(Float)
    q10: Mapped[float | None] = mapped_column(Float)
    q50: Mapped[float | None] = mapped_column(Float)
    q90: Mapped[float | None] = mapped_column(Float)
    actual: Mapped[float | None] = mapped_column(Float)  # null: unit did not qualify in the target season
    interval_source: Mapped[str | None] = mapped_column(String(40))
    evaluation_label: Mapped[str] = mapped_column(String(48))  # development | retrospective_2026_previously_viewed
    __table_args__ = (Index("ix_abp_key_pitcher", "backtest_key", "pitcher_id", "unit", "metric"),)


class AgingAnalysis(Base):
    __tablename__ = "aging_analyses"
    analysis_id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    backtest_key: Mapped[str] = mapped_column(ForeignKey("aging_backtests.backtest_key", ondelete="CASCADE"))
    kind: Mapped[str] = mapped_column(String(32))  # survivor_confounding
    data_cutoff: Mapped[int] = mapped_column(SmallInteger)
    unit: Mapped[str] = mapped_column(String(24))
    metric: Mapped[str] = mapped_column(String(32))
    results: Mapped[dict] = mapped_column(JSONB)
    created_at: Mapped[datetime] = _now()


class PitchCharacteristicForecast(Base):
    """Forecast of one pitcher-season characteristic for a future season from a historical cutoff.

    One row per (target season, unit, metric, method); `selected` marks the method chosen for that target and horizon
    by the development backtest. Values are conditional on the unit qualifying in the target season."""

    __tablename__ = "pitch_characteristic_forecasts"
    forecast_id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    backtest_key: Mapped[str] = mapped_column(ForeignKey("aging_backtests.backtest_key", ondelete="CASCADE"))
    pitcher_id: Mapped[int] = mapped_column(Integer)
    created_at: Mapped[datetime] = _now()
    historical_cutoff: Mapped[int] = mapped_column(SmallInteger)  # last season of data used
    target_season: Mapped[int] = mapped_column(SmallInteger)
    horizon: Mapped[int] = mapped_column(SmallInteger)
    projected_age: Mapped[float | None] = mapped_column(Float)  # exact age on June 30 of target season (birth date)
    projected_age_completed: Mapped[int | None] = mapped_column(SmallInteger)
    unit: Mapped[str] = mapped_column(String(24))  # pitch family or ALL
    metric: Mapped[str] = mapped_column(String(32))
    value_units: Mapped[str | None] = mapped_column(String(40))
    model_version: Mapped[str] = mapped_column(ForeignKey("aging_model_registry.model_version", ondelete="CASCADE"))
    method: Mapped[str] = mapped_column(String(32))
    selected: Mapped[bool] = mapped_column(Boolean, default=False)
    selection_model_version: Mapped[str | None] = mapped_column(String(80))
    point: Mapped[float] = mapped_column(Float)  # median forecast
    mean: Mapped[float | None] = mapped_column(Float)
    sd: Mapped[float | None] = mapped_column(Float)
    lo80: Mapped[float | None] = mapped_column(Float)  # q10
    hi80: Mapped[float | None] = mapped_column(Float)  # q90
    quantiles: Mapped[dict] = mapped_column(JSONB)  # {"0.05": .., "0.1": .., ...}
    distribution: Mapped[dict] = mapped_column(JSONB)  # params for Monte Carlo (normal | quantile_interp)
    sample_draws: Mapped[list | None] = mapped_column(JSONB)
    draws_seed: Mapped[int | None] = mapped_column(BigInteger)
    last_observed_season: Mapped[int | None] = mapped_column(SmallInteger)
    last_observed_value: Mapped[float | None] = mapped_column(Float)
    assumptions: Mapped[dict] = mapped_column(JSONB)
    __table_args__ = (
        UniqueConstraint("backtest_key", "pitcher_id", "target_season", "unit", "metric", "method",
                         name="uq_pcf_row"),
        Index("ix_pcf_pitcher_cutoff", "pitcher_id", "historical_cutoff", "target_season"),
    )
