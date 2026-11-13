"""Model lifecycle tables (np-refit step): training provenance, promotion gates, promotion/rollback audit trail.

`model_registry.status` values: candidate | promoted (= the production model) | previous_production (the rollback
target, one slot per pitcher) | archived. "promoted" is kept as the stored value for production so every existing
consumer (API, challenge, simulator, MLflow `promoted` alias, frontend) keeps working; API views add `stage`.
"""
from __future__ import annotations

from datetime import date, datetime

from sqlalchemy import BigInteger, Date, DateTime, ForeignKey, Integer, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from .models import Base

STAGE_BY_STATUS = {"promoted": "production", "previous_production": "previous_production",
                   "candidate": "candidate", "archived": "archived"}


class ModelLifecycle(Base):
    """One row per model trained by the lifecycle code (deployment refits and scheduled retrains).

    Legacy season-split models have no row; their data cutoff is derived from the splits they were trained on."""

    __tablename__ = "model_lifecycle"
    model_version: Mapped[str] = mapped_column(ForeignKey("model_registry.model_version", ondelete="CASCADE"),
                                               primary_key=True)
    pitcher_id: Mapped[int] = mapped_column(Integer, index=True)
    training_scope: Mapped[str] = mapped_column(String(32))  # deployment_refit | scheduled_retrain
    data_cutoff: Mapped[date] = mapped_column(Date)  # last game date included in the fit
    deployment_cutoff: Mapped[date] = mapped_column(Date)  # fit used games strictly before this date
    n_train_samples: Mapped[int] = mapped_column(Integer)
    n_train_games: Mapped[int] = mapped_column(Integer)
    selected_params: Mapped[dict] = mapped_column(JSONB)
    dev_metrics: Mapped[dict] = mapped_column(JSONB)  # rolling-origin development metrics of the recipe
    dev_evaluation_run_id: Mapped[int | None] = mapped_column(BigInteger)
    gate_results: Mapped[dict | None] = mapped_column(JSONB)  # latest gate evaluation
    retrain_policy: Mapped[str | None] = mapped_column(String(24))
    trigger: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)


class ModelPromotionEvent(Base):
    """Append-only audit trail: gate evaluations, promotions, gate failures and rollbacks."""

    __tablename__ = "model_promotion_events"
    event_id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    pitcher_id: Mapped[int] = mapped_column(Integer, index=True)
    model_version: Mapped[str] = mapped_column(ForeignKey("model_registry.model_version", ondelete="CASCADE"),
                                               index=True)
    action: Mapped[str] = mapped_column(String(24))  # promote | gate_failed | rollback | gates_evaluated
    from_status: Mapped[str | None] = mapped_column(String(24))
    to_status: Mapped[str | None] = mapped_column(String(24))
    replaced_version: Mapped[str | None] = mapped_column(String(64))  # production model before this event
    gate_results: Mapped[dict | None] = mapped_column(JSONB)
    source: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)
