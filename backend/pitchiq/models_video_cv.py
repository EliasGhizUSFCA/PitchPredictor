"""Video alignment tables (pitchiq.video.alignment). Owned by pitchiq/video/align_service.py and alignment_eval.py.

* `video_alignment_runs`: one row per automatic alignment attempt on a licensed_local asset, with every diagnostic
  the detector produced (scene cuts, view scores, motion series, confidence components). The applied values are
  also written to `video_assets` (pre_pitch_pause_time, release_frame_timestamp, ...).
* `video_alignment_validations`: accuracy reports of the detector on clips with known ground truth. Only
  `data_kind='synthetic'` exists today (tests/video_synth.py); no licensed real footage was available.
"""
from __future__ import annotations

from datetime import datetime

from sqlalchemy import BigInteger, Boolean, CheckConstraint, DateTime, Float, ForeignKey, Integer, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from .models import Base


def _now():
    return mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)


class VideoAlignmentRun(Base):
    __tablename__ = "video_alignment_runs"
    alignment_run_id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    video_asset_id: Mapped[int] = mapped_column(ForeignKey("video_assets.video_asset_id", ondelete="CASCADE"),
                                                index=True)
    detector_version: Mapped[str] = mapped_column(String(32))
    params: Mapped[dict] = mapped_column(JSONB)
    status: Mapped[str] = mapped_column(String(16))  # auto_aligned | unaligned
    applied: Mapped[bool] = mapped_column(Boolean, default=True)  # False when a user correction was kept
    alignment_confidence: Mapped[float | None] = mapped_column(Float)
    clip_start_time: Mapped[float | None] = mapped_column(Float)
    onset_time: Mapped[float | None] = mapped_column(Float)
    release_time: Mapped[float | None] = mapped_column(Float)
    pre_pitch_pause_time: Mapped[float | None] = mapped_column(Float)
    pitch_end_time: Mapped[float | None] = mapped_column(Float)
    clip_duration: Mapped[float | None] = mapped_column(Float)
    fps: Mapped[float | None] = mapped_column(Float)
    n_frames: Mapped[int | None] = mapped_column(Integer)
    file_fingerprint: Mapped[str | None] = mapped_column(String(64))
    result: Mapped[dict] = mapped_column(JSONB)  # full AlignmentResult (cuts, segments, motion, components, ...)
    error: Mapped[str | None] = mapped_column(Text)
    runtime_ms: Mapped[float | None] = mapped_column(Float)
    created_at: Mapped[datetime] = _now()
    __table_args__ = (CheckConstraint("status IN ('auto_aligned','unaligned')", name="ck_valign_status"),)


class VideoAlignmentValidation(Base):
    __tablename__ = "video_alignment_validations"
    validation_id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    detector_version: Mapped[str] = mapped_column(String(32))
    role: Mapped[str] = mapped_column(String(16))  # development | evaluation
    data_kind: Mapped[str] = mapped_column(String(16), default="synthetic")
    generator: Mapped[str] = mapped_column(String(96))
    n_clips: Mapped[int] = mapped_column(Integer)
    seeds: Mapped[dict | None] = mapped_column(JSONB)
    params: Mapped[dict] = mapped_column(JSONB)
    summary: Mapped[dict] = mapped_column(JSONB)
    per_clip: Mapped[list] = mapped_column(JSONB)
    mlflow_run_id: Mapped[str | None] = mapped_column(String(64))
    created_at: Mapped[datetime] = _now()
    __table_args__ = (CheckConstraint("data_kind IN ('synthetic','licensed_real')", name="ck_valid_data_kind"),
                      CheckConstraint("role IN ('development','evaluation')", name="ck_valid_role"))
