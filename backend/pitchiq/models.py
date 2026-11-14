"""PostgreSQL schema for PitchIQ (SQLAlchemy 2.0 ORM).

Provenance: every row ingested from an external source carries `source` and
`ingested_at` (or equivalent) so the historical record is auditable.
"""
from __future__ import annotations

from datetime import date, datetime

from sqlalchemy import (
    BigInteger,
    Boolean,
    CheckConstraint,
    Date,
    DateTime,
    Float,
    ForeignKey,
    ForeignKeyConstraint,
    Index,
    Integer,
    SmallInteger,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


def _now():
    return mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)


# --------------------------------------------------------------------------- people / games
class Pitcher(Base):
    __tablename__ = "pitchers"
    pitcher_id: Mapped[int] = mapped_column(Integer, primary_key=True)  # MLBAM id
    full_name: Mapped[str] = mapped_column(String(120))
    throws: Mapped[str | None] = mapped_column(String(1))
    is_active_target: Mapped[bool] = mapped_column(Boolean, default=False, server_default="false")
    seasons: Mapped[list | None] = mapped_column(JSONB)
    extra: Mapped[dict | None] = mapped_column(JSONB)
    source: Mapped[str] = mapped_column(String(64), default="mlb_stats_api")
    created_at: Mapped[datetime] = _now()
    __table_args__ = (CheckConstraint("throws IN ('L','R','S')", name="ck_pitcher_throws"),)


class Batter(Base):
    __tablename__ = "batters"
    batter_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    full_name: Mapped[str | None] = mapped_column(String(120))
    bats: Mapped[str | None] = mapped_column(String(1))
    extra: Mapped[dict | None] = mapped_column(JSONB)
    source: Mapped[str] = mapped_column(String(64), default="mlb_stats_api")
    created_at: Mapped[datetime] = _now()


class Game(Base):
    __tablename__ = "games"
    game_pk: Mapped[int] = mapped_column(Integer, primary_key=True)
    game_date: Mapped[date] = mapped_column(Date, index=True)
    season: Mapped[int] = mapped_column(SmallInteger, index=True)
    game_type: Mapped[str] = mapped_column(String(2))
    home_team: Mapped[str] = mapped_column(String(8))
    away_team: Mapped[str] = mapped_column(String(8))
    venue: Mapped[str | None] = mapped_column(String(120))
    feed_fetched_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    extra: Mapped[dict | None] = mapped_column(JSONB)
    source: Mapped[str] = mapped_column(String(64), default="statcast")
    created_at: Mapped[datetime] = _now()


class GameLineup(Base):
    """Batting-order slots from the official MLB boxscore (`battingOrder` codes like 301)."""

    __tablename__ = "game_lineups"
    game_pk: Mapped[int] = mapped_column(ForeignKey("games.game_pk", ondelete="CASCADE"), primary_key=True)
    batter_id: Mapped[int] = mapped_column(ForeignKey("batters.batter_id"), primary_key=True)
    team: Mapped[str] = mapped_column(String(8))
    side: Mapped[str] = mapped_column(String(4))  # home / away
    batting_order_slot: Mapped[int] = mapped_column(SmallInteger)  # 1..9
    sub_sequence: Mapped[int] = mapped_column(SmallInteger, default=0)  # 0 = starter
    is_starter: Mapped[bool] = mapped_column(Boolean)
    raw_code: Mapped[str] = mapped_column(String(8))
    source: Mapped[str] = mapped_column(String(64), default="mlb_stats_api.boxscore")
    __table_args__ = (
        CheckConstraint("batting_order_slot BETWEEN 1 AND 9", name="ck_lineup_slot"),
    )


class PlateAppearance(Base):
    __tablename__ = "plate_appearances"
    game_pk: Mapped[int] = mapped_column(ForeignKey("games.game_pk", ondelete="CASCADE"), primary_key=True)
    at_bat_number: Mapped[int] = mapped_column(SmallInteger, primary_key=True)
    pitcher_id: Mapped[int] = mapped_column(ForeignKey("pitchers.pitcher_id"), index=True)
    batter_id: Mapped[int] = mapped_column(ForeignKey("batters.batter_id"), index=True)
    inning: Mapped[int] = mapped_column(SmallInteger)
    inning_topbot: Mapped[str] = mapped_column(String(3))
    stand: Mapped[str] = mapped_column(String(1))
    p_throws: Mapped[str] = mapped_column(String(1))
    batting_order_slot: Mapped[int | None] = mapped_column(SmallInteger)
    times_through_order: Mapped[int | None] = mapped_column(SmallInteger)
    n_pitches: Mapped[int] = mapped_column(SmallInteger)
    event: Mapped[str | None] = mapped_column(String(64))


class Pitch(Base):
    """One historical pitch. Natural key = (game_pk, at_bat_number, pitch_number)."""

    __tablename__ = "pitches"
    pitch_uid: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    game_pk: Mapped[int] = mapped_column(ForeignKey("games.game_pk", ondelete="CASCADE"))
    at_bat_number: Mapped[int] = mapped_column(SmallInteger)
    pitch_number: Mapped[int] = mapped_column(SmallInteger)  # within plate appearance
    game_pitch_seq: Mapped[int] = mapped_column(Integer)  # 1-based order of this pitcher's pitches in game
    pitcher_id: Mapped[int] = mapped_column(ForeignKey("pitchers.pitcher_id"))
    batter_id: Mapped[int] = mapped_column(ForeignKey("batters.batter_id"))
    catcher_id: Mapped[int | None] = mapped_column(Integer)
    game_date: Mapped[date] = mapped_column(Date)
    season: Mapped[int] = mapped_column(SmallInteger)
    stand: Mapped[str] = mapped_column(String(1))
    p_throws: Mapped[str] = mapped_column(String(1))
    balls: Mapped[int] = mapped_column(SmallInteger)
    strikes: Mapped[int] = mapped_column(SmallInteger)
    outs_when_up: Mapped[int] = mapped_column(SmallInteger)
    inning: Mapped[int] = mapped_column(SmallInteger)
    inning_topbot: Mapped[str] = mapped_column(String(3))
    on_1b: Mapped[int | None] = mapped_column(Integer)
    on_2b: Mapped[int | None] = mapped_column(Integer)
    on_3b: Mapped[int | None] = mapped_column(Integer)
    home_score: Mapped[int] = mapped_column(SmallInteger)
    away_score: Mapped[int] = mapped_column(SmallInteger)
    bat_score: Mapped[int | None] = mapped_column(SmallInteger)
    fld_score: Mapped[int | None] = mapped_column(SmallInteger)
    home_team: Mapped[str] = mapped_column(String(8))
    away_team: Mapped[str] = mapped_column(String(8))
    n_thruorder_pitcher: Mapped[int | None] = mapped_column(SmallInteger)
    # ---- label + post-release measurements (NEVER pre-pitch model inputs for this pitch)
    pitch_type: Mapped[str | None] = mapped_column(String(4))
    pitch_name: Mapped[str | None] = mapped_column(String(40))
    release_speed: Mapped[float | None] = mapped_column(Float)
    release_spin_rate: Mapped[float | None] = mapped_column(Float)
    pfx_x: Mapped[float | None] = mapped_column(Float)
    pfx_z: Mapped[float | None] = mapped_column(Float)
    plate_x: Mapped[float | None] = mapped_column(Float)
    plate_z: Mapped[float | None] = mapped_column(Float)
    zone: Mapped[int | None] = mapped_column(SmallInteger)
    description: Mapped[str | None] = mapped_column(String(48))
    events: Mapped[str | None] = mapped_column(String(48))
    type: Mapped[str | None] = mapped_column(String(1))
    des: Mapped[str | None] = mapped_column(Text)
    raw: Mapped[dict | None] = mapped_column(JSONB)  # full original Statcast row
    source: Mapped[str] = mapped_column(String(64), default="baseball_savant.statcast_search")
    ingested_at: Mapped[datetime] = _now()

    pre_pitch_state: Mapped["PrePitchState"] = relationship(back_populates="pitch", uselist=False)

    __table_args__ = (
        UniqueConstraint("game_pk", "at_bat_number", "pitch_number", name="uq_pitch_natural"),
        UniqueConstraint("game_pk", "pitcher_id", "game_pitch_seq", name="uq_pitch_game_seq"),
        ForeignKeyConstraint(
            ["game_pk", "at_bat_number"],
            ["plate_appearances.game_pk", "plate_appearances.at_bat_number"],
            name="fk_pitch_pa",
        ),
        CheckConstraint("balls BETWEEN 0 AND 3", name="ck_pitch_balls"),
        CheckConstraint("strikes BETWEEN 0 AND 2", name="ck_pitch_strikes"),
        CheckConstraint("outs_when_up BETWEEN 0 AND 2", name="ck_pitch_outs"),
        Index("ix_pitches_pitcher_date", "pitcher_id", "game_date", "game_pk", "game_pitch_seq"),
        Index("ix_pitches_season", "pitcher_id", "season"),
    )


class PrePitchState(Base):
    """Reconstructed state known immediately BEFORE the pitch. One row per pitch per feature version."""

    __tablename__ = "pre_pitch_states"
    pitch_uid: Mapped[int] = mapped_column(ForeignKey("pitches.pitch_uid", ondelete="CASCADE"), primary_key=True)
    feature_version: Mapped[str] = mapped_column(String(32), primary_key=True)
    pitcher_id: Mapped[int] = mapped_column(Integer, index=True)
    season: Mapped[int] = mapped_column(SmallInteger)
    split: Mapped[str] = mapped_column(String(12), index=True)  # train / validation / test / unassigned
    features: Mapped[dict] = mapped_column(JSONB)
    label: Mapped[str | None] = mapped_column(String(8))  # model class (after rare-class grouping)
    computed_at: Mapped[datetime] = _now()
    pitch: Mapped[Pitch] = relationship(back_populates="pre_pitch_state")


class PitchSequence(Base):
    """Per plate appearance pitch sequences (for analytics + transition matrices)."""

    __tablename__ = "pitch_sequences"
    game_pk: Mapped[int] = mapped_column(Integer, primary_key=True)
    at_bat_number: Mapped[int] = mapped_column(SmallInteger, primary_key=True)
    pitcher_id: Mapped[int] = mapped_column(Integer, index=True)
    sequence: Mapped[list] = mapped_column(JSONB)  # e.g. ["FF","SL","SL"]
    results: Mapped[list] = mapped_column(JSONB)
    __table_args__ = (
        ForeignKeyConstraint(
            ["game_pk", "at_bat_number"],
            ["plate_appearances.game_pk", "plate_appearances.at_bat_number"],
            ondelete="CASCADE",
        ),
    )


# --------------------------------------------------------------------------- video
class VideoAsset(Base):
    __tablename__ = "video_assets"
    video_asset_id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    pitch_uid: Mapped[int | None] = mapped_column(ForeignKey("pitches.pitch_uid", ondelete="SET NULL"), index=True)
    source_provider: Mapped[str] = mapped_column(String(48))
    source_event_id: Mapped[str] = mapped_column(String(64))  # e.g. Statcast playId (UUID)
    source_page_url: Mapped[str | None] = mapped_column(Text)
    authorized_embed_reference: Mapped[str | None] = mapped_column(Text)
    permitted_media_reference: Mapped[str | None] = mapped_column(Text)
    game_pk: Mapped[int] = mapped_column(Integer)
    pitcher_id: Mapped[int | None] = mapped_column(Integer)
    batter_id: Mapped[int | None] = mapped_column(Integer)
    at_bat_number: Mapped[int | None] = mapped_column(SmallInteger)
    pitch_number: Mapped[int | None] = mapped_column(SmallInteger)
    inning: Mapped[int | None] = mapped_column(SmallInteger)
    clip_start_time: Mapped[float | None] = mapped_column(Float)
    pre_pitch_pause_time: Mapped[float | None] = mapped_column(Float)
    release_frame_timestamp: Mapped[float | None] = mapped_column(Float)
    pitch_end_time: Mapped[float | None] = mapped_column(Float)
    clip_duration: Mapped[float | None] = mapped_column(Float)
    alignment_confidence: Mapped[float | None] = mapped_column(Float)
    # availability: unchecked | external_available | embed_available | licensed_local | metadata_only | unavailable
    video_availability_status: Mapped[str] = mapped_column(String(24), default="unchecked")
    # exact_verified | strong_review | possible | no_match
    match_confidence: Mapped[str] = mapped_column(String(16))
    match_evidence: Mapped[dict | None] = mapped_column(JSONB)
    # not_applicable | unaligned | auto_aligned | user_corrected
    synchronization_status: Mapped[str] = mapped_column(String(16), default="not_applicable")
    # external_link | official_embed | licensed_local | none
    playback_method: Mapped[str] = mapped_column(String(16))
    rights_or_permission_status: Mapped[str] = mapped_column(String(48))
    review_status: Mapped[str] = mapped_column(String(16), default="auto")  # auto | pending | approved | rejected
    discovered_at: Mapped[datetime] = _now()
    last_verified_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    verification_notes: Mapped[str | None] = mapped_column(Text)
    __table_args__ = (
        UniqueConstraint("source_provider", "source_event_id", name="uq_video_source_event"),
        CheckConstraint(
            "match_confidence IN ('exact_verified','strong_review','possible','no_match')",
            name="ck_video_match_conf",
        ),
        CheckConstraint(
            "playback_method IN ('external_link','official_embed','licensed_local','none')",
            name="ck_video_playback",
        ),
        Index("ix_video_game", "game_pk", "at_bat_number", "pitch_number"),
    )


# --------------------------------------------------------------------------- ML
class ModelRegistry(Base):
    __tablename__ = "model_registry"
    model_version: Mapped[str] = mapped_column(String(64), primary_key=True)
    pitcher_id: Mapped[int] = mapped_column(Integer, index=True)
    model_family: Mapped[str] = mapped_column(String(32))  # frequency | logreg | lightgbm | gru
    feature_version: Mapped[str] = mapped_column(String(32))
    feature_schema: Mapped[dict] = mapped_column(JSONB)
    classes: Mapped[list] = mapped_column(JSONB)
    config: Mapped[dict] = mapped_column(JSONB)
    dataset_fingerprint: Mapped[dict] = mapped_column(JSONB)
    artifact_path: Mapped[str] = mapped_column(Text)
    artifact_sha256: Mapped[str] = mapped_column(String(64))
    calibration: Mapped[dict | None] = mapped_column(JSONB)
    validation_metrics: Mapped[dict | None] = mapped_column(JSONB)
    status: Mapped[str] = mapped_column(String(24), default="candidate")  # candidate | promoted | previous_production | archived
    created_at: Mapped[datetime] = _now()


class EvaluationRun(Base):
    __tablename__ = "evaluation_runs"
    evaluation_run_id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    pitcher_id: Mapped[int] = mapped_column(Integer)
    model_version: Mapped[str] = mapped_column(ForeignKey("model_registry.model_version"))
    kind: Mapped[str] = mapped_column(String(24))  # holdout_benchmark | validation | walk_forward | ablation
    split: Mapped[str] = mapped_column(String(24))
    dataset_definition: Mapped[dict] = mapped_column(JSONB)
    n_samples: Mapped[int] = mapped_column(Integer)
    metrics: Mapped[dict] = mapped_column(JSONB)
    status: Mapped[str] = mapped_column(String(16), default="completed")
    created_at: Mapped[datetime] = _now()


class PitchPrediction(Base):
    __tablename__ = "pitch_predictions"
    prediction_id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    pitch_uid: Mapped[int | None] = mapped_column(ForeignKey("pitches.pitch_uid"), index=True)
    model_version: Mapped[str] = mapped_column(ForeignKey("model_registry.model_version"))
    context: Mapped[str] = mapped_column(String(16))  # challenge | benchmark | simulator | dashboard
    evaluation_run_id: Mapped[int | None] = mapped_column(ForeignKey("evaluation_runs.evaluation_run_id", ondelete="CASCADE"), index=True)
    input_features: Mapped[dict] = mapped_column(JSONB)
    probabilities: Mapped[dict] = mapped_column(JSONB)
    top_pitch: Mapped[str] = mapped_column(String(8))
    attribution: Mapped[dict | None] = mapped_column(JSONB)
    actual_pitch: Mapped[str | None] = mapped_column(String(8))  # filled ONLY after the prediction is frozen
    is_correct_top1: Mapped[bool | None] = mapped_column(Boolean)
    prob_actual: Mapped[float | None] = mapped_column(Float)
    inference_ms: Mapped[float | None] = mapped_column(Float)
    trace_id: Mapped[str | None] = mapped_column(String(36))
    created_at: Mapped[datetime] = _now()
    revealed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class PredictionChallenge(Base):
    __tablename__ = "prediction_challenges"
    challenge_id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    pitcher_id: Mapped[int] = mapped_column(Integer)
    model_version: Mapped[str] = mapped_column(ForeignKey("model_registry.model_version"))
    size: Mapped[int] = mapped_column(Integer)
    sampling_strategy: Mapped[str] = mapped_column(String(32))
    strategy_params: Mapped[dict | None] = mapped_column(JSONB)
    seed: Mapped[int] = mapped_column(BigInteger)
    split: Mapped[str] = mapped_column(String(12))
    pitch_uids: Mapped[list] = mapped_column(JSONB)
    status: Mapped[str] = mapped_column(String(16), default="active")
    created_at: Mapped[datetime] = _now()


class ChallengeAttempt(Base):
    __tablename__ = "challenge_attempts"
    attempt_id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    challenge_id: Mapped[int] = mapped_column(ForeignKey("prediction_challenges.challenge_id", ondelete="CASCADE"))
    position: Mapped[int] = mapped_column(Integer)
    pitch_uid: Mapped[int] = mapped_column(ForeignKey("pitches.pitch_uid"))
    prediction_id: Mapped[int] = mapped_column(ForeignKey("pitch_predictions.prediction_id"))
    video_asset_id: Mapped[int | None] = mapped_column(ForeignKey("video_assets.video_asset_id"))
    mode: Mapped[str] = mapped_column(String(16))  # in_app | external | data_only
    user_guess: Mapped[str | None] = mapped_column(String(8))
    revealed: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = _now()
    revealed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    __table_args__ = (UniqueConstraint("challenge_id", "position", name="uq_attempt_position"),)


# --------------------------------------------------------------------------- jobs + observability
class Job(Base):
    """Postgres-backed job queue (claimed with SELECT ... FOR UPDATE SKIP LOCKED)."""

    __tablename__ = "jobs"
    job_id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    kind: Mapped[str] = mapped_column(String(48), index=True)
    params: Mapped[dict] = mapped_column(JSONB, default=dict)
    status: Mapped[str] = mapped_column(String(16), default="queued", index=True)  # queued|running|completed|failed|cancelled
    progress: Mapped[float] = mapped_column(Float, default=0.0)
    records_processed: Mapped[int] = mapped_column(Integer, default=0)
    message: Mapped[str | None] = mapped_column(Text)
    result: Mapped[dict | None] = mapped_column(JSONB)
    error: Mapped[str | None] = mapped_column(Text)
    attempts: Mapped[int] = mapped_column(SmallInteger, default=0)
    max_attempts: Mapped[int] = mapped_column(SmallInteger, default=2)
    cancel_requested: Mapped[bool] = mapped_column(Boolean, default=False)
    log: Mapped[list] = mapped_column(JSONB, default=list)
    created_at: Mapped[datetime] = _now()
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class PipelineTrace(Base):
    """Real per-request processing stages for the Engineering Console."""

    __tablename__ = "pipeline_traces"
    trace_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    kind: Mapped[str] = mapped_column(String(24))
    model_version: Mapped[str | None] = mapped_column(String(64))
    stages: Mapped[list] = mapped_column(JSONB)
    total_ms: Mapped[float] = mapped_column(Float)
    created_at: Mapped[datetime] = _now()


class SavedScenario(Base):
    __tablename__ = "saved_scenarios"
    scenario_id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    pitcher_id: Mapped[int] = mapped_column(Integer)
    name: Mapped[str] = mapped_column(String(120))
    state: Mapped[dict] = mapped_column(JSONB)
    created_at: Mapped[datetime] = _now()


class DataValidationReport(Base):
    __tablename__ = "data_validation_reports"
    report_id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    pitcher_id: Mapped[int] = mapped_column(Integer)
    scope: Mapped[str] = mapped_column(String(48))
    summary: Mapped[dict] = mapped_column(JSONB)
    issues: Mapped[list] = mapped_column(JSONB)
    created_at: Mapped[datetime] = _now()


from . import models_lifecycle  # noqa: E402,F401  (np-refit lifecycle tables)
from . import models_aging  # noqa: E402,F401  (aging dataset tables)
from . import models_video_cv  # noqa: E402,F401  (video alignment tables)
