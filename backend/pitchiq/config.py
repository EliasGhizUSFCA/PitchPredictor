"""Runtime configuration (environment variables prefixed PITCHIQ_)."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="PITCHIQ_", env_file=".env", extra="ignore")

    database_url: str = "postgresql+psycopg://pitchiq:pitchiq@localhost:5432/pitchiq"

    # Proof-of-concept pitcher. Other pitchers can be added via the `pitchers` table + CLI.
    default_pitcher_id: int = 519242  # Chris Sale
    seasons: list[int] = [2023, 2024, 2025, 2026]
    game_types: list[str] = ["R"]  # regular season only
    # History-only seasons (aging step): pitch-level rows for these seasons may sit in `pitches` for career/aging
    # analysis, but every modeling path (features, splits, training, simulator history, analytics, video, validation)
    # reads only `seasons` above. See pitchiq/seasons.py.
    career_seasons: list[int] = [2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022]
    # MLB-wide aging dataset (Savant season leaderboards). Thresholds define low_sample flags and "qualifying" seasons.
    aging_seasons: list[int] = list(range(2015, 2027))
    aging_min_pitch_type_pitches: int = 100
    aging_min_season_pitches: int = 500

    # Chronological split (by season). Adjust after checking sample counts.
    train_seasons: list[int] = [2023, 2024]
    validation_seasons: list[int] = [2025]
    test_seasons: list[int] = [2026]

    # Evaluation roles (np-refit). pre_pitch_states.split keeps train/validation/test; these roles say how a season
    # may be used. development: model selection, tuning, calibration and promotion gates. retrospective_viewed: the
    # former "locked test" season, which has been viewed and is reported only as retrospective evidence.
    development_seasons: list[int] = [2023, 2024, 2025]
    retrospective_viewed_seasons: list[int] = [2026]
    # Rolling-origin development evaluation: expanding window, blocks of whole games inside these seasons.
    dev_eval_seasons: list[int] = [2024, 2025]
    dev_block_games: int = 5

    # Promotion gates. Fixed before any gated promotion was run; changing them is a protocol change.
    gate_max_dev_log_loss_regression: float = 0.005  # candidate dev log loss <= production dev log loss + this
    gate_max_dev_ece: float = 0.05  # top-label ECE of the deployed output (raw or chrono-calibrated) on dev folds
    gate_min_training_samples: int = 2000
    gate_prob_tolerance: float = 1e-6
    # Scheduled retraining checked automatically after each feature rebuild (after_each_start | every_6_starts |
    # weekly | manual = no automatic check). Default chosen on the 2025 (development) policy replay; see docs/np-refit.md.
    auto_retrain_policy: str = "after_each_start"

    data_dir: Path = BACKEND_ROOT / "data"
    artifact_dir: Path = BACKEND_ROOT / "artifacts"

    # HTTP politeness for public MLB endpoints.
    http_user_agent: str = "PitchIQ-research/0.1 (non-commercial analytics)"
    http_min_interval_s: float = 1.0
    http_timeout_s: float = 60.0

    # Video: Mode A (in-app playback) is only enabled for sources an operator has licensed.
    licensed_video_dir: Path | None = None
    video_metadata_stale_days: int = 30

    # Pitch classes rarer than this share of training pitches are grouped as "OTHER" for modeling.
    min_class_share: float = 0.01

    # Experiment tracking. Postgres (model_registry, evaluation_runs) is the source of truth; MLflow is an optional
    # mirror for experiment comparison. Tracking is a no-op when disabled or when mlflow is not installed.
    mlflow_enabled: bool = True
    # Default: a local SQLite store under backend/mlruns (MLflow 3 puts the plain ./mlruns file store in maintenance
    # mode and refuses it unless MLFLOW_ALLOW_FILE_STORE=true). Any MLflow URI works (file:, sqlite:, http:, ...).
    mlflow_tracking_uri: str | None = None
    mlflow_artifact_root: Path | None = None  # default backend/mlruns/artifacts (used for new experiments)
    mlflow_experiment_prefix: str = "pitchiq"  # experiment per pitcher: "<prefix>-<pitcher_id>"
    mlflow_log_model_artifacts: bool = True  # copy the checksummed joblib artifact into the run
    mlflow_register_models: bool = True  # register promoted models in the MLflow model registry (alias "promoted")

    @property
    def mlflow_uri(self) -> str:
        return self.mlflow_tracking_uri or f"sqlite:///{BACKEND_ROOT / 'mlruns' / 'mlflow.db'}"

    @property
    def mlflow_artifact_dir(self) -> Path:
        return Path(self.mlflow_artifact_root or BACKEND_ROOT / "mlruns" / "artifacts")


@lru_cache
def get_settings() -> Settings:
    return Settings()
