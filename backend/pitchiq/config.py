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

    # Chronological split (by season). Adjust after checking sample counts.
    train_seasons: list[int] = [2023, 2024]
    validation_seasons: list[int] = [2025]
    test_seasons: list[int] = [2026]

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
