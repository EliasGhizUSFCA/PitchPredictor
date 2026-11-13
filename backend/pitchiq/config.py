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


@lru_cache
def get_settings() -> Settings:
    return Settings()
