"""Which seasons each code path may read.

`Settings.seasons` are the modeling seasons (next-pitch features, splits, training, simulator history, analytics,
video library, ingestion validation). `Settings.career_seasons` are history-only: their pitch-level rows can live in
`pitches` for career/aging analysis (pitchiq.aging) but must never reach a modeling path. Every query of `pitches`
that feeds an existing feature filters with `modeling_seasons()`.
"""
from __future__ import annotations

from .config import get_settings


def modeling_seasons() -> list[int]:
    return sorted(int(s) for s in get_settings().seasons)


def history_only_seasons() -> list[int]:
    m = set(modeling_seasons())
    return sorted(int(s) for s in get_settings().career_seasons if int(s) not in m)


def is_history_only(season: int) -> bool:
    return int(season) not in set(modeling_seasons())
