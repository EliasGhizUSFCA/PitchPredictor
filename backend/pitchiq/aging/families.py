"""Pitch-type families, measurement eras and age helpers for the aging dataset.

Official Statcast codes are always stored unchanged. `pitch_family` is an analysis grouping that absorbs
classification changes, most importantly the 2023 introduction of the sweeper (ST) and slurve (SV) codes, which split
pitches that earlier seasons label SL. `pitch_group` follows Baseball Savant's fastball / breaking / offspeed grouping
and also absorbs four-seam <-> sinker relabelling (both are "fastball").

Savant season leaderboards fold the Statcast knuckle-curve (KC) and slow-curve (CS) codes into CU; this module maps
them to CU's family as well, so leaderboard rows and pitch-level rows agree on the family.
"""
from __future__ import annotations

from datetime import date

PITCH_FAMILY: dict[str, str] = {
    "FF": "four_seam", "FA": "four_seam",
    "SI": "sinker", "FT": "sinker",
    "FC": "cutter",
    "SL": "slider", "ST": "slider", "SV": "slider",
    "CU": "curveball", "KC": "curveball", "CS": "curveball",
    "CH": "changeup",
    "FS": "splitter", "FO": "splitter",
    "SC": "screwball",
    "KN": "knuckleball",
    "EP": "eephus",
}

FAMILY_GROUP: dict[str, str] = {
    "four_seam": "fastball", "sinker": "fastball", "cutter": "fastball",
    "slider": "breaking", "curveball": "breaking",
    "changeup": "offspeed", "splitter": "offspeed", "screwball": "offspeed", "knuckleball": "offspeed",
    "eephus": "offspeed",
}

# Leaderboard codes (Savant pitch-movement / arsenal leaderboards) and the Statcast codes folded into them.
LEADERBOARD_CODE: dict[str, str] = {"KC": "CU", "CS": "CU", "FA": "FF", "FT": "SI"}

TRACKMAN_LAST_SEASON = 2019  # Statcast used TrackMan radar 2015-2019; Hawk-Eye optical tracking from 2020.
SHORT_SEASONS = {2020}  # 60-game season


def pitch_family(code: str | None) -> str:
    if not code:
        return "unknown"
    return PITCH_FAMILY.get(code.upper(), "other")


def pitch_group(code: str | None) -> str:
    return FAMILY_GROUP.get(pitch_family(code), "other")


def leaderboard_code(code: str) -> str:
    return LEADERBOARD_CODE.get(code.upper(), code.upper())


def measurement_era(season: int) -> str:
    return "trackman" if season <= TRACKMAN_LAST_SEASON else "hawkeye"


def era_change(season_from: int, season_to: int) -> bool:
    return measurement_era(season_from) != measurement_era(season_to)


def season_midpoint(season: int) -> date:
    return date(season, 6, 30)


def age_years(birth: date | None, on: date) -> float | None:
    """Exact age in years (days / 365.2425)."""
    if birth is None:
        return None
    return (on - birth).days / 365.2425


def age_completed_years(birth: date | None, on: date) -> int | None:
    """Age in completed years on `on` (the usual 'baseball age' when `on` is June 30)."""
    if birth is None:
        return None
    return on.year - birth.year - ((on.month, on.day) < (birth.month, birth.day))
