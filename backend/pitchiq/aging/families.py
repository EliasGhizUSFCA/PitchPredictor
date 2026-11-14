"""Pitch-type families, measurement eras and age helpers for the aging dataset.

Official Statcast codes are always stored unchanged. `pitch_family` is an analysis grouping that absorbs
classification changes, most importantly the sweeper (ST) and slurve (SV) codes that Statcast introduced publicly in
2023. Savant serves ST/SV for earlier seasons too (as currently classified), but sparsely: in the stored leaderboard
data ST pitcher-season rows grow from 7 (2015) to 118 (2021), 176 (2022) and 286 (2023), while many pitchers' sliders
move from SL to SL+ST or ST across seasons. Family-level deltas (SL/ST/SV -> "slider") are therefore comparable across
seasons; per-code deltas are not, and aging_observations.reclass_flag marks pairs whose codes inside a family changed.
For a group unit (group:fastball / group:breaking / group:offspeed) reclass_flag is true only when one of its member
families changed codes; a plain arsenal change (adding or dropping a cutter, sinker, ...) is not a reclassification and
is visible as codes_from != codes_to.
`pitch_group` follows Baseball Savant's fastball / breaking / offspeed grouping and also absorbs four-seam <-> sinker
relabelling (both are "fastball").

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
FULL_SEASON_GAMES = 162
SEASON_GAMES: dict[int, int] = {2020: 60}  # scheduled team games where different from 162


def season_games(season: int) -> int:
    return SEASON_GAMES.get(int(season), FULL_SEASON_GAMES)


def scaled_threshold(base: int, season: int) -> int:
    """Workload threshold scaled to the season's length (500 pitches -> 185 in the 60-game 2020 season), so that
    qualification / survival in 2020 means the same share of a season as in a full season."""
    return int(round(base * season_games(season) / FULL_SEASON_GAMES))


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
