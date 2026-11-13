"""Official Statcast pitch type codes (https://baseballsavant.mlb.com/csv-docs)."""
PITCH_TYPE_NAMES: dict[str, str] = {
    "FF": "Four-Seam Fastball",
    "SI": "Sinker",
    "FC": "Cutter",
    "SL": "Slider",
    "ST": "Sweeper",
    "SV": "Slurve",
    "CU": "Curveball",
    "KC": "Knuckle Curve",
    "CS": "Slow Curve",
    "CH": "Changeup",
    "FS": "Split-Finger",
    "FO": "Forkball",
    "SC": "Screwball",
    "KN": "Knuckleball",
    "EP": "Eephus",
    "FA": "Fastball",
    "PO": "Pitchout",
    "IN": "Intentional Ball",
    "AB": "Automatic Ball",
    "OTHER": "Other (grouped rare pitches)",
}

# Non-competitive / non-pitch rows that are excluded from modeling.
NON_MODEL_PITCH_TYPES = {"PO", "IN", "AB", "", None}


def pitch_name(code: str | None) -> str:
    if not code:
        return "Unknown"
    return PITCH_TYPE_NAMES.get(code, code)


def result_category(description: str | None) -> str:
    """Collapse Statcast `description` into a compact previous-pitch outcome category."""
    d = (description or "").lower()
    if d in ("ball", "blocked_ball", "pitchout", "intent_ball", "automatic_ball"):
        return "ball"
    if d == "hit_by_pitch":
        return "hbp"
    if d in ("called_strike", "automatic_strike"):
        return "called_strike"
    if d in ("swinging_strike", "swinging_strike_blocked", "missed_bunt", "foul_tip"):
        return "swinging_strike"
    if d in ("foul", "foul_bunt", "bunt_foul_tip", "foul_pitchout"):
        return "foul"
    if d.startswith("hit_into_play"):
        return "in_play"
    return "other" if d else "none"
