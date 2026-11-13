"""Pitch <-> video candidate matching with explicit confidence levels.

exact_verified : game_pk, pitcher, batter, at_bat_number, pitch_number all agree AND the pre-pitch count
                 agrees AND the provider's pitch classification agrees with Statcast's (or the provider
                 is a licensed manifest that names the exact pitch keys).
strong_review  : all identifiers agree but a secondary check disagrees or is missing -> human review.
possible       : same game + batter (+ inning) but pitch-level keys disagree/missing.
no_match       : otherwise.
A title string is never used as evidence.
"""
from __future__ import annotations

from dataclasses import dataclass

from .providers.base import VideoCandidate


@dataclass
class PitchKey:
    pitch_uid: int
    game_pk: int
    pitcher_id: int
    batter_id: int
    at_bat_number: int
    pitch_number: int
    inning: int
    balls: int
    strikes: int
    pitch_type: str | None


def match(p: PitchKey, c: VideoCandidate) -> tuple[str, dict]:
    ev = {
        "game_pk": c.game_pk == p.game_pk,
        "pitcher_id": c.pitcher_id == p.pitcher_id,
        "batter_id": c.batter_id == p.batter_id,
        "at_bat_number": c.at_bat_number == p.at_bat_number,
        "pitch_number": c.pitch_number == p.pitch_number,
        "inning": None if c.inning is None else c.inning == p.inning,
        "count": None if c.balls_before is None else (c.balls_before, c.strikes_before) == (p.balls, p.strikes),
        "pitch_type": None if c.feed_pitch_type is None or p.pitch_type is None else c.feed_pitch_type == p.pitch_type,
    }
    ids = all(ev[k] for k in ("game_pk", "pitcher_id", "batter_id", "at_bat_number", "pitch_number"))
    secondary = [ev[k] for k in ("inning", "count", "pitch_type")]
    if ids and c.source_provider == "licensed_local":
        return "exact_verified", ev
    if ids and all(v is True for v in secondary):
        return "exact_verified", ev
    if ids:
        return "strong_review", ev
    if ev["game_pk"] and ev["batter_id"] and ev["inning"] is not False:
        return "possible", ev
    return "no_match", ev
