"""MLB Stats API (statsapi.mlb.com): player metadata, game live feed, boxscore batting order.

The live feed is also the official source of each pitch's Statcast `playId`, used for video matching.
"""
from __future__ import annotations

from dataclasses import dataclass

from ..http import PoliteClient

STATS = "https://statsapi.mlb.com/api"


def fetch_person(client: PoliteClient, person_id: int) -> dict:
    d = client.get_json(f"{STATS}/v1/people/{person_id}", namespace="people")
    return d["people"][0]


def fetch_people(client: PoliteClient, ids: list[int]) -> list[dict]:
    out = []
    for i in range(0, len(ids), 100):
        chunk = ids[i:i + 100]
        d = client.get_json(f"{STATS}/v1/people", {"personIds": ",".join(map(str, chunk))}, namespace="people")
        out.extend(d.get("people", []))
    return out


def fetch_live_feed(client: PoliteClient, game_pk: int, use_cache: bool = True) -> dict:
    return client.get_json(f"{STATS}/v1.1/game/{game_pk}/feed/live", namespace="live_feed", use_cache=use_cache)


@dataclass
class LineupEntry:
    batter_id: int
    side: str
    team: str
    slot: int
    sub_sequence: int
    raw_code: str


def lineup_from_feed(feed: dict) -> list[LineupEntry]:
    """Boxscore `battingOrder` codes: '301' -> slot 3, first substitute; '300' -> slot 3 starter."""
    out = []
    teams = feed["liveData"]["boxscore"]["teams"]
    gd_teams = (feed.get("gameData") or {}).get("teams") or {}
    for side in ("home", "away"):
        abbrev = (teams[side]["team"].get("abbreviation") or (gd_teams.get(side) or {}).get("abbreviation")
                  or side)
        for key, p in teams[side]["players"].items():
            code = p.get("battingOrder")
            if not code:
                continue
            code = str(code)
            slot, sub = int(code[:-2]), int(code[-2:])
            if 1 <= slot <= 9:
                out.append(LineupEntry(p["person"]["id"], side, abbrev, slot, sub, code))
    return out


@dataclass
class FeedPitch:
    """One pitch event from the official live feed, keyed like Statcast."""
    play_id: str | None
    at_bat_number: int  # Statcast at_bat_number == atBatIndex + 1
    pitch_number: int
    pitcher_id: int
    batter_id: int
    inning: int
    half: str
    balls_before: int | None
    strikes_before: int | None
    pitch_type: str | None
    start_time: str | None
    end_time: str | None


def pitches_from_feed(feed: dict, pitcher_id: int | None = None) -> list[FeedPitch]:
    out: list[FeedPitch] = []
    for play in feed["liveData"]["plays"]["allPlays"]:
        m = play["matchup"]
        if pitcher_id is not None and m["pitcher"]["id"] != pitcher_id:
            continue
        ab = play["about"]["atBatIndex"] + 1
        prev_b, prev_s = 0, 0
        for ev in play.get("playEvents", []):
            if not ev.get("isPitch"):
                continue
            cnt = ev.get("count", {})
            out.append(FeedPitch(
                play_id=ev.get("playId"),
                at_bat_number=ab,
                pitch_number=ev.get("pitchNumber"),
                pitcher_id=m["pitcher"]["id"],
                batter_id=m["batter"]["id"],
                inning=play["about"]["inning"],
                half=play["about"]["halfInning"],
                balls_before=prev_b,
                strikes_before=prev_s,
                pitch_type=((ev.get("details") or {}).get("type") or {}).get("code"),
                start_time=ev.get("startTime"),
                end_time=ev.get("endTime"),
            ))
            # Count AFTER this pitch is reported in ev["count"]; it is the "before" of the next pitch.
            prev_b, prev_s = min(cnt.get("balls", prev_b), 3), min(cnt.get("strikes", prev_s), 2)
    return out
