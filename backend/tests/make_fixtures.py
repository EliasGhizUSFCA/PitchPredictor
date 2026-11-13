"""Regenerate offline test fixtures from REAL cached source data (run after a live ingestion).

Selected games include pitch-clock automatic calls (718010, 745794, 824590) to exercise edge cases.
"""
import json
import sys
from pathlib import Path

import pandas as pd
from sqlalchemy import text

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pitchiq.db import session_scope  # noqa: E402
from pitchiq.http import PoliteClient  # noqa: E402
from pitchiq.ingest import mlb_stats  # noqa: E402

GAMES = [718010, 716478, 745794, 744845, 776149, 776221, 824590, 823204]
OUT = Path(__file__).parent / "fixtures"


def slim_feed(feed):
    plays = []
    for p in feed["liveData"]["plays"]["allPlays"]:
        plays.append({"about": {k: p["about"][k] for k in ("atBatIndex", "inning", "halfInning")},
                      "matchup": {"batter": {"id": p["matchup"]["batter"]["id"]},
                                  "pitcher": {"id": p["matchup"]["pitcher"]["id"]}},
                      "playEvents": [{k: e.get(k) for k in ("isPitch", "playId", "pitchNumber", "startTime", "endTime", "count")}
                                     | {"details": {"type": (e.get("details") or {}).get("type")}}
                                     for e in p.get("playEvents", []) if e.get("isPitch")]})
    teams = {}
    for side in ("home", "away"):
        t = feed["liveData"]["boxscore"]["teams"][side]
        teams[side] = {"team": {"abbreviation": t["team"].get("abbreviation")},
                       "players": {k: {"person": {"id": v["person"]["id"]}, "battingOrder": v.get("battingOrder")}
                                   for k, v in t["players"].items() if v.get("battingOrder")}}
    gteams = {side: {"abbreviation": feed["gameData"]["teams"][side].get("abbreviation")} for side in ("home", "away")}
    return {"gameData": {"venue": feed["gameData"].get("venue"), "status": feed["gameData"].get("status"),
                         "teams": gteams},
            "liveData": {"plays": {"allPlays": plays}, "boxscore": {"teams": teams}}}


with session_scope() as s:
    df = pd.DataFrame([r.raw for r in s.execute(text("SELECT raw FROM pitches WHERE game_pk = ANY(:g)"), {"g": GAMES})])
    batters = [r[0] for r in s.execute(text("SELECT DISTINCT batter_id FROM pitches WHERE game_pk = ANY(:g)"), {"g": GAMES})]
cols = [c for c in df.columns]
df.to_csv(OUT / "statcast_subset.csv", index=False, columns=cols)
c = PoliteClient()
for g in GAMES:
    (OUT / "live_feed" / f"{g}.json").write_text(json.dumps(slim_feed(mlb_stats.fetch_live_feed(c, g))))
lineup_ids = set()
for g in GAMES:
    f = json.loads((OUT / "live_feed" / f"{g}.json").read_text())
    for side in ("home", "away"):
        lineup_ids |= {v["person"]["id"] for v in f["liveData"]["boxscore"]["teams"][side]["players"].values()}
people = mlb_stats.fetch_people(c, sorted(set(batters) | lineup_ids | {519242}))
(OUT / "people.json").write_text(json.dumps({"people": [{k: p.get(k) for k in ("id", "fullName", "batSide", "pitchHand")} for p in people]}))
print(len(df), "pitches;", len(people), "people")
