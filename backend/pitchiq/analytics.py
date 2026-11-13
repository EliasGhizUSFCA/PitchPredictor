"""Observed historical frequencies (NOT model predictions). Every result carries sample sizes + provenance."""
from __future__ import annotations

from collections import Counter, defaultdict

import pandas as pd
from sqlalchemy import text
from sqlalchemy.orm import Session

from .ml.metrics import pitch_count_bucket
from .pitch_types import pitch_name, result_category

PROVENANCE = {"source": "Baseball Savant Statcast pitch-level data (regular season), batting order from MLB Stats "
                        "API boxscores", "kind": "observed_frequency"}

BASE_SQL = """
SELECT p.pitch_uid, p.game_pk, p.game_date, p.season, p.batter_id, b.full_name AS batter_name, p.stand,
       p.balls, p.strikes, p.inning, p.outs_when_up, p.pitch_type, p.release_speed, p.release_spin_rate,
       p.description, p.events, p.at_bat_number, p.pitch_number, p.game_pitch_seq, p.n_thruorder_pitcher,
       pa.batting_order_slot, CASE WHEN p.inning_topbot='Top' THEN p.away_team ELSE p.home_team END AS opponent
FROM pitches p JOIN plate_appearances pa USING (game_pk, at_bat_number) LEFT JOIN batters b ON b.batter_id = p.batter_id
WHERE p.pitcher_id = :pid AND p.pitch_type IS NOT NULL
"""


def load(session: Session, pitcher_id: int, filters: dict) -> pd.DataFrame:
    sql, params = BASE_SQL, {"pid": pitcher_id}
    for key, col in (("season", "p.season"), ("batter_id", "p.batter_id"), ("game_pk", "p.game_pk"),
                     ("stand", "p.stand")):
        if filters.get(key) is not None:
            sql += f" AND {col} = :{key}"
            params[key] = filters[key]
    if filters.get("count"):
        b, s = filters["count"].split("-")
        sql += " AND p.balls = :fb AND p.strikes = :fs"
        params.update(fb=int(b), fs=int(s))
    df = pd.read_sql(text(sql + " ORDER BY p.game_date, p.game_pk, p.game_pitch_seq"), session.connection(), params=params)
    if len(df):
        # cumulative pitch count before this pitch (thrown pitches only)
        df["pc_before"] = df.groupby("game_pk").cumcount()
    return df


def _usage(df: pd.DataFrame, by: str, order: list | None = None) -> dict:
    classes = df["pitch_type"].value_counts().index.tolist()
    rows = []
    keys = order or sorted(df[by].dropna().unique().tolist(), key=lambda x: (str(type(x)), x))
    for k in keys:
        g = df[df[by] == k]
        n = len(g)
        if n == 0:
            continue
        vc = g["pitch_type"].value_counts()
        rows.append({"key": str(k), "n": n, **{c: float(vc.get(c, 0) / n) for c in classes},
                     "counts": {c: int(vc.get(c, 0)) for c in classes}})
    return {"by": by, "classes": classes, "rows": rows}


def pitcher_analytics(session: Session, pitcher_id: int, filters: dict) -> dict:
    df = load(session, pitcher_id, filters)
    if df.empty:
        return {"n": 0, "provenance": PROVENANCE, "filters": filters}
    df["count_str"] = df.balls.astype(str) + "-" + df.strikes.astype(str)
    df["pc_bucket"] = df.pc_before.map(pitch_count_bucket)
    df["inning_g"] = df.inning.clip(upper=9).astype(int).astype(str).replace({"9": "9+"})
    counts_order = [f"{b}-{s}" for b in range(4) for s in range(3)]
    classes = df["pitch_type"].value_counts().index.tolist()

    # Transition matrix (previous pitch within the same PA -> next pitch)
    trans = defaultdict(Counter)
    first = Counter()
    for _, g in df.groupby(["game_pk", "at_bat_number"]):
        seq = g.sort_values("pitch_number")["pitch_type"].tolist()
        if seq:
            first[seq[0]] += 1
        for a, b in zip(seq, seq[1:]):
            trans[a][b] += 1
    tm = {"from": classes, "to": classes, "matrix": [[trans[a][b] for b in classes] for a in classes],
          "row_n": [sum(trans[a].values()) for a in classes], "first_pitch_of_pa": dict(first)}

    # Batter matchups
    bat = []
    for (bid, name, stand), g in df.groupby(["batter_id", "batter_name", "stand"], dropna=False):
        pas = g.groupby(["game_pk", "at_bat_number"]).size().shape[0]
        evs = g.dropna(subset=["events"])["events"]
        vc = g.pitch_type.value_counts()
        bat.append({"batter_id": int(bid), "name": name, "stand": stand, "pitches": len(g), "plate_appearances": pas,
                    "strikeouts": int(evs.isin(["strikeout", "strikeout_double_play"]).sum()),
                    "walks": int(evs.isin(["walk"]).sum()), "hits": int(evs.isin(["single", "double", "triple", "home_run"]).sum()),
                    "usage": {c: float(vc.get(c, 0) / len(g)) for c in classes}})
    bat.sort(key=lambda r: -r["pitches"])

    # Per-game performance / trends
    games = []
    for gpk, g in df.groupby("game_pk", sort=False):
        evs = g.dropna(subset=["events"])["events"]
        vc = g.pitch_type.value_counts()
        ff = g[g.pitch_type == "FF"].release_speed
        games.append({"game_pk": int(gpk), "game_date": str(g.game_date.iloc[0]), "opponent": g.opponent.iloc[0],
                      "pitches": len(g), "batters_faced": int(g.groupby("at_bat_number").ngroups),
                      "strikeouts": int(evs.isin(["strikeout", "strikeout_double_play"]).sum()),
                      "walks": int((evs == "walk").sum()),
                      "hits": int(evs.isin(["single", "double", "triple", "home_run"]).sum()),
                      "whiffs": int(g.description.isin(["swinging_strike", "swinging_strike_blocked"]).sum()),
                      "ff_velo": float(ff.mean()) if len(ff) else None,
                      "usage": {c: float(vc.get(c, 0) / len(g)) for c in classes}})

    arsenal = []
    for c in classes:
        g = df[df.pitch_type == c]
        arsenal.append({"pitch_type": c, "pitch_name": pitch_name(c), "n": len(g), "share": len(g) / len(df),
                        "velo": float(g.release_speed.mean()) if g.release_speed.notna().any() else None,
                        "spin": float(g.release_spin_rate.mean()) if g.release_spin_rate.notna().any() else None,
                        "whiff_rate": float(g.description.isin(["swinging_strike", "swinging_strike_blocked"]).mean())})
    return {
        "n": len(df), "games": int(df.game_pk.nunique()), "filters": filters, "provenance": PROVENANCE,
        "date_range": [str(df.game_date.min()), str(df.game_date.max())], "classes": classes,
        "arsenal": arsenal,
        "by_season": _usage(df, "season"),
        "by_count": _usage(df, "count_str", counts_order),
        "by_hand": _usage(df, "stand"),
        "by_batting_order": _usage(df, "batting_order_slot"),
        "by_inning": _usage(df, "inning_g", [str(i) for i in range(1, 9)] + ["9+"]),
        "by_pitch_count": _usage(df, "pc_bucket", ["0-24", "25-49", "50-74", "75-99", "100+"]),
        "by_times_through_order": _usage(df, "n_thruorder_pitcher"),
        "transitions": tm,
        "batters": bat[:40],
        "games_log": games,
    }


def filter_options(session: Session, pitcher_id: int) -> dict:
    seasons = [r[0] for r in session.execute(text("SELECT DISTINCT season FROM pitches WHERE pitcher_id=:p ORDER BY 1"), {"p": pitcher_id})]
    games = [dict(r._mapping) for r in session.execute(text(
        """SELECT g.game_pk, g.game_date::text AS game_date,
                  CASE WHEN min(p.inning_topbot)='Top' THEN g.away_team ELSE g.home_team END AS opponent, count(*) AS pitches
           FROM games g JOIN pitches p USING (game_pk) WHERE p.pitcher_id=:p GROUP BY g.game_pk ORDER BY g.game_date DESC"""),
        {"p": pitcher_id})]
    batters = [dict(r._mapping) for r in session.execute(text(
        """SELECT b.batter_id, b.full_name AS name, count(*) AS pitches FROM pitches p JOIN batters b USING (batter_id)
           WHERE p.pitcher_id=:p GROUP BY b.batter_id ORDER BY count(*) DESC LIMIT 300"""), {"p": pitcher_id})]
    return {"seasons": seasons, "games": games, "batters": batters,
            "counts": [f"{b}-{s}" for b in range(4) for s in range(3)]}


__all__ = ["pitcher_analytics", "filter_options", "result_category"]
