"""End-to-end historical ingestion: Statcast -> games/PAs/pitches, MLB Stats API -> people + lineups."""
from __future__ import annotations

import logging
import math
from collections import defaultdict
from datetime import datetime, timezone
from typing import Callable

import pandas as pd
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from ..config import get_settings
from ..http import PoliteClient, SourceUnavailable
from ..logging_setup import log
from ..models import (Batter, DataValidationReport, Game, GameLineup, Pitch, Pitcher,
                      PitchSequence, PlateAppearance)
from ..pitch_types import result_category
from . import mlb_stats, statcast

logger = logging.getLogger(__name__)
Progress = Callable[[float, str, int], None]


def _nz(v):
    if v is None:
        return None
    if isinstance(v, float) and math.isnan(v):
        return None
    if pd.isna(v) if not isinstance(v, (list, dict)) else False:
        return None
    return v


def _int(v):
    v = _nz(v)
    return None if v is None else int(v)


def _float(v):
    v = _nz(v)
    return None if v is None else float(v)


def _upsert(session: Session, model, rows: list[dict], keys: list[str], update: bool = True) -> None:
    if not rows:
        return
    for i in range(0, len(rows), 1000):
        chunk = rows[i:i + 1000]
        stmt = insert(model).values(chunk)
        if update:
            cols = {c: stmt.excluded[c] for c in chunk[0].keys() if c not in keys}
            stmt = stmt.on_conflict_do_update(index_elements=keys, set_=cols) if cols else stmt.on_conflict_do_nothing(index_elements=keys)
        else:
            stmt = stmt.on_conflict_do_nothing(index_elements=keys)
        session.execute(stmt)


def ensure_pitcher(session: Session, client: PoliteClient, pitcher_id: int, target: bool = True) -> Pitcher:
    p = session.get(Pitcher, pitcher_id)
    if p is None:
        person = mlb_stats.fetch_person(client, pitcher_id)
        p = Pitcher(pitcher_id=pitcher_id, full_name=person["fullName"],
                    throws=person.get("pitchHand", {}).get("code"), extra={"mlb_person": person})
        session.add(p)
    p.is_active_target = p.is_active_target or target
    session.flush()
    return p


def ingest_statcast(session: Session, pitcher_id: int, seasons: list[int], progress: Progress | None = None,
                    client: PoliteClient | None = None, use_cache: bool = True) -> dict:
    s = get_settings()
    client = client or PoliteClient()
    pitcher = ensure_pitcher(session, client, pitcher_id)
    frames = []
    for i, season in enumerate(seasons):
        df = statcast.fetch_pitcher_season(client, pitcher_id, season, s.game_types, use_cache=use_cache)
        log(logger, "statcast_season", pitcher_id=pitcher_id, season=season, rows=len(df))
        if progress:
            progress(0.4 * (i + 1) / len(seasons), f"Fetched {season}: {len(df)} pitches", len(df))
        if len(df):
            frames.append(df)
    if not frames:
        return {"pitches": 0}
    df = pd.concat(frames, ignore_index=True)
    df = df[df["pitcher"] == pitcher_id].copy()
    df["game_date"] = pd.to_datetime(df["game_date"]).dt.date
    df = df.sort_values(["game_date", "game_pk", "at_bat_number", "pitch_number"]).reset_index(drop=True)
    df["game_pitch_seq"] = df.groupby("game_pk").cumcount() + 1

    pitcher.seasons = sorted({int(x) for x in df["game_year"].unique()})

    # --- games
    games = df.groupby("game_pk").first().reset_index()
    _upsert(session, Game, [dict(
        game_pk=int(r.game_pk), game_date=r.game_date, season=int(r.game_year), game_type=str(r.game_type),
        home_team=str(r.home_team), away_team=str(r.away_team), source="baseball_savant.statcast_search",
    ) for r in games.itertuples()], ["game_pk"])

    # --- batters (names filled from MLB Stats API below)
    _upsert(session, Batter, [dict(batter_id=int(b), bats=None) for b in df["batter"].unique()], ["batter_id"], update=False)

    # --- plate appearances
    pa_rows = []
    for (gpk, ab), g in df.groupby(["game_pk", "at_bat_number"], sort=False):
        first, last = g.iloc[0], g.iloc[-1]
        pa_rows.append(dict(
            game_pk=int(gpk), at_bat_number=int(ab), pitcher_id=pitcher_id, batter_id=int(first.batter),
            inning=int(first.inning), inning_topbot=str(first.inning_topbot), stand=str(first.stand),
            p_throws=str(first.p_throws), times_through_order=_int(first.get("n_thruorder_pitcher")),
            n_pitches=len(g), event=_nz(last.get("events")),
        ))
    _upsert(session, PlateAppearance, pa_rows, ["game_pk", "at_bat_number"])

    # --- pitches
    raw_cols = list(df.columns)
    rows = []
    for r in df.to_dict("records"):
        raw = {k: (_nz(r[k]) if not isinstance(r[k], (pd.Timestamp,)) else str(r[k])) for k in raw_cols}
        raw["game_date"] = str(r["game_date"])
        rows.append(dict(
            game_pk=int(r["game_pk"]), at_bat_number=int(r["at_bat_number"]), pitch_number=int(r["pitch_number"]),
            game_pitch_seq=int(r["game_pitch_seq"]), pitcher_id=pitcher_id, batter_id=int(r["batter"]),
            catcher_id=_int(r.get("fielder_2")), game_date=r["game_date"], season=int(r["game_year"]),
            stand=str(r["stand"]), p_throws=str(r["p_throws"]), balls=int(r["balls"]), strikes=int(r["strikes"]),
            outs_when_up=int(r["outs_when_up"]), inning=int(r["inning"]), inning_topbot=str(r["inning_topbot"]),
            on_1b=_int(r.get("on_1b")), on_2b=_int(r.get("on_2b")), on_3b=_int(r.get("on_3b")),
            home_score=int(r["home_score"]), away_score=int(r["away_score"]),
            bat_score=_int(r.get("bat_score")), fld_score=_int(r.get("fld_score")),
            home_team=str(r["home_team"]), away_team=str(r["away_team"]),
            n_thruorder_pitcher=_int(r.get("n_thruorder_pitcher")),
            pitch_type=_nz(r.get("pitch_type")), pitch_name=_nz(r.get("pitch_name")),
            release_speed=_float(r.get("release_speed")), release_spin_rate=_float(r.get("release_spin_rate")),
            pfx_x=_float(r.get("pfx_x")), pfx_z=_float(r.get("pfx_z")),
            plate_x=_float(r.get("plate_x")), plate_z=_float(r.get("plate_z")), zone=_int(r.get("zone")),
            description=_nz(r.get("description")), events=_nz(r.get("events")), type=_nz(r.get("type")),
            des=_nz(r.get("des")), raw=raw,
        ))
    _upsert(session, Pitch, rows, ["game_pk", "at_bat_number", "pitch_number"])

    # --- pitch sequences (per PA)
    seq_rows = []
    for (gpk, ab), g in df.groupby(["game_pk", "at_bat_number"], sort=False):
        seq_rows.append(dict(game_pk=int(gpk), at_bat_number=int(ab), pitcher_id=pitcher_id,
                             sequence=[_nz(x) for x in g["pitch_type"]],
                             results=[result_category(_nz(x)) for x in g["description"]]))
    _upsert(session, PitchSequence, seq_rows, ["game_pk", "at_bat_number"])
    session.flush()
    if progress:
        progress(0.5, f"Stored {len(rows)} pitches across {len(games)} games", len(rows))
    return {"pitches": len(rows), "games": int(len(games)), "plate_appearances": len(pa_rows),
            "seasons": pitcher.seasons}


def ingest_game_feeds(session: Session, pitcher_id: int, progress: Progress | None = None,
                      client: PoliteClient | None = None, refetch: bool = False) -> dict:
    """Fetch official live feeds: lineups, batter metadata and per-game validation evidence."""
    client = client or PoliteClient()
    games = session.scalars(
        select(Game).where(Game.game_pk.in_(select(Pitch.game_pk).where(Pitch.pitcher_id == pitcher_id).distinct()))
        .order_by(Game.game_date)
    ).all()
    lineup_rows, failures = [], []
    for i, g in enumerate(games):
        try:
            feed = mlb_stats.fetch_live_feed(client, g.game_pk, use_cache=not refetch)
        except SourceUnavailable as e:
            failures.append({"game_pk": g.game_pk, "error": str(e)})
            continue
        gd = feed.get("gameData", {})
        g.venue = (gd.get("venue") or {}).get("name")
        g.feed_fetched_at = datetime.now(timezone.utc)
        sale_feed_pitches = mlb_stats.pitches_from_feed(feed, pitcher_id)
        g.extra = {**(g.extra or {}), "feed_pitch_count": len(sale_feed_pitches),
                   "feed_status": (gd.get("status") or {}).get("detailedState")}
        for e in mlb_stats.lineup_from_feed(feed):
            lineup_rows.append(dict(game_pk=g.game_pk, batter_id=e.batter_id, team=e.team, side=e.side,
                                    batting_order_slot=e.slot, sub_sequence=e.sub_sequence,
                                    is_starter=e.sub_sequence == 0, raw_code=e.raw_code))
        if progress and i % 5 == 0:
            progress(0.5 + 0.4 * (i + 1) / max(len(games), 1), f"Live feed {i + 1}/{len(games)}", i + 1)

    # Batters appearing in lineups must exist before FK insert.
    all_batters = sorted({r["batter_id"] for r in lineup_rows})
    _upsert(session, Batter, [dict(batter_id=b) for b in all_batters], ["batter_id"], update=False)
    _upsert(session, GameLineup, lineup_rows, ["game_pk", "batter_id"])

    # Batter metadata
    ids = list(session.scalars(select(Batter.batter_id).where(Batter.full_name.is_(None))))
    for person in mlb_stats.fetch_people(client, ids) if ids else []:
        b = session.get(Batter, person["id"])
        if b:
            b.full_name = person.get("fullName")
            b.bats = (person.get("batSide") or {}).get("code")
    session.flush()

    # Propagate batting-order slot onto plate appearances.
    slot = {(r["game_pk"], r["batter_id"]): r["batting_order_slot"] for r in lineup_rows}
    for pa in session.scalars(select(PlateAppearance).where(PlateAppearance.pitcher_id == pitcher_id)):
        pa.batting_order_slot = slot.get((pa.game_pk, pa.batter_id))
    session.flush()
    return {"games": len(games), "lineup_rows": len(lineup_rows), "feed_failures": failures}


def validate(session: Session, pitcher_id: int) -> dict:
    """Automated data validation; persists a report and returns the summary."""
    issues: list[dict] = []
    pitches = session.scalars(select(Pitch).where(Pitch.pitcher_id == pitcher_id)
                              .order_by(Pitch.game_pk, Pitch.game_pitch_seq)).all()
    by_game: dict[int, list[Pitch]] = defaultdict(list)
    for p in pitches:
        by_game[p.game_pk].append(p)
    games = {g.game_pk: g for g in session.scalars(select(Game).where(Game.game_pk.in_(list(by_game))))}
    pas = {(pa.game_pk, pa.at_bat_number): pa for pa in
           session.scalars(select(PlateAppearance).where(PlateAppearance.pitcher_id == pitcher_id))}
    missing_label = 0
    for gpk, ps in by_game.items():
        feed_n = ((games[gpk].extra or {}).get("feed_pitch_count"))
        if feed_n is not None and feed_n != len(ps):
            issues.append({"type": "pitch_count_mismatch", "game_pk": gpk, "statcast": len(ps), "live_feed": feed_n})
        # within-PA pitch numbers contiguous starting at 1; counts consistent
        by_pa = defaultdict(list)
        for p in ps:
            by_pa[p.at_bat_number].append(p)
        for ab, seq in by_pa.items():
            nums = [p.pitch_number for p in seq]
            if nums != list(range(1, len(nums) + 1)):
                issues.append({"type": "pa_pitch_gap", "game_pk": gpk, "at_bat_number": ab, "pitch_numbers": nums})
            if seq[0].balls != 0 or seq[0].strikes != 0:
                issues.append({"type": "pa_not_starting_0_0", "game_pk": gpk, "at_bat_number": ab})
            if pas.get((gpk, ab)) and pas[(gpk, ab)].batting_order_slot is None:
                issues.append({"type": "missing_batting_order", "game_pk": gpk, "at_bat_number": ab,
                               "batter_id": seq[0].batter_id})
        for p in ps:
            if not p.pitch_type:
                missing_label += 1
    summary = {
        "pitches": len(pitches), "games": len(by_game), "issues": len(issues),
        "missing_pitch_type": missing_label,
        "by_type": {t: sum(1 for i in issues if i["type"] == t) for t in {i["type"] for i in issues}},
        "seasons": {str(s): sum(1 for p in pitches if p.season == s) for s in sorted({p.season for p in pitches})},
    }
    session.add(DataValidationReport(pitcher_id=pitcher_id, scope="ingestion", summary=summary, issues=issues[:2000]))
    session.flush()
    log(logger, "validation", **summary)
    return summary
