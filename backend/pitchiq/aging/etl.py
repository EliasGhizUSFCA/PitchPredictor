"""MLB-wide longitudinal pitching ETL for aging analysis (`pitchiq aging-etl`, job kind `aging_etl`).

Steps (all requests through PoliteClient; nothing is synthesised):

1. For each season: Savant pitch-movement (per pitch type), pitch-arsenals (speed / spin / usage), pitch-arsenal-stats,
   custom leaderboard (command proxies, workload), grouped Statcast search per season (release extension, release
   position, arm angle) and per pitch type (extension, whiffs / swings per type - fills 2015-2016, where
   pitch-arsenal-stats has no data).
2. Merge sources per (pitcher, season, pitch type) and per (pitcher, season) with an explicit precedence; every
   output column records which source supplied it (`field_sources`) and the exact URL (`source_urls`).
3. MLB Stats API bulk people -> pitcher_bio (birth date, throws, primary position). Age on June 30 of each season is
   computed from the birth date (never typed in).
4. Optional: the target pitcher's pitch-level Statcast for history-only seasons into `pitches` via the existing
   ingestion pipeline. Those seasons are excluded from every modeling path (pitchiq/seasons.py); the `pitch_ages`
   view gives each pitch's exact age from pitcher_bio.
5. Rebuild aging_observations (consecutive-season pairs incl. non-survivors) and aging_cohorts from everything stored.
6. Coverage and cross-source validation are stored on the aging_etl_runs row.
"""
from __future__ import annotations

import logging
import math
from collections import Counter
from datetime import date, datetime, timezone
from typing import Callable

import numpy as np
import pandas as pd
from sqlalchemy import delete, select, text
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from ..config import get_settings
from ..http import PoliteClient
from ..logging_setup import log
from ..models_aging import (AgingCohort, AgingEtlRun, AgingObservation, PitcherBio, PitcherSeasonPitchStats,
                            PitcherSeasonStats)
from . import sources as src
from .families import (FAMILY_GROUP, SHORT_SEASONS, age_completed_years, age_years, era_change, measurement_era,
                       pitch_family, pitch_group, scaled_threshold, season_midpoint)

logger = logging.getLogger(__name__)
Progress = Callable[[float, str, int], None]

UNITS = {
    "pitches": "count", "total_pitches": "count", "pitch_count": "count", "outs": "count", "batters_faced": "count",
    "games": "count", "games_started": "count", "swings": "count", "whiffs": "count",
    "ip": "innings (outs/3)", "usage_pct": "% of the pitcher's pitches (0-100)",
    "avg_speed_mph": "mph (release speed)", "avg_spin_rpm": "rpm",
    "break_z_in": "inches, vertical break incl. gravity (Savant pitcher_break_z)",
    "ivb_in": "inches, induced vertical break (Savant pitcher_break_z_induced)",
    "break_x_in": "inches, horizontal break (Savant pitcher_break_x convention)",
    "break_z_vs_league_in": "inches vs league average for the pitch type (Savant diff_z)",
    "break_x_vs_league_in": "inches vs league average for the pitch type (Savant diff_x)",
    "extension_ft": "feet", "release_pos_x_ft": "feet (catcher's view)", "release_pos_z_ft": "feet",
    "arm_angle_deg": "degrees (Savant arm angle, 2020+)",
    "k_pct": "% of batters faced", "bb_pct": "% of batters faced", "whiff_pct": "% of swings",
    "put_away_pct": "% of 2-strike pitches ending in a strikeout", "in_zone_pct": "% of pitches",
    "oz_swing_pct": "% of out-of-zone pitches swung at (chase)", "z_swing_pct": "% of in-zone pitches swung at",
    "swing_pct": "% of pitches", "edge_pct": "% of pitches", "f_strike_pct": "% of PAs", "meatball_pct": "% of pitches",
    "iz_contact_pct": "% of in-zone swings", "oz_contact_pct": "% of out-of-zone swings",
    "run_value": "runs (pitcher perspective, positive = good for pitcher)", "run_value_per_100": "runs per 100 pitches",
    "woba": "wOBA", "xwoba": "expected wOBA", "hard_hit_pct": "% of batted balls >= 95 mph", "era": "earned runs / 9",
    "fastball_avg_speed_mph": "mph", "fastball_avg_spin_rpm": "rpm", "breaking_avg_speed_mph": "mph",
    "breaking_avg_spin_rpm": "rpm", "offspeed_avg_speed_mph": "mph", "offspeed_avg_spin_rpm": "rpm",
    "age_june30": "completed years on June 30", "age_june30_exact": "years on June 30 (days/365.2425)",
}

FAMILY_METRICS = ["usage_pct", "avg_speed_mph", "avg_spin_rpm", "ivb_in", "break_z_in", "break_x_in", "whiff_pct",
                  "run_value_per_100", "extension_ft"]
ALL_METRICS = ["pitch_count", "ip", "k_pct", "bb_pct", "in_zone_pct", "oz_swing_pct", "z_swing_pct", "whiff_pct",
               "f_strike_pct", "edge_pct", "fastball_avg_speed_mph", "extension_ft", "release_pos_z_ft",
               "release_pos_x_ft", "arm_angle_deg", "xwoba"]
COHORT_SUMMARY = [("ALL", "fastball_avg_speed_mph"), ("ALL", "k_pct"), ("ALL", "bb_pct"), ("ALL", "in_zone_pct"),
                  ("ALL", "oz_swing_pct"), ("ALL", "whiff_pct"), ("ALL", "extension_ft"), ("ALL", "ip"),
                  ("group:fastball", "avg_speed_mph"), ("four_seam", "avg_speed_mph"), ("four_seam", "avg_spin_rpm"),
                  ("sinker", "avg_speed_mph"), ("slider", "avg_speed_mph"), ("changeup", "avg_speed_mph")]

PS_PRECEDENCE = {  # output column -> [(source column, source key)], first non-null wins
    "pitches": [("mv_pitches", "movement"), ("as_pitches", "arsenal_stats"), ("gs_pitches", "search")],
    "total_pitches": [("mv_total_pitches", "movement"), ("gs_total_pitches", "search")],
    "usage_pct": [("ars_usage", "arsenals"), ("as_usage", "arsenal_stats"), ("mv_usage", "movement")],
    "avg_speed_mph": [("mv_avg_speed", "movement"), ("ars_avg_speed", "arsenals"), ("gs_velocity", "search")],
    "avg_spin_rpm": [("ars_avg_spin", "arsenals"), ("gs_spin", "search")],
    "break_z_in": [("break_z_in", "movement")], "ivb_in": [("ivb_in", "movement")],
    "break_x_in": [("break_x_in", "movement")],
    "break_z_vs_league_in": [("break_z_vs_league_in", "movement")],
    "break_x_vs_league_in": [("break_x_vs_league_in", "movement")],
    "extension_ft": [("gs_extension", "search")], "release_pos_x_ft": [("gs_pos_x", "search")],
    "release_pos_z_ft": [("gs_pos_z", "search")], "arm_angle_deg": [("gs_arm_angle", "search")],
    "swings": [("gs_swings", "search")], "whiffs": [("gs_whiffs", "search")],
    "whiff_pct": [("as_whiff", "arsenal_stats"), ("gs_whiff_pct", "search")],
    "k_pct": [("as_k", "arsenal_stats")], "put_away_pct": [("as_put_away", "arsenal_stats")],
    "run_value": [("as_run_value", "arsenal_stats")], "run_value_per_100": [("as_rv100", "arsenal_stats")],
    "batters_faced": [("as_pa", "arsenal_stats")], "woba": [("as_woba", "arsenal_stats")],
    "xwoba": [("as_xwoba", "arsenal_stats")], "hard_hit_pct": [("as_hard_hit", "arsenal_stats")],
}
SEASON_PRECEDENCE = {
    "games": [("cu_p_game", "custom")], "games_started": [("cu_p_starting_p", "custom")],
    "outs": [("cu_p_out", "custom")], "batters_faced": [("cu_pa", "custom"), ("gs_pa", "search")],
    "pitch_count": [("cu_pitch_count", "custom"), ("gs_pitches", "search")],
    "k_pct": [("cu_k_percent", "custom"), ("gs_k", "search")], "bb_pct": [("cu_bb_percent", "custom"), ("gs_bb", "search")],
    "in_zone_pct": [("cu_in_zone_percent", "custom")], "oz_swing_pct": [("cu_oz_swing_percent", "custom")],
    "z_swing_pct": [("cu_z_swing_percent", "custom")], "swing_pct": [("cu_swing_percent", "custom")],
    "whiff_pct": [("cu_whiff_percent", "custom"), ("gs_whiff_pct", "search")],
    "edge_pct": [("cu_edge_percent", "custom")], "f_strike_pct": [("cu_f_strike_percent", "custom")],
    "meatball_pct": [("cu_meatball_percent", "custom")], "iz_contact_pct": [("cu_iz_contact_percent", "custom")],
    "oz_contact_pct": [("cu_oz_contact_percent", "custom")],
    "fastball_avg_speed_mph": [("cu_fastball_avg_speed", "custom")],
    "fastball_avg_spin_rpm": [("cu_fastball_avg_spin", "custom")],
    "breaking_avg_speed_mph": [("cu_breaking_avg_speed", "custom")],
    "breaking_avg_spin_rpm": [("cu_breaking_avg_spin", "custom")],
    "offspeed_avg_speed_mph": [("cu_offspeed_avg_speed", "custom")],
    "offspeed_avg_spin_rpm": [("cu_offspeed_avg_spin", "custom")],
    "extension_ft": [("gs_extension", "search")], "release_pos_x_ft": [("gs_pos_x", "search")],
    "release_pos_z_ft": [("gs_pos_z", "search")],
    "arm_angle_deg": [("cu_arm_angle", "custom"), ("gs_arm_angle", "search")],
    "xwoba": [("cu_xwoba", "custom")], "era": [("cu_p_era", "custom")],
}
def metric_unit(metric: str) -> str | None:
    """Short unit code for aging_observations.unit (full descriptions in UNITS)."""
    for suf, u in (("_mph", "mph"), ("_rpm", "rpm"), ("_in", "in"), ("_ft", "ft"), ("_pct", "pct"), ("_deg", "deg")):
        if metric.endswith(suf):
            return u
    return {"pitch_count": "count", "ip": "innings", "xwoba": "woba", "run_value_per_100": "runs_per_100"}.get(metric)


INT_COLS = {"pitches", "total_pitches", "swings", "whiffs", "batters_faced", "games", "games_started", "outs",
            "pitch_count"}


# --------------------------------------------------------------------------- helpers
def _clean(v):
    if v is None or v is pd.NA or v is pd.NaT:
        return None
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating, float)):
        f = float(v)
        return None if math.isnan(f) or math.isinf(f) else f
    if isinstance(v, (np.bool_,)):
        return bool(v)
    if isinstance(v, pd.Timestamp):
        return v.to_pydatetime()
    return v


def _isnull(v) -> bool:
    return _clean(v) is None


def _coalesce(df: pd.DataFrame, cands: list[tuple[str, str]]) -> tuple[pd.Series, pd.Series]:
    val = pd.Series(np.nan, index=df.index, dtype="float64")
    who = pd.Series(None, index=df.index, dtype="object")
    for col, key in cands:
        if col not in df:
            continue
        c = pd.to_numeric(df[col], errors="coerce")
        m = val.isna() & c.notna()
        val[m] = c[m]
        who[m] = key
    return val, who


def _ip_formatted_to_outs(s) -> int | None:
    if _isnull(s):
        return None
    try:
        whole, _, frac = str(s).partition(".")
        return int(whole) * 3 + (int(frac) if frac else 0)
    except ValueError:
        return None


def _chunk(rows: list[dict]) -> int:
    return max(1, 30000 // max(1, len(rows[0]))) if rows else 1  # stay well below 65535 bind parameters


def _upsert_rows(session: Session, model, rows: list[dict], keys: list[str]) -> None:
    k = _chunk(rows)
    for i in range(0, len(rows), k):
        chunk = rows[i:i + k]
        stmt = insert(model).values(chunk)
        cols = {c: stmt.excluded[c] for c in chunk[0] if c not in keys}
        session.execute(stmt.on_conflict_do_update(index_elements=keys, set_=cols))


def _insert_rows(session: Session, model, rows: list[dict]) -> None:
    k = _chunk(rows)
    for i in range(0, len(rows), k):
        session.execute(insert(model).values(rows[i:i + k]))


# --------------------------------------------------------------------------- fetch + merge (per season)
def fetch_season(client: PoliteClient, season: int, logs: dict[str, src.SourceLog], *, use_cache: bool = True,
                 search_pitch_types: bool = True, game_types=("R",)) -> dict:
    mv = [src.fetch_movement(client, season, t, logs["movement"], use_cache) for t in src.MOVEMENT_TYPES]
    ars = [src.fetch_arsenals(client, season, k, logs["arsenals"], use_cache) for k in src.ARSENAL_KINDS]
    ast = src.fetch_arsenal_stats(client, season, logs["arsenal_stats"], use_cache)
    cu = src.fetch_custom(client, season, logs["custom"], use_cache)
    gs_all = src.fetch_search(client, season, None, logs["search"], game_types, use_cache)
    gs_types = [src.fetch_search(client, season, t, logs["search"], game_types, use_cache)
                for t in (src.SEARCH_TYPES if search_pitch_types else [])]
    return {"movement": [f for f in mv if f], "arsenals": [f for f in ars if f], "arsenal_stats": ast,
            "custom": cu, "search_all": gs_all, "search_types": [f for f in gs_types if f]}


def merge_pitch_type_rows(season: int, fetched: dict) -> pd.DataFrame:
    keys = ["pitcher_id", "season", "pitch_type"]
    parts = [p for p in (pd.concat([src.movement_long(f) for f in fetched["movement"]], ignore_index=True)
                         if fetched["movement"] else pd.DataFrame(),
                         src.arsenals_long(fetched["arsenals"]),
                         src.arsenal_stats_long(fetched["arsenal_stats"]) if fetched["arsenal_stats"] else pd.DataFrame(),
                         src.search_long(fetched["search_types"])) if not p.empty]
    if not parts:
        return pd.DataFrame()
    df = parts[0]
    for p in parts[1:]:
        df = df.merge(p, on=keys, how="outer")
    if "gs_whiffs" in df:
        sw = pd.to_numeric(df["gs_swings"], errors="coerce")
        df["gs_whiff_pct"] = np.where(sw > 0, 100.0 * pd.to_numeric(df["gs_whiffs"], errors="coerce") / sw, np.nan)
    for out, cands in PS_PRECEDENCE.items():
        df[out], df["fs_" + out] = _coalesce(df, cands)
    return df


def merge_season_rows(season: int, fetched: dict) -> pd.DataFrame:
    cu = src.custom_frame(fetched["custom"]) if fetched["custom"] else pd.DataFrame()
    gs = src.search_season_frame(fetched["search_all"]) if fetched["search_all"] else pd.DataFrame()
    if cu.empty and gs.empty:
        return pd.DataFrame()
    df = cu.merge(gs, on=["pitcher_id", "season"], how="outer") if not cu.empty and not gs.empty \
        else (cu if not cu.empty else gs)
    if "gs_whiffs" in df:
        sw = pd.to_numeric(df["gs_swings"], errors="coerce")
        df["gs_whiff_pct"] = np.where(sw > 0, 100.0 * pd.to_numeric(df["gs_whiffs"], errors="coerce") / sw, np.nan)
    for out, cands in SEASON_PRECEDENCE.items():
        df[out], df["fs_" + out] = _coalesce(df, cands)
    return df


# --------------------------------------------------------------------------- bios
def ensure_bios(session: Session, client: PoliteClient, ids: set[int], log_: src.SourceLog,
                refresh: bool = False, use_cache: bool = True) -> dict:
    have = set(session.scalars(select(PitcherBio.pitcher_id).where(PitcherBio.birth_date.is_not(None))))
    want = sorted(ids if refresh else ids - have)
    people = src.fetch_people(client, want, log_, use_cache=use_cache) if want else []
    rows = []
    for p in people:
        rows.append(dict(
            pitcher_id=int(p["id"]), full_name=p.get("fullName"),
            birth_date=date.fromisoformat(p["birthDate"]) if p.get("birthDate") else None,
            throws=(p.get("pitchHand") or {}).get("code"),
            primary_position=(p.get("primaryPosition") or {}).get("abbreviation"),
            birth_country=p.get("birthCountry"),
            mlb_debut_date=date.fromisoformat(p["mlbDebutDate"]) if p.get("mlbDebutDate") else None,
            source="mlb_stats_api.people", source_url=p["_source_url"], fetched_at=p["_fetched_at"],
            updated_at=datetime.now(timezone.utc)))
    if rows:
        _upsert_rows(session, PitcherBio, rows, ["pitcher_id"])
    session.flush()
    returned = {r["pitcher_id"] for r in rows}
    return {"requested": len(want), "returned": len(returned), "not_returned": sorted(set(want) - returned)[:200],
            "already_present": len(ids & have) if not refresh else 0}


def _bio_map(session: Session) -> dict[int, PitcherBio]:
    return {b.pitcher_id: b for b in session.scalars(select(PitcherBio))}


# --------------------------------------------------------------------------- rows for the two stats tables
def _row_urls(r, keys: list[str]) -> tuple[dict, datetime | None]:
    urls, fats = {}, []
    for k in keys:
        u = r.get("src_" + k)
        if isinstance(u, list) and len(u) == 1:
            u = u[0]
        if not _isnull(u) if not isinstance(u, list) else bool(u):
            urls[k] = u
            f = r.get("fat_" + k)
            if not _isnull(f):
                fats.append(pd.Timestamp(f).to_pydatetime())
    return urls, (min(fats) if fats else None)


def pitch_type_rows(df: pd.DataFrame, bios: dict, min_pitches: int, run_id: int, stats: Counter) -> list[dict]:
    rows = []
    now = datetime.now(timezone.utc)
    for r in df.to_dict("records"):
        season, code = int(r["season"]), str(r["pitch_type"])
        out = {c: _clean(r.get(c)) for c in PS_PRECEDENCE}
        for c in INT_COLS & out.keys():
            out[c] = None if out[c] is None else int(round(out[c]))
        flags = []
        n = out["pitches"]
        low = n is None or n < min_pitches
        if low:
            flags.append("low_sample")
        counts = {k: _clean(r.get(c)) for c, k in (("mv_pitches", "movement"), ("as_pitches", "arsenal_stats"),
                                                     ("gs_pitches", "search"))}
        cv = [v for v in counts.values() if v is not None]
        if len(cv) >= 2:
            stats["pitch_count_compared"] += 1
            if max(cv) - min(cv) > max(2, 0.02 * max(cv)):
                flags.append("pitch_count_sources_disagree")
                stats["pitch_count_disagree"] += 1
        sp = (_clean(r.get("mv_avg_speed")), _clean(r.get("ars_avg_speed")))
        if None not in sp:
            stats["speed_compared"] += 1
            if abs(sp[0] - sp[1]) > 0.25:
                flags.append("speed_sources_disagree")
                stats["speed_disagree"] += 1
        if _isnull(r.get("mv_pitches")):
            flags.append("no_movement_row")
        if season < src.ARSENAL_STATS_FIRST_SEASON:
            flags.append("arsenal_stats_unavailable_season")
        if season in SHORT_SEASONS:
            flags.append("short_season")
        b = bios.get(int(r["pitcher_id"]))
        if b is not None and b.primary_position not in (None, "P", "TWP"):
            flags.append("position_player")
        dups = {k: r.get("dup_" + k) for k in ("movement", "arsenals", "arsenal_stats", "search")
                if isinstance(r.get("dup_" + k), dict)}
        if dups:
            flags.append("multiple_source_rows_per_key")  # e.g. a switch-pitcher listed once per hand
            stats["pitch_type_duplicate_keys"] += 1
        urls, fat = _row_urls(r, ["movement", "arsenals", "arsenal_stats", "search"])
        vbs = {"movement": {"pitches": counts["movement"], "avg_speed": sp[0]},
               "arsenals": {"avg_speed": sp[1], "avg_spin": _clean(r.get("ars_avg_spin")),
                            "usage_pct": _clean(r.get("ars_usage"))},
               "arsenal_stats": {"pitches": counts["arsenal_stats"], "usage_pct": _clean(r.get("as_usage")),
                                 "whiff_pct": _clean(r.get("as_whiff"))},
               "search": {"pitches": counts["search"], "velocity": _clean(r.get("gs_velocity")),
                          "spin": _clean(r.get("gs_spin")), "whiff_pct": _clean(r.get("gs_whiff_pct")),
                          "codes": r.get("search_codes") if isinstance(r.get("search_codes"), list) else None}}
        vbs = {k: {kk: vv for kk, vv in v.items() if vv is not None} for k, v in vbs.items()}
        if dups:
            vbs["duplicates"] = dups
        name = r.get("pitch_name") if not _isnull(r.get("pitch_name")) else r.get("as_pitch_name")
        rows.append(dict(
            pitcher_id=int(r["pitcher_id"]), season=season, pitch_type=code, pitch_family=pitch_family(code),
            pitch_group=pitch_group(code), pitch_name=None if _isnull(name) else str(name)[:40],
            pitch_hand=None if _isnull(r.get("pitch_hand")) else str(r["pitch_hand"])[:1], **out,
            measurement_era=measurement_era(season), low_sample=low, quality_flags=flags,
            values_by_source={k: v for k, v in vbs.items() if v},
            field_sources={c: r["fs_" + c] for c in PS_PRECEDENCE if not _isnull(r.get("fs_" + c))},
            source_urls=urls, fetched_at=fat, etl_run_id=run_id, updated_at=now))
    return rows


def season_rows(df: pd.DataFrame, hand_from_types: dict, bios: dict, min_season_pitches: int, run_id: int,
                stats: Counter) -> list[dict]:
    rows = []
    now = datetime.now(timezone.utc)
    for r in df.to_dict("records"):
        pid, season = int(r["pitcher_id"]), int(r["season"])
        out = {c: _clean(r.get(c)) for c in SEASON_PRECEDENCE}
        for c in INT_COLS & out.keys():
            out[c] = None if out[c] is None else int(round(out[c]))
        flags = []
        outs = out.pop("outs")
        fmt_outs = _ip_formatted_to_outs(r.get("cu_p_formatted_ip"))
        if outs is None:
            outs = fmt_outs
        elif fmt_outs is not None:
            stats["ip_compared"] += 1
            if fmt_outs != outs:
                flags.append("ip_formatted_vs_outs_mismatch")
                stats["ip_mismatch"] += 1
        ip = None if outs is None else outs / 3.0
        pc = (_clean(r.get("cu_pitch_count")), _clean(r.get("gs_pitches")))
        if None not in pc:
            stats["season_pitch_count_compared"] += 1
            if abs(pc[0] - pc[1]) > max(2, 0.01 * pc[0]):
                flags.append("pitch_count_sources_disagree")
                stats["season_pitch_count_disagree"] += 1
        if _isnull(r.get("cu_pitch_count")):
            flags.append("no_custom_row")
        if _isnull(r.get("gs_pitches")):
            flags.append("no_search_row")
        n = out["pitch_count"]
        low = n is None or n < min_season_pitches
        if low:
            flags.append("low_sample")
        if season in SHORT_SEASONS:
            flags.append("short_season")
        b = bios.get(pid)
        birth = b.birth_date if b else None
        age_int = age_completed_years(birth, season_midpoint(season))
        age_exact = age_years(birth, season_midpoint(season))
        sav_age = _clean(r.get("cu_player_age"))
        if birth is None:
            flags.append("no_birth_date")
        elif sav_age is not None:
            stats["age_compared"] += 1
            if int(sav_age) != age_int:
                flags.append("savant_age_differs")
                stats["age_differs"] += 1
        if b is not None and b.primary_position not in (None, "P", "TWP"):
            flags.append("position_player")
        hand = r.get("cu_pitch_hand")
        hand = hand if not _isnull(hand) else hand_from_types.get((pid, season)) or (b.throws if b else None)
        name = r.get("cu_player_name") if not _isnull(r.get("cu_player_name")) else r.get("gs_player_name")
        if any(isinstance(r.get("dup_" + k), dict) for k in ("custom", "search")):
            flags.append("multiple_source_rows_per_key")  # e.g. a switch-pitcher listed once per hand
            stats["season_duplicate_keys"] += 1
        urls, fat = _row_urls(r, ["custom", "search"])
        rows.append(dict(
            pitcher_id=pid, season=season, player_name=None if _isnull(name) else str(name)[:120],
            pitch_hand=None if _isnull(hand) else str(hand)[:1], age_june30=age_int,
            age_june30_exact=age_exact, savant_player_age=None if sav_age is None else int(sav_age),
            ip=ip, outs=outs, **out, measurement_era=measurement_era(season), low_sample=low, quality_flags=flags,
            field_sources={c: r["fs_" + c] for c in SEASON_PRECEDENCE if not _isnull(r.get("fs_" + c))},
            source_urls=urls, fetched_at=fat, etl_run_id=run_id, updated_at=now))
    return rows


# --------------------------------------------------------------------------- observations + cohorts
def _weighted(df: pd.DataFrame, by: list[str], metric: str, weight: str = "pitches") -> pd.Series:
    w = pd.to_numeric(df[weight], errors="coerce").fillna(0)
    x = pd.to_numeric(df[metric], errors="coerce")
    m = x.notna() & (w > 0)
    tmp = pd.DataFrame({"wx": (x * w).where(m, 0.0), "w": w.where(m, 0.0)})
    for b in by:
        tmp[b] = df[b].values
    g = tmp.groupby(by)[["wx", "w"]].sum()
    return (g["wx"] / g["w"]).where(g["w"] > 0)


def build_units(psps: pd.DataFrame, pss: pd.DataFrame) -> pd.DataFrame:
    """(pitcher, season, unit) frame with n, codes and metric values for families, groups and ALL."""
    frames = []
    if not psps.empty:
        d = psps.copy()
        d["pitches"] = pd.to_numeric(d["pitches"], errors="coerce").fillna(0)
        d = d[d["pitches"] > 0]
        for col, prefix in (("pitch_family", ""), ("pitch_group", "group:")):
            d["unit"] = prefix + d[col].astype(str)
            by = ["pitcher_id", "season", "unit"]
            g = d.groupby(by).agg(n=("pitches", "sum"), usage_pct=("usage_pct", lambda s: s.sum(min_count=1)),
                                  codes=("pitch_type", lambda s: sorted(set(s))),
                                  whiffs=("whiffs", lambda s: s.sum(min_count=1)),
                                  swings=("swings", lambda s: s.sum(min_count=1)),
                                  n_swing_rows=("swings", "count"), n_rows=("pitch_type", "size"))
            for m in FAMILY_METRICS:
                if m in ("usage_pct",):
                    continue
                g[m] = _weighted(d, by, m)
            # whiff% by summed whiffs/swings when every code in the unit has swing counts, else pitch-weighted.
            exact = (g["n_swing_rows"] == g["n_rows"]) & (g["swings"] > 0)
            g.loc[exact, "whiff_pct"] = 100.0 * g.loc[exact, "whiffs"] / g.loc[exact, "swings"]
            frames.append(g.drop(columns=["whiffs", "swings", "n_swing_rows", "n_rows"]).reset_index())
    if not pss.empty:
        a = pss.copy()
        a["unit"] = "ALL"
        a["n"] = pd.to_numeric(a["pitch_count"], errors="coerce")
        a["codes"] = None
        frames.append(a[["pitcher_id", "season", "unit", "n", "codes"] + ALL_METRICS])
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def build_observations(units: pd.DataFrame, pss: pd.DataFrame, bios: dict, min_pitches: int,
                       min_season_pitches: int, run_id: int | None) -> list[dict]:
    if units.empty or pss.empty:
        return []
    last = int(pss["season"].max())
    covered = {int(x) for x in pss["season"].unique()}
    pc = pss.set_index(["pitcher_id", "season"])
    workload = pd.to_numeric(pc["pitch_count"], errors="coerce")
    ipser = pd.to_numeric(pc["ip"], errors="coerce")
    # Qualification (at risk in t, survived into t+1, returned later) uses the season-length-scaled threshold, so a
    # full-workload pitcher in the 60-game 2020 season (500 -> 185 pitches) is not counted as a dropout.
    season_idx = pd.Series(pc.index.get_level_values("season"), index=pc.index)
    qual = workload >= season_idx.map(lambda x: scaled_threshold(min_season_pitches, int(x)))
    qual_seasons: dict[int, list[int]] = {}
    for (pid, s), q in qual.items():
        if q:
            qual_seasons.setdefault(int(pid), []).append(int(s))
    for v in qual_seasons.values():
        v.sort()

    recs = {(int(r["pitcher_id"]), r["unit"], int(r["season"])): r for r in units.to_dict("records")}

    def _codes(r):
        return r["codes"] if r is not None and isinstance(r["codes"], list) else None

    def _family_reclass(pid: int, fam: str, season: int) -> bool:
        a, b = _codes(recs.get((pid, fam, season))), _codes(recs.get((pid, fam, season + 1)))
        return bool(a and b and set(a) != set(b))
    rows = []
    now = datetime.now(timezone.utc)
    for (pid, unit, season), rec in recs.items():
        # Survival is only defined when season t+1 was fetched (a missing season must not look like a dropout).
        if season >= last or (season + 1) not in covered or not bool(qual.get((pid, season), False)):
            continue
        n_from = rec["n"]
        base_thr = min_season_pitches if unit == "ALL" else min_pitches
        if _isnull(n_from) or n_from < scaled_threshold(base_thr, season):
            continue
        nxt = recs.get((pid, unit, season + 1))
        n_to = None if nxt is None else _clean(nxt["n"])
        survived = bool(qual.get((pid, season + 1), False))
        present_next = n_to is not None and n_to >= scaled_threshold(base_thr, season + 1)
        later = [q for q in qual_seasons.get(pid, []) if q > season]
        nq = later[0] if later else None
        b = bios.get(pid)
        birth = b.birth_date if b else None
        codes_from, codes_to = _codes(rec), _codes(nxt)
        if unit == "ALL":
            reclass = False
        elif unit.startswith("group:"):
            # a group is flagged only for code changes inside one of its families, not for arsenal changes
            g = unit.split(":", 1)[1]
            reclass = any(_family_reclass(pid, f, season) for f, gg in FAMILY_GROUP.items() if gg == g)
        else:
            reclass = bool(codes_from and codes_to and set(codes_from) != set(codes_to))
        common = dict(
            pitcher_id=pid, pitch_family=unit, season_from=season, season_to=season + 1,
            birth_year=birth.year if birth else None,
            age_from=age_years(birth, season_midpoint(season)),
            age_from_int=age_completed_years(birth, season_midpoint(season)),
            age_to=age_years(birth, season_midpoint(season + 1)),
            n_from=int(n_from), n_to=None if n_to is None else int(n_to),
            workload_pitches_from=_clean(workload.get((pid, season))),
            workload_pitches_to=_clean(workload.get((pid, season + 1))),
            ip_from=_clean(ipser.get((pid, season))), ip_to=_clean(ipser.get((pid, season + 1))),
            survived=survived, unit_present_next=present_next, both_qualified=survived and present_next,
            returned_later=(not survived) and nq is not None, next_qualifying_season=nq,
            era_from=measurement_era(season), era_to=measurement_era(season + 1),
            era_change=era_change(season, season + 1),
            short_season=season in SHORT_SEASONS or (season + 1) in SHORT_SEASONS,
            reclass_flag=reclass, codes_from=codes_from, codes_to=codes_to,
            min_pitches_threshold=min_pitches, min_season_pitches_threshold=min_season_pitches,
            etl_run_id=run_id, computed_at=now)
        for c in ("workload_pitches_from", "workload_pitches_to"):
            common[c] = None if common[c] is None else int(common[c])
        metrics = ALL_METRICS if unit == "ALL" else FAMILY_METRICS
        for m in metrics:
            vf = _clean(rec.get(m))
            if vf is None:
                continue
            vt = None if nxt is None or n_to is None or n_to <= 0 else _clean(nxt.get(m))
            rows.append(dict(common, metric=m, unit=metric_unit(m), value_from=vf, value_to=vt,
                             delta=None if vt is None else vt - vf))
    return rows


def _summ(x: pd.Series) -> dict:
    x = x.dropna()
    if x.empty:
        return {"n": 0}
    return {"n": int(len(x)), "mean": float(x.mean()), "sd": float(x.std(ddof=1)) if len(x) > 1 else None,
            "median": float(x.median())}


def build_cohorts(obs: pd.DataFrame, run_id: int | None) -> list[dict]:
    """Age / birth-band cohorts. Pooled survival counts exclude short-season pairs (2019->2020, 2020->2021): opt-outs
    and the 60-game schedule make those transitions incomparable; they are counted separately in
    metric_summaries["short_season_pairs"] and remain in their own per-season scopes. `n_returned_later` looks ahead
    to every season in `obs` (see pitchiq.aging.asof for an as-of-cutoff recomputation)."""
    if obs.empty:
        return []
    obs = obs.copy()
    obs["birth_band"] = obs["birth_year"].map(
        lambda y: None if _isnull(y) else f"{int(y) // 5 * 5}-{int(y) // 5 * 5 + 4}")
    base = obs[(obs["pitch_family"] == "ALL") & (obs["metric"] == "pitch_count")]
    if base.empty:
        return []
    clean = obs[obs["both_qualified"] & ~obs["era_change"] & ~obs["short_season"]]
    lookahead = int(obs["season_to"].max())
    now = datetime.now(timezone.utc)
    rows = []
    for ctype, col in (("age", "age_from_int"), ("birth_year_band", "birth_band")):
        b = base[base[col].notna()]
        cl = clean[clean[col].notna()]
        for scope_season in [None] + sorted(b["season_from"].unique().tolist()):
            if scope_season is None:
                short = b[b["short_season"].astype(bool)]
                bs, cs = b[~b["short_season"].astype(bool)], cl
            else:
                short = b.iloc[0:0]
                bs, cs = b[b["season_from"] == scope_season], cl[cl["season_from"] == scope_season]
            for key in sorted(set(bs[col].tolist()) | set(short[col].tolist()), key=str):
                g = bs[bs[col] == key]
                gs = short[short[col] == key]
                gk = cs[cs[col] == key]
                summaries = {}
                for unit, metric in COHORT_SUMMARY:
                    s = _summ(gk.loc[(gk["pitch_family"] == unit) & (gk["metric"] == metric), "delta"])
                    if s["n"]:
                        summaries[f"{unit}|{metric}"] = s
                if not gs.empty:
                    summaries["short_season_pairs"] = {"n": int(len(gs)), "survived": int(gs["survived"].sum()),
                                                       "returned_later": int(gs["returned_later"].sum())}
                n = int(len(g))
                ns = int(g["survived"].sum())
                rows.append(dict(
                    cohort_type=ctype, cohort_key=str(int(key)) if ctype == "age" else str(key),
                    scope="pooled" if scope_season is None else str(int(scope_season)),
                    n_pitchers=n, n_survived=ns, n_dropped=n - ns, n_returned_later=int(g["returned_later"].sum()),
                    n_era_change=int(g["era_change"].sum()), survival_rate=ns / n if n else None,
                    mean_age=_clean(g["age_from"].mean()),
                    metric_summaries={"deltas_exclude": "era-change (2019->2020) and 2020 pairs; both seasons qualified",
                                      "survival_counts_exclude": ("short-season pairs (counted in short_season_pairs)"
                                                                  if scope_season is None else None),
                                      "returned_later_lookahead_through": lookahead,
                                      **summaries},
                    etl_run_id=run_id, computed_at=now))
    return rows


def _load_frame(session: Session, sql: str) -> pd.DataFrame:
    return pd.read_sql(text(sql), session.connection())


def rebuild_observations(session: Session, min_pitches: int, min_season_pitches: int, run_id: int | None) -> dict:
    pss = _load_frame(session, "SELECT * FROM pitcher_season_stats")
    psps = _load_frame(session, "SELECT pitcher_id, season, pitch_type, pitch_family, pitch_group, pitches, usage_pct, "
                                "avg_speed_mph, avg_spin_rpm, ivb_in, break_z_in, break_x_in, whiff_pct, whiffs, "
                                "swings, run_value_per_100, extension_ft FROM pitcher_season_pitch_stats")
    bios = _bio_map(session)
    units = build_units(psps, pss)
    obs = build_observations(units, pss, bios, min_pitches, min_season_pitches, run_id)
    session.execute(delete(AgingObservation))
    session.execute(delete(AgingCohort))
    if obs:
        _insert_rows(session, AgingObservation, obs)
    cohorts = build_cohorts(pd.DataFrame(obs), run_id)
    if cohorts:
        _insert_rows(session, AgingCohort, cohorts)
    session.flush()
    return {"observations": len(obs), "cohorts": len(cohorts)}


# --------------------------------------------------------------------------- career pitch-level (target pitcher)
def ingest_career_pitches(session: Session, pitcher_id: int, seasons: list[int], client: PoliteClient | None = None,
                          use_cache: bool = True, progress: Progress | None = None) -> dict:
    """Pitch-level Statcast for history-only seasons into `pitches` (existing pipeline). Modeling seasons are refused
    here (they belong to `pitchiq ingest`), and the pitcher's `seasons` list keeps naming modeling seasons only."""
    from ..ingest.pipeline import ingest_statcast
    from ..models import Pitcher
    from ..seasons import history_only_seasons, is_history_only
    hist = sorted({int(s) for s in seasons if is_history_only(int(s))})
    if not hist:
        return {"seasons_requested": [], "pitches": 0}
    existing = session.get(Pitcher, pitcher_id)
    before = list(existing.seasons) if existing is not None and existing.seasons is not None else None
    res = ingest_statcast(session, pitcher_id, hist, progress=progress, client=client, use_cache=use_cache)
    p = session.get(Pitcher, pitcher_id)
    p.seasons = before
    by_season = dict(session.execute(text(
        "SELECT season, count(*) FROM pitches WHERE pitcher_id=:p AND season = ANY(:s) GROUP BY 1 ORDER BY 1"),
        {"p": pitcher_id, "s": history_only_seasons() or hist}).all())
    extra = dict(p.extra or {})
    extra["history_only_seasons"] = sorted(int(s) for s in by_season)
    p.extra = extra
    session.flush()
    return {"seasons_requested": hist, "pitches_ingested": res.get("pitches", 0), "games": res.get("games", 0),
            "rows_by_season": {str(k): int(v) for k, v in by_season.items()}}


def career_validation(session: Session, pitcher_id: int) -> dict:
    """Compare the pitch-level rows in `pitches` with the Savant season totals in pitcher_season_stats."""
    rows = session.execute(text("""
        SELECT p.season, count(*) AS rows, count(p.pitch_type) AS typed, s.pitch_count AS savant_pitch_count,
               min(a.age_exact_years) AS age_min, max(a.age_exact_years) AS age_max
        FROM pitches p LEFT JOIN pitcher_season_stats s ON s.pitcher_id = p.pitcher_id AND s.season = p.season
        LEFT JOIN pitch_ages a ON a.pitch_uid = p.pitch_uid
        WHERE p.pitcher_id = :p GROUP BY p.season, s.pitch_count ORDER BY p.season"""), {"p": pitcher_id}).all()
    return {str(r.season): {"pitch_rows": r.rows, "typed_pitches": r.typed, "untyped_pitches": r.rows - r.typed,
                            "savant_pitch_count": r.savant_pitch_count,
                            "savant_matches_rows": None if r.savant_pitch_count is None else
                            r.savant_pitch_count == r.rows,
                            "savant_matches_typed": None if r.savant_pitch_count is None else
                            r.savant_pitch_count == r.typed,
                            "age_range": None if r.age_min is None else [round(float(r.age_min), 3),
                                                                         round(float(r.age_max), 3)]}
            for r in rows}


# --------------------------------------------------------------------------- coverage
def coverage_report(session: Session, min_pitches: int, min_season_pitches: int) -> dict:
    q = lambda sql, **kw: session.execute(text(sql), kw).all()  # noqa: E731
    per_season = {}
    for r in q("""SELECT season, count(*) n, count(*) FILTER (WHERE pitch_count >= :m) q,
                         count(age_june30) aged, count(extension_ft) ext, count(arm_angle_deg) arm,
                         count(oz_swing_pct) chase, count(in_zone_pct) zone, count(k_pct) k
                  FROM pitcher_season_stats GROUP BY season ORDER BY season""", m=min_season_pitches):
        per_season[str(r.season)] = {"pitchers": r.n, "qualifying_pitchers": r.q, "with_age": r.aged,
                                     "with_extension": r.ext, "with_arm_angle": r.arm, "with_chase": r.chase,
                                     "with_zone": r.zone, "with_k_pct": r.k}
    for r in q("""SELECT season, count(*) n, count(*) FILTER (WHERE NOT low_sample) ok,
                         count(DISTINCT pitch_type) types, count(avg_speed_mph) speed, count(avg_spin_rpm) spin,
                         count(ivb_in) ivb, count(whiff_pct) whiff, count(run_value_per_100) rv,
                         count(extension_ft) ext, sum(pitches) pitches
                  FROM pitcher_season_pitch_stats GROUP BY season ORDER BY season"""):
        per_season.setdefault(str(r.season), {}).update(
            {"pitch_type_rows": r.n, "pitch_type_rows_not_low_sample": r.ok, "distinct_pitch_types": r.types,
             "with_speed": r.speed, "with_spin": r.spin, "with_ivb": r.ivb, "with_whiff": r.whiff,
             "with_run_value": r.rv, "with_type_extension": r.ext, "pitches_in_type_rows": int(r.pitches or 0)})
    types = {f"{r.season}:{r.pitch_type}": r.n for r in q(
        "SELECT season, pitch_type, count(*) n FROM pitcher_season_pitch_stats WHERE NOT low_sample "
        "GROUP BY 1, 2 ORDER BY 1, 2")}
    tot = q("""SELECT count(DISTINCT pitcher_id) p, count(*) ps, count(*) FILTER (WHERE pitch_count >= :m) qps,
                      min(season) s0, max(season) s1, count(DISTINCT season) ns FROM pitcher_season_stats""",
            m=min_season_pitches)[0]
    tt = q("SELECT count(*) n, count(DISTINCT pitch_type) ty, count(DISTINCT pitch_family) fam "
           "FROM pitcher_season_pitch_stats")[0]
    bio = q("""SELECT count(*) n, count(birth_date) b FROM pitcher_bio
               WHERE pitcher_id IN (SELECT DISTINCT pitcher_id FROM pitcher_season_stats
                                    UNION SELECT DISTINCT pitcher_id FROM pitcher_season_pitch_stats)""")[0]
    missing_bio = q("""SELECT count(*) FROM (SELECT DISTINCT pitcher_id FROM pitcher_season_stats
                       UNION SELECT DISTINCT pitcher_id FROM pitcher_season_pitch_stats) x
                       WHERE pitcher_id NOT IN (SELECT pitcher_id FROM pitcher_bio WHERE birth_date IS NOT NULL)""")[0][0]
    obs = {f"{r.pitch_family}": {"rows": r.n, "pitcher_seasons": r.ps, "both_qualified": r.bq}
           for r in q("""SELECT pitch_family, count(*) n, count(DISTINCT (pitcher_id, season_from)) ps,
                                count(*) FILTER (WHERE both_qualified) bq
                         FROM aging_observations GROUP BY 1 ORDER BY 1""")}
    surv_sql = """SELECT count(*) n, count(*) FILTER (WHERE survived) s, count(*) FILTER (WHERE returned_later) r,
                       count(*) FILTER (WHERE era_change) e, count(*) FILTER (WHERE short_season) ss
                FROM aging_observations WHERE pitch_family='ALL' AND metric='pitch_count'"""
    surv = q(surv_sql)[0]
    surv_ns = q(surv_sql + " AND NOT short_season")[0]
    surv_by = {str(r.season_from): {"at_risk": r.n, "survived": r.s, "returned_later": r.r,
                                    "survival_rate": round(r.s / r.n, 4) if r.n else None}
               for r in q("""SELECT season_from, count(*) n, count(*) FILTER (WHERE survived) s,
                                    count(*) FILTER (WHERE returned_later) r
                             FROM aging_observations WHERE pitch_family='ALL' AND metric='pitch_count'
                             GROUP BY 1 ORDER BY 1""")}
    reclass = q("""SELECT pitch_family, count(DISTINCT (pitcher_id, season_from)) n FROM aging_observations
                   WHERE reclass_flag GROUP BY 1 ORDER BY 2 DESC""")
    st_by_season = {str(r.season): r.n for r in q(
        "SELECT season, count(*) n FROM pitcher_season_pitch_stats WHERE pitch_type IN ('ST','SV') AND NOT low_sample "
        "GROUP BY 1 ORDER BY 1")}
    return {
        "thresholds": {"min_pitch_type_pitches": min_pitches, "min_season_pitches": min_season_pitches},
        "totals": {"pitchers": tot.p, "pitcher_seasons": tot.ps, "qualifying_pitcher_seasons": tot.qps,
                   "seasons": [tot.s0, tot.s1], "n_seasons": tot.ns, "pitch_type_rows": tt.n,
                   "distinct_pitch_types": tt.ty, "distinct_pitch_families": tt.fam,
                   "pitchers_with_bio": bio.n, "pitchers_with_birth_date": bio.b,
                   "pitchers_missing_birth_date": missing_bio},
        "per_season": per_season, "qualified_pitch_type_rows_by_season_type": types,
        "sweeper_slurve_rows_by_season": st_by_season,
        "observations_by_unit": obs,  # ("observations" is the total row count added by run_aging_etl)
        "survival": {"pitcher_seasons_at_risk": surv.n, "survived": surv.s, "dropped": surv.n - surv.s,
                     "returned_later": surv.r, "era_change_pairs": surv.e, "short_season_pairs": surv.ss,
                     "excluding_short_season_pairs": {"pitcher_seasons_at_risk": surv_ns.n, "survived": surv_ns.s,
                                                      "dropped": surv_ns.n - surv_ns.s, "returned_later": surv_ns.r},
                     "by_season_from": surv_by,
                     "qualifying_threshold_by_season": {str(x): scaled_threshold(min_season_pitches, int(x))
                                                        for x in sorted({int(k) for k in per_season})},
                     "returned_later_note": "returned_later / next_qualifying_season look ahead to the last loaded "
                                            "season; use pitchiq.aging.asof for as-of-cutoff values"},
        "reclass_flagged_pitcher_seasons_by_unit": {r.pitch_family: r.n for r in reclass},
    }


# --------------------------------------------------------------------------- orchestration
def run_aging_etl(session: Session, seasons: list[int] | None = None, *, client: PoliteClient | None = None,
                  progress: Progress | None = None, use_cache: bool = True, search_pitch_types: bool = True,
                  career_pitcher_id: int | None = -1, career_seasons: list[int] | None = None,
                  refresh_bio: bool = False, min_pitches: int | None = None,
                  min_season_pitches: int | None = None) -> dict:
    s = get_settings()
    seasons = sorted(int(x) for x in (seasons or s.aging_seasons))
    min_pitches = int(min_pitches if min_pitches is not None else s.aging_min_pitch_type_pitches)
    min_season_pitches = int(min_season_pitches if min_season_pitches is not None else s.aging_min_season_pitches)
    if career_pitcher_id == -1:
        career_pitcher_id = s.default_pitcher_id
    career_seasons = sorted(int(x) for x in (career_seasons if career_seasons is not None else s.career_seasons))
    client = client or PoliteClient()
    params = {"seasons": seasons, "min_pitch_type_pitches": min_pitches, "min_season_pitches": min_season_pitches,
              "search_pitch_types": search_pitch_types, "career_pitcher_id": career_pitcher_id,
              "career_seasons": career_seasons if career_pitcher_id else [], "use_cache": use_cache,
              "game_types": list(s.game_types)}
    run = AgingEtlRun(status="running", params=params, units=UNITS)
    session.add(run)
    session.flush()
    logs = {k: src.SourceLog() for k in ("movement", "arsenals", "arsenal_stats", "custom", "search", "people")}
    stats: Counter = Counter()

    def prog(frac, msg, n=None):
        log(logger, "aging_etl", progress=round(frac, 3), message=msg)
        if progress:
            progress(frac, msg, n)

    type_frames, season_frames = [], []
    for i, season in enumerate(seasons):
        fetched = fetch_season(client, season, logs, use_cache=use_cache, search_pitch_types=search_pitch_types,
                               game_types=tuple(s.game_types))
        t = merge_pitch_type_rows(season, fetched)
        a = merge_season_rows(season, fetched)
        type_frames.append(t)
        season_frames.append(a)
        prog(0.6 * (i + 1) / len(seasons), f"Fetched {season}: {len(a)} pitcher-seasons, {len(t)} pitch-type rows",
             len(t))
    types_df = pd.concat([t for t in type_frames if not t.empty], ignore_index=True) if any(
        not t.empty for t in type_frames) else pd.DataFrame()
    season_df = pd.concat([a for a in season_frames if not a.empty], ignore_index=True) if any(
        not a.empty for a in season_frames) else pd.DataFrame()

    ids = set()
    for d in (types_df, season_df):
        if not d.empty:
            ids |= {int(x) for x in d["pitcher_id"].dropna().unique()}
    if career_pitcher_id:
        ids.add(int(career_pitcher_id))
    bio_res = ensure_bios(session, client, ids, logs["people"], refresh=refresh_bio, use_cache=use_cache)
    bios = _bio_map(session)
    prog(0.7, f"Bios: {len(ids)} pitchers ({bio_res['requested']} requested)")

    hand = {}
    if not types_df.empty:
        hh = types_df.dropna(subset=["pitch_hand"]).groupby(["pitcher_id", "season"])["pitch_hand"].first()
        hand = {(int(k[0]), int(k[1])): v for k, v in hh.items()}
    ps_rows = pitch_type_rows(types_df, bios, min_pitches, run.etl_run_id, stats) if not types_df.empty else []
    ss_rows = season_rows(season_df, hand, bios, min_season_pitches, run.etl_run_id, stats) if not season_df.empty else []
    session.execute(delete(PitcherSeasonPitchStats).where(PitcherSeasonPitchStats.season.in_(seasons)))
    session.execute(delete(PitcherSeasonStats).where(PitcherSeasonStats.season.in_(seasons)))
    _insert_rows(session, PitcherSeasonPitchStats, ps_rows)
    _insert_rows(session, PitcherSeasonStats, ss_rows)
    session.flush()
    prog(0.75, f"Stored {len(ss_rows)} pitcher-seasons and {len(ps_rows)} pitch-type rows", len(ps_rows))

    career = None
    if career_pitcher_id and career_seasons:
        career = ingest_career_pitches(session, int(career_pitcher_id), career_seasons, client=client,
                                       use_cache=use_cache)
        prog(0.85, f"Career pitch-level history: {career.get('pitches_ingested', 0)} pitches")

    obs_res = rebuild_observations(session, min_pitches, min_season_pitches, run.etl_run_id)
    prog(0.95, f"Observations: {obs_res['observations']}, cohorts: {obs_res['cohorts']}", obs_res["observations"])

    def rate(a, b):
        return None if not stats[b] else round(1 - stats[a] / stats[b], 4)
    validation = {
        "pitch_type_count_agreement": {"compared": stats["pitch_count_compared"],
                                       "disagree": stats["pitch_count_disagree"],
                                       "agreement_rate": rate("pitch_count_disagree", "pitch_count_compared")},
        "speed_movement_vs_arsenals": {"compared": stats["speed_compared"], "disagree_gt_0.25mph": stats["speed_disagree"],
                                       "agreement_rate": rate("speed_disagree", "speed_compared")},
        "season_pitch_count_custom_vs_search": {"compared": stats["season_pitch_count_compared"],
                                                "disagree": stats["season_pitch_count_disagree"],
                                                "agreement_rate": rate("season_pitch_count_disagree",
                                                                       "season_pitch_count_compared")},
        "ip_formatted_vs_outs": {"compared": stats["ip_compared"], "mismatch": stats["ip_mismatch"]},
        "duplicate_source_keys_collapsed": {"pitch_type_rows": stats["pitch_type_duplicate_keys"],
                                            "pitcher_season_rows": stats["season_duplicate_keys"]},
        "savant_age_vs_computed_june30_age": {"compared": stats["age_compared"], "differs": stats["age_differs"],
                                              "agreement_rate": rate("age_differs", "age_compared")},
        "bios": bio_res,
    }
    if career_pitcher_id:
        validation["career_pitch_level_vs_savant"] = career_validation(session, int(career_pitcher_id))
    run.sources = {k: v.as_dict() for k, v in logs.items()}
    run.sources["endpoints"] = {"movement": src.MOVEMENT_URL, "arsenals": src.ARSENALS_URL,
                                "arsenal_stats": src.ARSENAL_STATS_URL, "custom": src.CUSTOM_URL,
                                "search": src.SEARCH_URL, "people": src.PEOPLE_URL}
    run.coverage = {**coverage_report(session, min_pitches, min_season_pitches), "career": career,
                    **obs_res}
    run.validation = validation
    run.status = "succeeded"
    run.finished_at = datetime.now(timezone.utc)
    session.flush()
    prog(1.0, "Aging ETL complete")
    return {"etl_run_id": run.etl_run_id, "pitcher_seasons": len(ss_rows), "pitch_type_rows": len(ps_rows),
            **obs_res, "career": career, "sources": {k: v.as_dict() for k, v in logs.items()},
            "validation": validation, "coverage_totals": run.coverage["totals"]}


__all__ = ["run_aging_etl", "rebuild_observations", "ingest_career_pitches", "coverage_report", "UNITS",
           "FAMILY_GROUP"]
