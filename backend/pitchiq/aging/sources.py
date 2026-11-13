"""Baseball Savant season leaderboards + grouped Statcast search, MLB Stats API people.

Every request goes through PoliteClient (1 request/s per host, raw responses cached under data/raw). Each fetch
returns the parsed frame plus provenance: the exact URL requested and when the response was fetched from upstream
(the mtime of the cached raw response; "now" when it was not cached).

Endpoints and columns were verified against live responses (2015 and 2024) before this module was written:

* pitch-movement (per pitch type, 2015+): avg_speed (mph), pitches_thrown, total_pitches, pitch_per,
  pitcher_break_z / pitcher_break_z_induced / pitcher_break_x (inches), diff_z / diff_x (inches vs league).
  Supported pitch_type values: FF SI FC SL ST SV CH CU FS KN FO SC (KC/CS/EP/FA return no rows).
* pitch-arsenals type=avg_speed | avg_spin | n_ (usage %), wide per code ff si fc sl ch cu fs kn st sv (2015+).
* pitch-arsenal-stats (2017+; 2015 and 2016 return a header only): pitches, pitch_usage, whiff_percent, k_percent,
  put_away, run_value, run_value_per_100, woba, est_woba, hard_hit_percent, pa.
  Leaderboards fold Statcast KC (knuckle curve) into CU: a KC pitcher's KC pitches appear as CU with equal counts.
* custom leaderboard (pitcher-season): the SELECTIONS below. release_extension / release_pos_* are NOT available
  there (they come back blank), so they are taken from the grouped Statcast search.
* statcast_search/csv with group_by=name (per season, optionally filtered by hfPT pitch type): pitches,
  total_pitches, velocity, spin_rate, release_extension, release_pos_x/z, arm_angle (2020+ only), whiffs, swings.
"""
from __future__ import annotations

import io
import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone

import pandas as pd
import requests

from ..http import PoliteClient, SourceUnavailable
from .families import leaderboard_code

SAVANT = "https://baseballsavant.mlb.com"
MOVEMENT_URL = f"{SAVANT}/leaderboard/pitch-movement"
ARSENALS_URL = f"{SAVANT}/leaderboard/pitch-arsenals"
ARSENAL_STATS_URL = f"{SAVANT}/leaderboard/pitch-arsenal-stats"
CUSTOM_URL = f"{SAVANT}/leaderboard/custom"
SEARCH_URL = f"{SAVANT}/statcast_search/csv"
PEOPLE_URL = "https://statsapi.mlb.com/api/v1/people"
NAMESPACE = "savant_leaderboards"

MOVEMENT_TYPES = ["FF", "SI", "FC", "SL", "ST", "SV", "CH", "CU", "FS", "KN", "FO", "SC"]
ARSENAL_CODES = ["ff", "si", "fc", "sl", "ch", "cu", "fs", "kn", "st", "sv"]
ARSENAL_KINDS = {"avg_speed": "{c}_avg_speed", "avg_spin": "{c}_avg_spin", "n_": "n_{c}"}
# Statcast codes queried in the grouped search; KC/CS are folded into CU like the leaderboards do.
SEARCH_TYPES = ["FF", "SI", "FC", "SL", "ST", "SV", "CU", "KC", "CS", "CH", "FS", "FO", "SC", "KN", "EP"]
ARSENAL_STATS_FIRST_SEASON = 2017

CUSTOM_SELECTIONS = [
    "player_age", "p_game", "p_starting_p", "p_formatted_ip", "p_out", "pa", "pitch_count", "k_percent",
    "bb_percent", "in_zone_percent", "oz_swing_percent", "z_swing_percent", "swing_percent", "whiff_percent",
    "edge_percent", "f_strike_percent", "meatball_percent", "iz_contact_percent", "oz_contact_percent",
    "fastball_avg_speed", "fastball_avg_spin", "breaking_avg_speed", "breaking_avg_spin", "offspeed_avg_speed",
    "offspeed_avg_spin", "pitch_hand", "arm_angle", "xwoba", "p_era",
]

REQUIRED = {
    "movement": ["pitcher_id", "year", "pitch_type", "avg_speed", "pitches_thrown", "total_pitches",
                 "pitcher_break_z", "pitcher_break_z_induced", "pitcher_break_x"],
    "arsenal_stats": ["player_id", "pitch_type", "pitches", "pitch_usage", "whiff_percent", "run_value_per_100"],
    "custom": ["player_id", "year", "pitch_count", "k_percent", "bb_percent", "in_zone_percent", "oz_swing_percent"],
    "search": ["player_id", "pitches", "total_pitches", "velocity", "release_extension", "whiffs", "swings"],
}


@dataclass
class Fetched:
    source: str
    season: int
    url: str
    fetched_at: datetime
    frame: pd.DataFrame
    variant: str | None = None


@dataclass
class SourceLog:
    requests: int = 0
    rows: int = 0
    empty: list = field(default_factory=list)
    failures: list = field(default_factory=list)
    missing_columns: list = field(default_factory=list)

    def as_dict(self) -> dict:
        return {"requests": self.requests, "rows": self.rows, "empty_responses": self.empty,
                "failures": self.failures, "missing_columns": self.missing_columns}


def full_url(url: str, params: dict | None) -> str:
    return requests.Request("GET", url, params=params).prepare().url


def _cached_fetched_at(client: PoliteClient, url: str, params: dict | None, namespace: str, suffix: str) -> datetime:
    key = url + "?" + json.dumps(params or {}, sort_keys=True)
    try:
        p = client._cache_path(namespace, key, suffix)
        if p.exists() and p.stat().st_size > 0:
            return datetime.fromtimestamp(os.path.getmtime(p), tz=timezone.utc)
    except Exception:  # noqa: BLE001 - provenance only; never fail a fetch over it
        pass
    return datetime.now(timezone.utc)


def _read_csv(text: str) -> pd.DataFrame:
    if not text or not text.strip():
        return pd.DataFrame()
    df = pd.read_csv(io.StringIO(text.lstrip("﻿")), low_memory=False)
    df.columns = [c.strip().lstrip("﻿") for c in df.columns]
    return df


def _get_csv(client: PoliteClient, source: str, season: int, url: str, params: dict, log: SourceLog,
             use_cache: bool, variant: str | None = None) -> Fetched | None:
    log.requests += 1
    try:
        text = client.get(url, params, namespace=NAMESPACE, suffix=".csv", use_cache=use_cache)
    except SourceUnavailable as e:
        log.failures.append({"season": season, "variant": variant, "error": str(e)})
        return None
    df = _read_csv(text)
    fetched_at = _cached_fetched_at(client, url, params, NAMESPACE, ".csv")
    if df.empty:
        log.empty.append({"season": season, "variant": variant})
    else:
        missing = [c for c in REQUIRED.get(source, []) if c not in df.columns]
        if missing:
            log.missing_columns.append({"season": season, "variant": variant, "missing": missing})
        log.rows += len(df)
    return Fetched(source, season, full_url(url, params), fetched_at, df, variant)


def fetch_movement(client, season, pitch_type, log, use_cache=True):
    params = {"year": str(season), "team": "", "min": "1", "pitch_type": pitch_type, "hand": "",
              "x": "pitcher_break_x_hidden", "z": "pitcher_break_z", "csv": "true"}
    return _get_csv(client, "movement", season, MOVEMENT_URL, params, log, use_cache, pitch_type)


def fetch_arsenals(client, season, kind, log, use_cache=True):
    params = {"year": str(season), "min": "1", "type": kind, "hand": "", "csv": "true"}
    return _get_csv(client, "arsenals", season, ARSENALS_URL, params, log, use_cache, kind)


def fetch_arsenal_stats(client, season, log, use_cache=True):
    params = {"type": "pitcher", "pitchType": "", "year": str(season), "team": "", "min": "1", "csv": "true"}
    return _get_csv(client, "arsenal_stats", season, ARSENAL_STATS_URL, params, log, use_cache)


def fetch_custom(client, season, log, use_cache=True):
    params = {"year": str(season), "type": "pitcher", "filter": "", "min": "1",
              "selections": ",".join(CUSTOM_SELECTIONS), "csv": "true"}
    return _get_csv(client, "custom", season, CUSTOM_URL, params, log, use_cache)


def fetch_search(client, season, pitch_type, log, game_types=("R",), use_cache=True):
    params = {"all": "true", "hfGT": "".join(f"{g}|" for g in game_types), "hfPT": f"{pitch_type}|" if pitch_type else "",
              "hfSea": f"{season}|", "player_type": "pitcher", "group_by": "name", "min_pitches": "0",
              "min_results": "0", "min_pas": "0", "sort_col": "pitches", "sort_order": "desc"}
    return _get_csv(client, "search", season, SEARCH_URL, params, log, use_cache, pitch_type or "ALL")


# --------------------------------------------------------------------------- normalisation to long frames
HAND_COLS = {"pitch_hand", "cu_pitch_hand"}


def collapse_duplicates(df: pd.DataFrame, keys: list[str], source: str) -> pd.DataFrame:
    """Collapse rows sharing a key into one. Savant lists switch-pitchers (Pat Venditte, 2015-2020) once per throwing
    hand with identical counts but hand-specific speed / spin / break, so values that agree are kept, values that
    differ are set to null (never averaged without weights), the hand becomes "S", and `dup_<source>` records the
    row count and the differing columns."""
    if df.empty:
        return df
    dup = df.duplicated(keys, keep=False)
    if not dup.any():
        return df
    out = []
    for _, g in df[dup].groupby(keys, sort=False):
        r = g.iloc[0].to_dict()
        differing = []
        for c in g.columns:
            if c in keys:
                continue
            if g[c].astype(str).nunique(dropna=False) > 1:
                differing.append(c)
                r[c] = "S" if c in HAND_COLS else None
        r["dup_" + source] = {"rows": int(len(g)), "differing": differing}
        out.append(r)
    return pd.concat([df[~dup], pd.DataFrame(out)], ignore_index=True)


def _num(s: pd.Series | None) -> pd.Series | None:
    if s is None:  # optional column absent from this response
        return None
    return pd.to_numeric(s.astype(str).str.replace(",", "", regex=False).replace({"": None, "nan": None}),
                         errors="coerce")


def movement_long(f: Fetched) -> pd.DataFrame:
    d = f.frame
    if d.empty:
        return pd.DataFrame()
    out = pd.DataFrame({
        "pitcher_id": _num(d["pitcher_id"]).astype("Int64"), "season": f.season,
        "pitch_type": d["pitch_type"].astype(str).str.upper(),
        "player_name": d.get("last_name, first_name"), "pitch_hand": d.get("pitch_hand"),
        "pitch_name": d.get("pitch_type_name"),
        "mv_avg_speed": _num(d["avg_speed"]), "mv_pitches": _num(d["pitches_thrown"]),
        "mv_total_pitches": _num(d["total_pitches"]), "mv_usage": _num(d["pitch_per"]) * 100.0,
        "break_z_in": _num(d["pitcher_break_z"]), "ivb_in": _num(d["pitcher_break_z_induced"]),
        "break_x_in": _num(d["pitcher_break_x"]),
        "break_z_vs_league_in": _num(d["diff_z"]) if "diff_z" in d else None,
        "break_x_vs_league_in": _num(d["diff_x"]) if "diff_x" in d else None,
    })
    out["src_movement"] = f.url
    out["fat_movement"] = f.fetched_at
    return collapse_duplicates(out.dropna(subset=["pitcher_id"]), ["pitcher_id", "season", "pitch_type"], "movement")


def arsenals_long(frames: list[Fetched]) -> pd.DataFrame:
    """Wide (one column per code) -> long (pitcher, pitch_type) with speed, spin and usage."""
    parts = []
    for f in frames:
        d = f.frame
        if d.empty:
            continue
        kind = f.variant
        col = {"avg_speed": "ars_avg_speed", "avg_spin": "ars_avg_spin", "n_": "ars_usage"}[kind]
        for c in ARSENAL_CODES:
            src = ARSENAL_KINDS[kind].format(c=c)
            if src not in d:
                continue
            vals = _num(d[src])
            m = vals.notna()
            if not m.any():
                continue
            parts.append(pd.DataFrame({"pitcher_id": _num(d.loc[m, "pitcher"]).astype("Int64"), "season": f.season,
                                       "pitch_type": c.upper(), "metric": col, "value": vals[m],
                                       "url": f.url, "fetched_at": f.fetched_at}))
    if not parts:
        return pd.DataFrame()
    long = collapse_duplicates(pd.concat(parts, ignore_index=True),
                               ["pitcher_id", "season", "pitch_type", "metric"], "arsenals")
    dups = set()
    if "dup_arsenals" in long:
        dups = {tuple(k) for k in long.loc[long["dup_arsenals"].notna(), ["pitcher_id", "season", "pitch_type"]]
                .itertuples(index=False)}
        long = long[long["value"].notna()]
    wide = long.pivot_table(index=["pitcher_id", "season", "pitch_type"], columns="metric", values="value",
                            aggfunc="first").reset_index()
    prov = long.groupby(["pitcher_id", "season", "pitch_type"]).agg(
        src_arsenals=("url", lambda s: sorted(set(s))), fat_arsenals=("fetched_at", "min")).reset_index()
    wide = wide.merge(prov, on=["pitcher_id", "season", "pitch_type"])
    wide.columns.name = None
    if dups:
        wide["dup_arsenals"] = [{"rows": 2, "differing": ["hand-specific values"]}
                                if (p, s_, t) in dups else None
                                for p, s_, t in wide[["pitcher_id", "season", "pitch_type"]].itertuples(index=False)]
    return wide


def arsenal_stats_long(f: Fetched) -> pd.DataFrame:
    d = f.frame
    if d.empty:
        return pd.DataFrame()
    out = pd.DataFrame({
        "pitcher_id": _num(d["player_id"]).astype("Int64"), "season": f.season,
        "pitch_type": d["pitch_type"].astype(str).str.upper(), "as_pitch_name": d.get("pitch_name"),
        "as_player_name": d.get("last_name, first_name"),
        "as_pitches": _num(d["pitches"]), "as_usage": _num(d["pitch_usage"]), "as_pa": _num(d.get("pa")),
        "as_whiff": _num(d["whiff_percent"]), "as_k": _num(d.get("k_percent")), "as_put_away": _num(d.get("put_away")),
        "as_run_value": _num(d.get("run_value")), "as_rv100": _num(d["run_value_per_100"]),
        "as_woba": _num(d.get("woba")), "as_xwoba": _num(d.get("est_woba")),
        "as_hard_hit": _num(d.get("hard_hit_percent")),
    })
    out["src_arsenal_stats"] = f.url
    out["fat_arsenal_stats"] = f.fetched_at
    return collapse_duplicates(out.dropna(subset=["pitcher_id"]), ["pitcher_id", "season", "pitch_type"],
                               "arsenal_stats")


def search_long(frames: list[Fetched]) -> pd.DataFrame:
    """Per-pitch-type grouped search -> leaderboard codes (KC, CS folded into CU by pitch-weighted means)."""
    parts = []
    for f in frames:
        d = f.frame
        if d.empty or f.variant in (None, "ALL"):
            continue
        parts.append(pd.DataFrame({
            "pitcher_id": _num(d["player_id"]).astype("Int64"), "season": f.season,
            "pitch_type": leaderboard_code(f.variant), "search_code": f.variant,
            "gs_pitches": _num(d["pitches"]), "gs_total_pitches": _num(d["total_pitches"]),
            "gs_velocity": _num(d["velocity"]), "gs_spin": _num(d.get("spin_rate")),
            "gs_extension": _num(d["release_extension"]), "gs_pos_x": _num(d.get("release_pos_x")),
            "gs_pos_z": _num(d.get("release_pos_z")), "gs_arm_angle": _num(d.get("arm_angle")),
            "gs_whiffs": _num(d["whiffs"]), "gs_swings": _num(d["swings"]),
            "url": f.url, "fetched_at": f.fetched_at}))
    if not parts:
        return pd.DataFrame()
    d = collapse_duplicates(pd.concat(parts, ignore_index=True).dropna(subset=["pitcher_id"]),
                            ["pitcher_id", "season", "search_code"], "search")

    def agg(g: pd.DataFrame) -> pd.Series:
        w = g["gs_pitches"].fillna(0)
        out = {"gs_pitches": w.sum(), "gs_total_pitches": g["gs_total_pitches"].max(),
               "gs_whiffs": g["gs_whiffs"].sum(min_count=1), "gs_swings": g["gs_swings"].sum(min_count=1),
               "search_codes": sorted(set(g["search_code"])), "src_search": sorted(set(g["url"])),
               "fat_search": g["fetched_at"].min()}
        for c in ("gs_velocity", "gs_spin", "gs_extension", "gs_pos_x", "gs_pos_z", "gs_arm_angle"):
            m = g[c].notna() & (w > 0)
            out[c] = float((g.loc[m, c] * w[m]).sum() / w[m].sum()) if m.any() else None
        return pd.Series(out)

    keys = ["pitcher_id", "season", "pitch_type"]
    n = d.groupby(keys)["search_code"].transform("size")
    single = d[n == 1].rename(columns={"url": "src_search", "fetched_at": "fat_search"})
    single["search_codes"] = single["search_code"].map(lambda c: [c])
    single["src_search"] = single["src_search"].map(lambda u: [u])
    single = single.drop(columns=["search_code"])
    multi = d[n > 1]
    if multi.empty:
        return single.reset_index(drop=True)
    folded = multi.groupby(keys).apply(agg, include_groups=False).reset_index()
    return pd.concat([single, folded], ignore_index=True)


def custom_frame(f: Fetched) -> pd.DataFrame:
    d = f.frame
    if d.empty:
        return pd.DataFrame()
    out = pd.DataFrame({"pitcher_id": _num(d["player_id"]).astype("Int64"), "season": f.season,
                        "cu_player_name": d.get("last_name, first_name")})
    for c in CUSTOM_SELECTIONS:
        if c in ("pitch_hand", "p_formatted_ip"):
            out["cu_" + c] = d[c] if c in d else None
        else:
            out["cu_" + c] = _num(d[c]) if c in d else None
    out["src_custom"] = f.url
    out["fat_custom"] = f.fetched_at
    return collapse_duplicates(out.dropna(subset=["pitcher_id"]), ["pitcher_id", "season"], "custom")


def search_season_frame(f: Fetched) -> pd.DataFrame:
    d = f.frame
    if d.empty:
        return pd.DataFrame()
    out = pd.DataFrame({
        "pitcher_id": _num(d["player_id"]).astype("Int64"), "season": f.season,
        "gs_player_name": d.get("player_name"), "gs_pitches": _num(d["pitches"]),
        "gs_velocity": _num(d["velocity"]), "gs_spin": _num(d.get("spin_rate")),
        "gs_extension": _num(d["release_extension"]), "gs_pos_x": _num(d.get("release_pos_x")),
        "gs_pos_z": _num(d.get("release_pos_z")), "gs_arm_angle": _num(d.get("arm_angle")),
        "gs_whiffs": _num(d["whiffs"]), "gs_swings": _num(d["swings"]), "gs_pa": _num(d.get("pa")),
        "gs_k": _num(d.get("k_percent")), "gs_bb": _num(d.get("bb_percent")),
    })
    out["src_search"] = f.url
    out["fat_search"] = f.fetched_at
    return collapse_duplicates(out.dropna(subset=["pitcher_id"]), ["pitcher_id", "season"], "search")


def fetch_people(client: PoliteClient, ids: list[int], log: SourceLog, use_cache: bool = True) -> list[dict]:
    """MLB Stats API bulk people endpoint (100 ids per request). Returns raw person dicts + provenance."""
    out = []
    for i in range(0, len(ids), 100):
        chunk = ids[i:i + 100]
        params = {"personIds": ",".join(map(str, chunk))}
        log.requests += 1
        try:
            d = client.get_json(PEOPLE_URL, params, namespace="people", use_cache=use_cache)
        except SourceUnavailable as e:
            log.failures.append({"ids": f"{chunk[0]}..{chunk[-1]}", "error": str(e)})
            continue
        fat = _cached_fetched_at(client, PEOPLE_URL, params, "people", ".json")
        url = full_url(PEOPLE_URL, params)
        for p in d.get("people", []):
            out.append({**p, "_source_url": url, "_fetched_at": fat})
        log.rows += len(d.get("people", []))
    return out
