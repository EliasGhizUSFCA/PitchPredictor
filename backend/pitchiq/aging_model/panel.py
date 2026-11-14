"""Pitcher-season panel for aging forecasts.

One row per (pitcher, season, unit, metric) with the observed value, its sample size, exact age on June 30 (from the
MLB Stats API birth date), workload and qualification. Units are pitch families (SL/ST/SV -> slider etc., values are
pitch-weighted means over the official codes of the family, which absorbs the 2023 sweeper relabelling) and "ALL"
(pitcher-season metrics). Horizontal break and release side are normalised so the same number means the same thing
for both hands (see TARGETS).

Only seasons that were loaded are present; a pitcher who did not pitch has no row (nothing is imputed).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sqlalchemy import text
from sqlalchemy.orm import Session

from ..aging.families import measurement_era, scaled_threshold, season_games
from ..config import get_settings


@dataclass(frozen=True)
class Target:
    unit: str  # pitch family or "ALL"
    metric: str
    label: str
    units: str
    weight: str = "n"  # which sample size scales sampling noise: n (unit pitches) or bf (batters faced)

    @property
    def key(self) -> str:
        return f"{self.unit}|{self.metric}"


FAMILY_METRICS: dict[str, tuple[str, str]] = {
    "avg_speed_mph": ("velocity", "mph"),
    "avg_spin_rpm": ("spin rate", "rpm"),
    "ivb_in": ("induced vertical break", "in"),
    "break_x_in": ("horizontal break (magnitude)", "in"),
    "extension_ft": ("release extension", "ft"),
    "release_pos_z_ft": ("release height", "ft"),
    "release_pos_x_armside_ft": ("release side (arm side +)", "ft"),
}
ALL_METRICS: dict[str, tuple[str, str, str]] = {
    "in_zone_pct": ("zone rate", "% of pitches", "n"),
    "oz_swing_pct": ("chase rate", "% of out-of-zone pitches", "n"),
    "whiff_pct": ("whiff rate", "% of swings", "n"),
    "k_pct": ("strikeout rate", "% of batters faced", "bf"),
    "bb_pct": ("walk rate", "% of batters faced", "bf"),
    "arm_angle_deg": ("arm angle (Hawk-Eye, 2020+)", "deg", "n"),
}
# Families the population backtest covers by default (Chris Sale's arsenal; the panel holds every family).
DEFAULT_FAMILIES = ["four_seam", "sinker", "slider", "changeup"]
# Location consistency is derivable only from pitch-level data, which exists for the target pitcher alone.
LOCATION_METRICS: dict[str, tuple[str, str]] = {
    "plate_x_sd_ft": ("horizontal location dispersion (SD of plate_x)", "ft"),
    "plate_z_sd_ft": ("vertical location dispersion (SD of plate_z)", "ft"),
}


def targets(families: list[str] | None = None) -> list[Target]:
    out = [Target(f, m, f"{f} {lab}", u) for f in (families or DEFAULT_FAMILIES) for m, (lab, u) in
           FAMILY_METRICS.items()]
    out += [Target("ALL", m, lab, u, w) for m, (lab, u, w) in ALL_METRICS.items()]
    return out


def location_targets(families: list[str] | None = None) -> list[Target]:
    return [Target(f, m, f"{f} {lab}", u) for f in (families or DEFAULT_FAMILIES) for m, (lab, u) in
            LOCATION_METRICS.items()]


def _wmean(df: pd.DataFrame, by: list[str], col: str, w: str) -> pd.Series:
    x = pd.to_numeric(df[col], errors="coerce")
    ww = pd.to_numeric(df[w], errors="coerce").fillna(0)
    m = x.notna() & (ww > 0)
    tmp = df[by].copy()
    tmp["wx"] = (x * ww).where(m, 0.0)
    tmp["w"] = ww.where(m, 0.0)
    g = tmp.groupby(by)[["wx", "w"]].sum()
    return (g["wx"] / g["w"]).where(g["w"] > 0)


def build_panel(pss: pd.DataFrame, psps: pd.DataFrame, bio: pd.DataFrame, min_unit_pitches: int | None = None,
                min_season_pitches: int | None = None) -> pd.DataFrame:
    """Pure function: long panel from the aging tables (frames as read from the database)."""
    s = get_settings()
    min_unit = s.aging_min_pitch_type_pitches if min_unit_pitches is None else int(min_unit_pitches)
    min_season = s.aging_min_season_pitches if min_season_pitches is None else int(min_season_pitches)
    pss = pss.copy()
    flags = pss["quality_flags"].map(lambda v: list(v) if isinstance(v, (list, tuple, np.ndarray)) else [])
    pss = pss[~flags.map(lambda f: "position_player" in f)]
    bio = bio.set_index("pitcher_id")
    pss["throws"] = pss["pitcher_id"].map(bio["throws"])
    pss["birth_year"] = pss["pitcher_id"].map(pd.to_datetime(bio["birth_date"]).dt.year)
    for c in ("pitch_count", "games", "games_started", "batters_faced", "ip"):
        pss[c] = pd.to_numeric(pss[c], errors="coerce")
    pss["gs_share"] = (pss["games_started"] / pss["games"]).where(pss["games"] > 0)
    base_cols = ["pitcher_id", "season", "age_june30_exact", "birth_year", "throws", "pitch_count", "ip",
                 "batters_faced", "gs_share"]
    base = pss[base_cols].rename(columns={"age_june30_exact": "age", "pitch_count": "workload"})
    frames = []
    # ---- pitcher-season (ALL) metrics
    for m, (_, _, w) in ALL_METRICS.items():
        d = base.copy()
        d["unit"], d["metric"] = "ALL", m
        d["value"] = pd.to_numeric(pss[m], errors="coerce").values
        d["n"] = (pss["batters_faced"] if w == "bf" else pss["pitch_count"]).values
        d["qualified"] = (d["workload"] >= d["season"].map(lambda x: scaled_threshold(min_season, int(x)))).values
        frames.append(d)
    # ---- pitch-family metrics (pitch-weighted means over official codes in the family)
    if not psps.empty:
        p = psps.copy()
        p["pitches"] = pd.to_numeric(p["pitches"], errors="coerce").fillna(0)
        p = p[p["pitches"] > 0]
        throws = p["pitcher_id"].map(bio["throws"])
        sign = np.where(throws == "R", -1.0, np.where(throws == "L", 1.0, np.nan))
        p["release_pos_x_armside_ft"] = pd.to_numeric(p["release_pos_x_ft"], errors="coerce") * sign
        by = ["pitcher_id", "season", "pitch_family"]
        g = p.groupby(by).agg(n=("pitches", "sum"), codes=("pitch_type", lambda x: ",".join(sorted(set(x)))))
        for m in FAMILY_METRICS:
            g[m] = _wmean(p, by, m, "pitches")
        g = g.reset_index().merge(base, on=["pitcher_id", "season"], how="inner")
        for m in FAMILY_METRICS:
            d = g[base_cols[:1] + ["season", "age", "birth_year", "throws", "workload", "ip", "batters_faced",
                                   "gs_share", "n"]].copy()
            d["unit"], d["metric"] = g["pitch_family"].values, m
            d["value"] = g[m].values
            d["codes"] = g["codes"].values
            d["qualified"] = (d["n"] >= d["season"].map(lambda x: scaled_threshold(min_unit, int(x)))).values
            frames.append(d)
    out = pd.concat(frames, ignore_index=True)
    out = out[out["value"].notna() & out["age"].notna()]
    out["season"] = out["season"].astype(int)
    out["era"] = out["season"].map(measurement_era)
    out["short_season"] = out["season"].map(lambda x: season_games(int(x)) != 162)
    return out.reset_index(drop=True)


def load_panel(session: Session, cutoff: int | None = None, pitcher_id: int | None = None, **kw) -> pd.DataFrame:
    """Panel from the database; `cutoff` keeps seasons <= cutoff only (the as-of view for forecasts)."""
    conds = ([] if cutoff is None else [f"season <= {int(cutoff)}"]) + \
        ([] if pitcher_id is None else [f"pitcher_id = {int(pitcher_id)}"])
    c = (" WHERE " + " AND ".join(conds)) if conds else ""
    conn = session.connection()
    pss = pd.read_sql(text("SELECT pitcher_id, season, age_june30_exact, pitch_count, games, games_started, "
                           "batters_faced, ip, quality_flags, " + ", ".join(ALL_METRICS) +
                           " FROM pitcher_season_stats" + c), conn)
    psps = pd.read_sql(text("SELECT pitcher_id, season, pitch_type, pitch_family, pitches, avg_speed_mph, "
                            "avg_spin_rpm, ivb_in, break_x_in, extension_ft, release_pos_x_ft, release_pos_z_ft "
                            "FROM pitcher_season_pitch_stats" + c), conn)
    bio = pd.read_sql(text("SELECT pitcher_id, birth_date, throws FROM pitcher_bio"), conn)
    return build_panel(pss, psps, bio, **kw)


def location_panel(session: Session, pitcher_id: int, cutoff: int | None = None,
                   min_unit_pitches: int | None = None) -> pd.DataFrame:
    """Target pitcher's location dispersion per family and season from pitch-level rows (all loaded seasons, incl.
    history-only career seasons: this is aging analysis, not next-pitch modeling)."""
    from ..aging.families import pitch_family
    s = get_settings()
    min_unit = s.aging_min_pitch_type_pitches if min_unit_pitches is None else int(min_unit_pitches)
    c = "" if cutoff is None else " AND p.season <= :c"
    df = pd.read_sql(text("SELECT p.season, p.pitch_type, p.plate_x, p.plate_z, a.age_exact_years AS age_pitch "
                          "FROM pitches p LEFT JOIN pitch_ages a ON a.pitch_uid = p.pitch_uid "
                          "WHERE p.pitcher_id = :p AND p.pitch_type IS NOT NULL AND p.plate_x IS NOT NULL "
                          "AND p.plate_z IS NOT NULL" + c), session.connection(),
                     params={"p": int(pitcher_id), "c": cutoff})
    if df.empty:
        return pd.DataFrame(columns=["pitcher_id", "season", "unit", "metric", "value", "n", "age", "qualified"])
    df["unit"] = df["pitch_type"].map(pitch_family)
    bio = session.execute(text("SELECT birth_date FROM pitcher_bio WHERE pitcher_id = :p"), {"p": pitcher_id}).first()
    rows = []
    for (season, unit), g in df.groupby(["season", "unit"]):
        age = None
        if bio is not None and bio[0] is not None:
            from ..aging.families import age_years, season_midpoint
            age = age_years(bio[0], season_midpoint(int(season)))
        for m, col in (("plate_x_sd_ft", "plate_x"), ("plate_z_sd_ft", "plate_z")):
            rows.append({"pitcher_id": int(pitcher_id), "season": int(season), "unit": unit, "metric": m,
                         "value": float(g[col].std(ddof=1)) if len(g) > 1 else np.nan, "n": int(len(g)), "age": age,
                         "qualified": len(g) >= scaled_threshold(min_unit, int(season))})
    out = pd.DataFrame(rows)
    return out[out["value"].notna()].reset_index(drop=True)


def target_frame(panel: pd.DataFrame, t: Target, qualified_only: bool = True) -> pd.DataFrame:
    d = panel[(panel["unit"] == t.unit) & (panel["metric"] == t.metric)]
    if qualified_only:
        d = d[d["qualified"].astype(bool)]
    return d.sort_values(["pitcher_id", "season"]).reset_index(drop=True)


__all__ = ["Target", "targets", "location_targets", "build_panel", "load_panel", "location_panel", "target_frame",
           "FAMILY_METRICS", "ALL_METRICS", "LOCATION_METRICS", "DEFAULT_FAMILIES"]
