"""MLB-wide batter leaderboards -> as-of empirical-Bayes batter priors (next-pitch features v2).

Sources (Baseball Savant, every request through `PoliteClient`: 1 request/s per host + raw cache under data/raw).
Each endpoint and column was checked against a live response (2024) before this module was written:

* ``/leaderboard/pitch-arsenal-stats?type=batter&pitchType=&year=Y&team=&min=1&csv=true`` - one row per batter and
  pitch type: pitches, pa, whiff_percent, k_percent, put_away, woba, est_woba, run_value, run_value_per_100,
  hard_hit_percent. (A ``hand=`` parameter is ignored by this endpoint: the response is identical, so it gives no
  platoon split.)
* ``/leaderboard/custom?year=Y&type=batter&filter=&min=1&selections=...&csv=true`` - plate discipline: pa,
  pitch_count, k_percent, bb_percent, swing_percent, oz_swing_percent, z_swing_percent, whiff_percent,
  iz_contact_percent, oz_contact_percent, in_zone_percent, woba, xwoba.
* ``/statcast_search/csv?player_type=batter&group_by=name&pitcher_throws=L|R&hfSea=Y|&hfGT=R|`` - platoon split: a
  batter's regular-season line against pitchers of one hand: pitches, swings, whiffs, takes, pa, so, bb, woba,
  xwoba, batter_run_value_per_100.

As-of rule: the priors used for a pitch in season s are built from seasons s-LOOKBACK .. s-1 only (never season s or
later). Per batter, season lines are pooled by their denominators; each rate is then shrunk toward the league mean of
the same source seasons with an empirical-Bayes strength k estimated by the method of moments
(E[(x_i - mu)^2] = tau^2 + sigma^2 / n_i fitted by least squares weighted by n_i, k = sigma^2 / tau^2): x_hat = (n x + k mu) / (n + k). A batter without any
prior-season line gets the league mean and ``blb_missing = 1``.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sqlalchemy import delete, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from .aging.sources import SourceLog, _get_csv, _num
from .http import PoliteClient
from .logging_setup import log
from .models import Pitcher
from .models_batter_priors import (BatterDisciplineSeason, BatterPitchTypeSeason, BatterPlatoonSeason,
                                   BatterPriorAsOf, BatterPriorMeta)

logger = logging.getLogger(__name__)

SAVANT = "https://baseballsavant.mlb.com"
ARSENAL_STATS_URL = f"{SAVANT}/leaderboard/pitch-arsenal-stats"
CUSTOM_URL = f"{SAVANT}/leaderboard/custom"
SEARCH_URL = f"{SAVANT}/statcast_search/csv"
NAMESPACE = "savant_batter_leaderboards"
LOOKBACK = 3  # as-of season s uses seasons s-3 .. s-1
CUSTOM_SELECTIONS = ["pa", "pitch_count", "k_percent", "bb_percent", "swing_percent", "oz_swing_percent",
                     "z_swing_percent", "whiff_percent", "iz_contact_percent", "oz_contact_percent",
                     "in_zone_percent", "woba", "xwoba"]
K_MIN, K_MAX = 1.0, 5000.0
METHOD = ("pool seasons s-3..s-1 by denominator; empirical-Bayes shrinkage toward the league mean of the same seasons, "
          "k = sigma^2 / tau^2 from the method of moments on E[(x_i-mu)^2] = tau^2 + sigma^2/n_i, n-weighted least squares (k clipped to "
          f"[{K_MIN:g}, {K_MAX:g}])")

# (feature, kind, description). Per-class features are expanded for each model class that is a leaderboard code.
PER_CLASS_PRIORS = [
    ("blb_whiff_{c}", "Prior-season MLB whiff% vs {c} (EB-smoothed; n = pitches seen)"),
    ("blb_woba_{c}", "Prior-season MLB wOBA vs {c} (EB-smoothed; n = PA)"),
    ("blb_rv100_{c}", "Prior-season MLB batter run value per 100 {c} pitches (EB-smoothed)"),
    ("blb_k_{c}", "Prior-season MLB strikeout rate in PA ending on {c} (EB-smoothed)"),
]
DISCIPLINE_PRIORS = [
    ("blb_swing", "Prior-season MLB swing rate"),
    ("blb_o_swing", "Prior-season MLB chase (out-of-zone swing) rate"),
    ("blb_z_swing", "Prior-season MLB in-zone swing rate"),
    ("blb_contact", "Prior-season MLB contact rate on swings (1 - whiff%)"),
    ("blb_iz_contact", "Prior-season MLB in-zone contact rate"),
    ("blb_oz_contact", "Prior-season MLB out-of-zone contact rate"),
    ("blb_k", "Prior-season MLB strikeout rate"),
    ("blb_bb", "Prior-season MLB walk rate"),
    ("blb_woba", "Prior-season MLB wOBA"),
]
PLATOON_PRIORS = [
    ("blb_plat_whiff", "Prior-season MLB whiff rate per swing vs pitchers of this pitcher's hand"),
    ("blb_plat_swing", "Prior-season MLB swing rate vs pitchers of this pitcher's hand"),
    ("blb_plat_woba", "Prior-season MLB wOBA vs pitchers of this pitcher's hand"),
    ("blb_plat_k", "Prior-season MLB strikeout rate vs pitchers of this pitcher's hand"),
]
FLAG_PRIORS = [
    ("blb_missing", "Batter has no MLB line in the prior seasons (priors = league means)"),
    ("blb_log_pa", "log(1 + prior-season MLB PA) behind the priors"),
]
LEADERBOARD_CODE = {"KC": "CU", "CS": "CU"}  # Savant leaderboards fold KC/CS into CU


def prior_codes(classes: list[str]) -> list[str]:
    return [c for c in classes if c != "OTHER"]


def feature_names(classes: list[str]) -> list[str]:
    names = [t.format(c=c) for t, _ in PER_CLASS_PRIORS for c in prior_codes(classes)]
    return names + [n for n, _ in DISCIPLINE_PRIORS + PLATOON_PRIORS + FLAG_PRIORS]


# ------------------------------------------------------------------ fetch + load
def _fetch_pitch_type(client, season, slog, use_cache):
    params = {"type": "batter", "pitchType": "", "year": str(season), "team": "", "min": "1", "csv": "true"}
    return _get_csv(client, "batter_arsenal_stats", season, ARSENAL_STATS_URL, params, slog, use_cache)


def _fetch_custom(client, season, slog, use_cache):
    params = {"year": str(season), "type": "batter", "filter": "", "min": "1",
              "selections": ",".join(CUSTOM_SELECTIONS), "csv": "true"}
    return _get_csv(client, "batter_custom", season, CUSTOM_URL, params, slog, use_cache)


def _fetch_platoon(client, season, hand, slog, use_cache):
    params = {"all": "true", "hfGT": "R|", "hfPT": "", "hfSea": f"{season}|", "player_type": "batter",
              "pitcher_throws": hand, "group_by": "name", "min_pitches": "0", "min_results": "0", "min_pas": "0",
              "sort_col": "pitches", "sort_order": "desc"}
    return _get_csv(client, "batter_platoon", season, SEARCH_URL, params, slog, use_cache, hand)


def _pct(s):
    v = _num(s)
    return None if v is None else v / 100.0


def _rows(df: pd.DataFrame) -> list[dict]:
    out = []
    for r in df.to_dict("records"):
        out.append({k: (None if (isinstance(v, float) and math.isnan(v)) or v is pd.NA else
                        (v.item() if hasattr(v, "item") else v)) for k, v in r.items()})
    return out


def _replace(session: Session, model, season: int, rows: list[dict], extra_filter=None) -> int:
    q = delete(model).where(model.season == season)
    if extra_filter is not None:
        q = q.where(extra_filter)
    session.execute(q)
    for i in range(0, len(rows), 1000):
        session.execute(insert(model).values(rows[i:i + 1000]))
    return len(rows)


def load_leaderboards(session: Session, seasons: list[int], hands: list[str] | None = None,
                      client: PoliteClient | None = None, use_cache: bool = True) -> dict:
    """Fetch and store the three batter sources for `seasons`. Idempotent (replaces each season's rows)."""
    client = client or PoliteClient()
    hands = hands or pitcher_hands(session)
    slog = SourceLog()
    counts = {"pitch_type": {}, "discipline": {}, "platoon": {}}
    for season in seasons:
        f = _fetch_pitch_type(client, season, slog, use_cache)
        if f is not None and not f.frame.empty:
            d = f.frame
            df = pd.DataFrame({"batter_id": _num(d["player_id"]).astype("Int64"), "season": season,
                               "pitch_type": d["pitch_type"].astype(str).str.upper(),
                               "player_name": d.get("last_name, first_name"), "pitches": _num(d["pitches"]),
                               "pa": _num(d.get("pa")), "whiff_pct": _pct(d.get("whiff_percent")),
                               "k_pct": _pct(d.get("k_percent")), "put_away_pct": _pct(d.get("put_away")),
                               "woba": _num(d.get("woba")), "est_woba": _num(d.get("est_woba")),
                               "run_value": _num(d.get("run_value")),
                               "run_value_per_100": _num(d.get("run_value_per_100")),
                               "hard_hit_pct": _pct(d.get("hard_hit_percent")),
                               "source_url": f.url, "fetched_at": f.fetched_at}).dropna(subset=["batter_id"])
            df = df.drop_duplicates(["batter_id", "season", "pitch_type"])
            counts["pitch_type"][season] = _replace(session, BatterPitchTypeSeason, season, _rows(df))
        f = _fetch_custom(client, season, slog, use_cache)
        if f is not None and not f.frame.empty:
            d = f.frame
            df = pd.DataFrame({"batter_id": _num(d["player_id"]).astype("Int64"), "season": season,
                               "player_name": d.get("last_name, first_name"), "pa": _num(d.get("pa")),
                               "pitch_count": _num(d.get("pitch_count")), "k_pct": _pct(d.get("k_percent")),
                               "bb_pct": _pct(d.get("bb_percent")), "swing_pct": _pct(d.get("swing_percent")),
                               "o_swing_pct": _pct(d.get("oz_swing_percent")),
                               "z_swing_pct": _pct(d.get("z_swing_percent")),
                               "whiff_pct": _pct(d.get("whiff_percent")),
                               "iz_contact_pct": _pct(d.get("iz_contact_percent")),
                               "oz_contact_pct": _pct(d.get("oz_contact_percent")),
                               "in_zone_pct": _pct(d.get("in_zone_percent")), "woba": _num(d.get("woba")),
                               "xwoba": _num(d.get("xwoba")), "source_url": f.url,
                               "fetched_at": f.fetched_at}).dropna(subset=["batter_id"])
            df = df.drop_duplicates(["batter_id", "season"])
            counts["discipline"][season] = _replace(session, BatterDisciplineSeason, season, _rows(df))
        for hand in hands:
            f = _fetch_platoon(client, season, hand, slog, use_cache)
            if f is None or f.frame.empty:
                continue
            d = f.frame
            df = pd.DataFrame({"batter_id": _num(d["player_id"]).astype("Int64"), "season": season,
                               "pitcher_throws": hand, "player_name": d.get("player_name"),
                               "pitches": _num(d["pitches"]), "swings": _num(d.get("swings")),
                               "whiffs": _num(d.get("whiffs")), "takes": _num(d.get("takes")),
                               "pa": _num(d.get("pa")), "so": _num(d.get("so")), "bb": _num(d.get("bb")),
                               "woba": _num(d.get("woba")), "xwoba": _num(d.get("xwoba")),
                               "run_value_per_100": _num(d.get("batter_run_value_per_100")),
                               "source_url": f.url, "fetched_at": f.fetched_at}).dropna(subset=["batter_id"])
            df = df.drop_duplicates(["batter_id", "season", "pitcher_throws"])
            counts["platoon"].setdefault(hand, {})[season] = _replace(
                session, BatterPlatoonSeason, season, _rows(df), BatterPlatoonSeason.pitcher_throws == hand)
    session.flush()
    rep = {"seasons": list(seasons), "hands": hands, "rows": counts, "sources": slog.as_dict()}
    log(logger, "batter_leaderboards_loaded", **{k: v for k, v in rep.items() if k != "sources"})
    return rep


def pitcher_hands(session: Session) -> list[str]:
    hands = sorted({h for h in session.scalars(select(Pitcher.throws)).all() if h in ("L", "R")})
    return hands or ["L", "R"]


# ------------------------------------------------------------------ empirical Bayes
def eb_strength(x: np.ndarray, n: np.ndarray) -> tuple[float, float, dict]:
    """Return (mu, k, diagnostics) for unit rates x_i with denominators n_i (> 0)."""
    m = np.isfinite(x) & np.isfinite(n) & (n > 0)
    x, n = x[m], n[m]
    if len(x) == 0:
        return float("nan"), K_MAX, {"n_units": 0}
    mu = float(np.sum(x * n) / np.sum(n))
    if len(x) < 10:
        return mu, K_MAX, {"n_units": int(len(x)), "note": "too few units; maximal shrinkage"}
    dev2 = (x - mu) ** 2
    # weighted least squares (weights n_i): small-sample units are noisy and would otherwise dominate the fit
    sw = np.sqrt(n)
    A = np.column_stack([np.ones_like(n), 1.0 / n]) * sw[:, None]
    (tau2, sigma2), *_ = np.linalg.lstsq(A, dev2 * sw, rcond=None)
    if sigma2 <= 0:
        k = K_MIN
    elif tau2 <= 0:
        k = K_MAX
    else:
        k = float(np.clip(sigma2 / tau2, K_MIN, K_MAX))
    return mu, k, {"n_units": int(len(x)), "tau2": float(tau2), "sigma2": float(sigma2)}


def _shrink(x, n, mu, k):
    x = np.where(np.isfinite(x) & (n > 0), x, mu)
    n = np.where(np.isfinite(n), n, 0.0)
    return (n * x + k * mu) / (n + k)


def _pool(df: pd.DataFrame, keys: list[str], specs: dict[str, tuple[str, str]]) -> pd.DataFrame:
    """specs: out_name -> (rate column, denominator column). Pools seasons by denominator."""
    parts = {}
    d = df.copy()
    for out, (rate, den) in specs.items():
        ok = d[rate].notna() & d[den].notna() & (d[den] > 0)
        d[f"_num_{out}"] = np.where(ok, d[rate] * d[den], 0.0)
        d[f"_den_{out}"] = np.where(ok, d[den], 0.0)
    g = d.groupby(keys)
    for out in specs:
        num, den = g[f"_num_{out}"].sum(), g[f"_den_{out}"].sum()
        parts[out] = (num / den.replace(0, np.nan))
        parts[f"n_{out}"] = den
    return pd.DataFrame(parts).reset_index()


def build_priors(session: Session, as_of_seasons: list[int], hands: list[str] | None = None,
                 classes: list[str] | None = None) -> dict:
    """Compute and store priors for each as-of season from the seasons before it (idempotent)."""
    hands = hands or pitcher_hands(session)
    pt = pd.read_sql(select(BatterPitchTypeSeason), session.connection())
    dc = pd.read_sql(select(BatterDisciplineSeason), session.connection())
    pl = pd.read_sql(select(BatterPlatoonSeason), session.connection())
    codes = sorted(set(pt["pitch_type"])) if not pt.empty else []
    if classes:
        codes = sorted({LEADERBOARD_CODE.get(c, c) for c in prior_codes(classes)} & set(codes)) or codes
    out = {}
    for s in as_of_seasons:
        src = list(range(s - LOOKBACK, s))
        P, D = pt[pt.season.isin(src)], dc[dc.season.isin(src)]
        for hand in hands:
            L = pl[pl.season.isin(src) & (pl.pitcher_throws == hand)]
            present = sorted({int(x) for x in pd.concat([P.season, D.season, L.season]).unique()})
            session.execute(delete(BatterPriorAsOf).where(BatterPriorAsOf.as_of_season == s,
                                                          BatterPriorAsOf.pitcher_throws == hand))
            session.execute(delete(BatterPriorMeta).where(BatterPriorMeta.as_of_season == s,
                                                          BatterPriorMeta.pitcher_throws == hand))
            if not present:
                out[(s, hand)] = {"as_of_season": s, "hand": hand, "source_seasons": [], "n_batters": 0}
                continue
            feats: dict[int, dict] = {}
            league, shrink = {}, {}

            def apply(frame: pd.DataFrame, name: str, rate: str, den: str):
                if frame.empty or rate not in frame:
                    return
                x, n = frame[rate].to_numpy(float), frame[den].to_numpy(float)
                mu, k, diag = eb_strength(x, n)
                league[name], shrink[name] = mu, {"k": k, **diag}
                vals = _shrink(x, n, mu, k)
                for b, v in zip(frame["batter_id"].astype(int), vals):
                    feats.setdefault(int(b), {})[name] = float(v)

            # per pitch type
            if not P.empty:
                pool = _pool(P, ["batter_id", "pitch_type"], {"whiff": ("whiff_pct", "pitches"),
                                                               "woba": ("woba", "pa"),
                                                               "rv100": ("run_value_per_100", "pitches"),
                                                               "k": ("k_pct", "pa")})
                for c in codes:
                    sub = pool[pool.pitch_type == c]
                    for m in ("whiff", "woba", "rv100", "k"):
                        apply(sub, f"blb_{m}_{c}", m, f"n_{m}")
            if not D.empty:
                D = D.copy()
                D["n_swings"] = D.pitch_count * D.swing_pct
                D["n_oz"] = D.pitch_count * (1 - D.in_zone_pct)
                D["n_iz"] = D.pitch_count * D.in_zone_pct
                D["n_iz_sw"] = D.n_iz * D.z_swing_pct
                D["n_oz_sw"] = D.n_oz * D.o_swing_pct
                D["contact_pct"] = 1 - D.whiff_pct
                pool = _pool(D, ["batter_id"], {"swing": ("swing_pct", "pitch_count"),
                                                "o_swing": ("o_swing_pct", "n_oz"),
                                                "z_swing": ("z_swing_pct", "n_iz"),
                                                "contact": ("contact_pct", "n_swings"),
                                                "iz_contact": ("iz_contact_pct", "n_iz_sw"),
                                                "oz_contact": ("oz_contact_pct", "n_oz_sw"),
                                                "k": ("k_pct", "pa"), "bb": ("bb_pct", "pa"),
                                                "woba": ("woba", "pa"), "pa": ("pa", "pa")})
                for m in ("swing", "o_swing", "z_swing", "contact", "iz_contact", "oz_contact", "k", "bb", "woba"):
                    apply(pool, f"blb_{m}", m, f"n_{m}")
                for b, npa in zip(pool.batter_id.astype(int), pool.n_pa):
                    feats.setdefault(int(b), {})["_pa"] = float(npa)
            if not L.empty:
                L = L.copy()
                L["whiff_rate"] = L.whiffs / L.swings.replace(0, np.nan)
                L["swing_rate"] = L.swings / L.pitches.replace(0, np.nan)
                L["k_rate"] = L.so / L.pa.replace(0, np.nan)
                pool = _pool(L, ["batter_id"], {"whiff": ("whiff_rate", "swings"), "swing": ("swing_rate", "pitches"),
                                                "woba": ("woba", "pa"), "k": ("k_rate", "pa")})
                for m in ("whiff", "swing", "woba", "k"):
                    apply(pool, f"blb_plat_{m}", m, f"n_{m}")
            rows = []
            for b, f in feats.items():
                npa = f.pop("_pa", 0.0)
                f = {k: f.get(k, league[k]) for k in league}  # a source missing for this batter -> league mean
                f["blb_missing"] = 0
                f["blb_log_pa"] = float(np.log1p(npa))
                rows.append({"batter_id": b, "as_of_season": s, "pitcher_throws": hand, "features": f,
                             "n_prior_pa": float(npa), "source_seasons": present})
            for i in range(0, len(rows), 1000):
                session.execute(insert(BatterPriorAsOf).values(rows[i:i + 1000]))
            session.add(BatterPriorMeta(as_of_season=s, pitcher_throws=hand, source_seasons=present,
                                        league_means=league, shrinkage=shrink, n_batters=len(rows), method=METHOD))
            out[(s, hand)] = {"as_of_season": s, "hand": hand, "source_seasons": present, "n_batters": len(rows),
                              "k": {k: round(v["k"], 1) for k, v in shrink.items()}}
    session.flush()
    rep = {f"{s}-{h}": v for (s, h), v in out.items()}
    log(logger, "batter_priors_built", as_of=as_of_seasons, hands=hands,
        n={k: v["n_batters"] for k, v in rep.items()})
    return rep


# ------------------------------------------------------------------ lookup used by make_features
@dataclass
class BatterPriors:
    """In-memory as-of lookup: features for (batter, season) from seasons < season only."""
    throws: str
    rows: dict = field(default_factory=dict)      # (batter_id, as_of_season) -> features
    league: dict = field(default_factory=dict)    # as_of_season -> league-mean features
    classes: list = field(default_factory=list)

    def lookup(self, batter_id, season) -> dict:
        names = feature_names(self.classes)
        season = int(season) if season is not None else None
        if season not in self.league:  # no prior-season data loaded for this season: explicitly missing
            return {n: (1 if n == "blb_missing" else None) for n in names}
        f = self.rows.get((int(batter_id), season)) if batter_id is not None else None
        if f is None:
            base = self.league[season]
            return {n: (1 if n == "blb_missing" else 0.0 if n == "blb_log_pa" else
                        _code_value(base, n)) for n in names}
        return {n: _code_value(f, n) for n in names}


def _code_value(f: dict, name: str):
    if name in f:
        return f[name]
    for src, dst in LEADERBOARD_CODE.items():  # model class KC/CS -> leaderboard CU
        if name.endswith("_" + src):
            return f.get(name[: -len(src)] + dst)
    return None


def load_priors(session: Session, pitcher_id: int, classes: list[str]) -> BatterPriors | None:
    p = session.get(Pitcher, pitcher_id)
    throws = (p.throws if p is not None else None) or "L"
    meta = session.scalars(select(BatterPriorMeta).where(BatterPriorMeta.pitcher_throws == throws)).all()
    if not meta:
        return None
    bp = BatterPriors(throws=throws, classes=list(classes))
    for m in meta:
        bp.league[int(m.as_of_season)] = {**m.league_means, "blb_missing": 1, "blb_log_pa": 0.0}
    for r in session.execute(select(BatterPriorAsOf.batter_id, BatterPriorAsOf.as_of_season, BatterPriorAsOf.features)
                             .where(BatterPriorAsOf.pitcher_throws == throws)).all():
        bp.rows[(int(r.batter_id), int(r.as_of_season))] = r.features
    return bp


def refresh(session: Session, pitcher_id: int, seasons: list[int] | None = None, client: PoliteClient | None = None,
            use_cache: bool = True) -> dict:
    """Full ETL for the configured modeling seasons: load leaderboards for s-LOOKBACK..max(s), build as-of priors."""
    from .seasons import modeling_seasons
    from .features.store import get_classes
    seasons = sorted(seasons or modeling_seasons())
    lb_seasons = list(range(min(seasons) - LOOKBACK, max(seasons) + 1))
    hands = pitcher_hands(session)
    rep = {"leaderboards": load_leaderboards(session, lb_seasons, hands, client, use_cache)}
    try:
        classes = get_classes(session, pitcher_id)
    except Exception:  # noqa: BLE001 - features not built yet
        classes = None
    as_of = sorted(set(seasons) | {max(seasons) + 1})
    rep["priors"] = build_priors(session, as_of, hands, classes)
    rep["built_at"] = datetime.now(timezone.utc).isoformat()
    return rep
