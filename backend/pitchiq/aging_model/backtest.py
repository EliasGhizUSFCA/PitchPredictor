"""Rolling-origin backtest, empirical intervals, metrics and per-target method selection.

For every target and historical cutoff c (origins 2016..2025 by default) each method is fitted on qualified rows with
season <= c and forecasts every unit that qualified in season c for c+1, c+2, c+3. Forecasts are scored against the
qualified value observed in the target season (units that did not qualify in the target season have no truth and are
not scored: the evaluation is conditional on continuation, like any forecast of a measured season).

Intervals: lgbm_quantile and eb_kalman have their own predictive quantiles, conformalized (CQR) with the same
earlier-origin rule because their raw 10/90 % quantiles were too narrow out of sample. Methods without their own
predictive distribution get empirical intervals from their own residuals at
earlier origins (origin c' with c' + h <= c, i.e. outcomes known at the cutoff; same horizon; the unit's origin age
band when it has >= 150 residuals). Before enough residuals exist a Gaussian with the sd of observed h-season changes
up to the cutoff is used.

Selection (one method per target, used for every horizon): lowest mean over horizons 1-3 of the method's mean
pinball loss (q10/q50/q90) relative to no_change, on development target seasons 2018-2025, excluding target season
2020 (60 games) and forecasts that cross the TrackMan -> Hawk-Eye change, restricted to units aged >= 32 at the origin
(the band of the forecast pitcher); a horizon falls back to all ages when that subset has < 100 rows.
Target season 2026 is reported separately as retrospective (previously viewed) and never used for selection.
Methods are compared on their common support (rows every method forecast).
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..logging_setup import log
from .methods import (METHODS, NATIVE_DISTRIBUTION, QUANTILES, Z90, OriginData, consecutive_pairs, crosses_era,
                      fitted_state, run_method)
from .panel import Target, target_frame

logger = logging.getLogger(__name__)

DEV_LAST_TARGET = 2025
RETRO_SEASONS = (2026,)
FIRST_SCORED_TARGET = 2018
SELECTION_MIN_AGE = 32.0
AGE_BANDS = [(0, 26.999, "<=26"), (27, 30.999, "27-30"), (31, 33.999, "31-33"), (34, 36.999, "34-36"),
             (37, 99, "37+")]
CONFORMAL_BANDS = [(0, 26.999, "<=26"), (27, 30.999, "27-30"), (31, 33.999, "31-33"), (34, 99, "34+")]
CONFORMAL_MIN_BAND = 150
CONFORMAL_MIN = 30
FORECAST_QUANTILES = (0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95)


def band(age: float, bands=AGE_BANDS) -> str | None:
    if age is None or not np.isfinite(age):
        return None
    for lo, hi, lab in bands:
        if lo <= age <= hi:
            return lab
    return None


@dataclass
class BacktestConfig:
    origins: tuple[int, ...] = tuple(range(2016, 2026))
    horizons: tuple[int, ...] = (1, 2, 3)
    methods: tuple[str, ...] = tuple(METHODS)
    first_scored_target: int = FIRST_SCORED_TARGET
    dev_last_target: int = DEV_LAST_TARGET
    selection_min_age: float = SELECTION_MIN_AGE
    final_cutoff: int | None = 2026

    def to_dict(self) -> dict:
        return {k: list(v) if isinstance(v, tuple) else v for k, v in self.__dict__.items()}


def change_sd(hist: pd.DataFrame, h: int) -> float:
    """SD of observed h-season within-pitcher changes (same era, no 2020) in `hist` (fallback interval scale)."""
    pr = consecutive_pairs(hist, k=h)
    if len(pr) >= 20:
        return float(pr["delta"].std())
    pr1 = consecutive_pairs(hist, k=1)
    if len(pr1) >= 20:
        return float(pr1["delta"].std() * np.sqrt(h))
    return float(hist["value"].std()) if len(hist) > 1 else np.nan


CQR_METHODS = ("lgbm_quantile", "eb_kalman")


def conformalize_quantiles(fc: pd.DataFrame, pool: pd.DataFrame, cutoff_known: int, coverage: float = 0.8):
    """Conformalized quantile regression (Romano et al. 2019) for quantile models whose own 10/90 % quantiles are
    too narrow out of sample: conformity score E = max(q10 - y, y - q90) of earlier forecasts (same method and
    horizon, outcome known by `cutoff_known`); the interval is widened (or narrowed) by the (coverage)-quantile of E.
    The raw model quantiles are kept in q10_raw / q90_raw."""
    fc = fc.copy()
    if pool.empty or "cqr" not in pool.columns:
        return fc
    pool = pool[(pool["target_season"] <= cutoff_known) & pool["cqr"].notna()]
    for (m, h), g in fc[fc["method"].isin(CQR_METHODS)].groupby(["method", "h"]):
        e = pool.loc[(pool["method"] == m) & (pool["h"] == h), "cqr"].astype(float).values
        if len(e) < CONFORMAL_MIN:
            continue
        n = len(e)
        qhat = float(np.quantile(e, min(1.0, coverage * (1 + 1 / n))))
        fc.loc[g.index, "q10_raw"] = fc.loc[g.index, "q10"]
        fc.loc[g.index, "q90_raw"] = fc.loc[g.index, "q90"]
        fc.loc[g.index, "q10"] = np.minimum(fc.loc[g.index, "q10"] - qhat, fc.loc[g.index, "q50"])
        fc.loc[g.index, "q90"] = np.maximum(fc.loc[g.index, "q90"] + qhat, fc.loc[g.index, "q50"])
        fc.loc[g.index, "interval_source"] = "native_cqr"
        if "sd" in fc.columns and fc.loc[g.index, "sd"].notna().all():
            # Gaussian model: keep it Gaussian with the conformalized 80 % width
            from scipy.stats import norm
            sd = ((fc.loc[g.index, "q90"] - fc.loc[g.index, "q10"]) / (2 * Z90)).clip(lower=1e-9)
            fc.loc[g.index, "sd_raw"] = fc.loc[g.index, "sd"]
            fc.loc[g.index, "sd"] = sd
            for c in [c for c in fc.columns if len(c) == 3 and c.startswith("q") and c[1:].isdigit()]:
                fc.loc[g.index, c] = fc.loc[g.index, "point"] + norm.ppf(int(c[1:]) / 100) * sd
        fc.loc[g.index, "cqr_qhat"] = qhat
        fc.loc[g.index, "n_resid"] = n
    return fc


def add_intervals(fc: pd.DataFrame, pool: pd.DataFrame, hist: pd.DataFrame, cutoff_known: int,
                  levels=QUANTILES) -> pd.DataFrame:
    """Empirical quantiles for non-native methods. `pool`: scored earlier forecasts (method, h, origin_season,
    target_season, origin_age, resid); only rows with target_season <= cutoff_known are used."""
    fc = fc.copy()
    pool = pool[pool["target_season"] <= cutoff_known] if not pool.empty else pool
    fc["interval_source"] = np.where(fc["method"].isin(NATIVE_DISTRIBUTION), "native", "")
    from scipy.stats import norm
    for (m, h), g in fc[~fc["method"].isin(NATIVE_DISTRIBUTION)].groupby(["method", "h"]):
        p = pool[(pool["method"] == m) & (pool["h"] == h)] if not pool.empty else pool
        sd = None
        bands = g["origin_age"].map(lambda a: band(a, CONFORMAL_BANDS))
        for b in bands.dropna().unique().tolist() + [None]:
            idx = g.index[bands == b] if b is not None else g.index[bands.isna()]
            if len(idx) == 0:
                continue
            pb = p[p["band"] == b] if b is not None and len(p) else p.iloc[0:0]
            src = None
            if len(pb) >= CONFORMAL_MIN_BAND:
                res, src = pb["resid"].values.astype(float), f"empirical_band:{b}"
            elif len(p) >= CONFORMAL_MIN:
                res, src = p["resid"].values.astype(float), "empirical_all_ages"
            if src:
                # split-conformal style: the method's point is the median; other quantiles add residual quantiles
                for q in levels:
                    off = float(np.quantile(res, q))
                    # the method's point stays the median; residual quantiles on the wrong side are clipped to 0
                    off = 0.0 if q == 0.5 else (min(off, 0.0) if q < 0.5 else max(off, 0.0))
                    fc.loc[idx, f"q{int(round(q * 100)):02d}"] = fc.loc[idx, "point"] + off
                fc.loc[idx, "n_resid"] = len(res)
            else:
                if sd is None:
                    sd = change_sd(hist, int(h))
                for q in levels:
                    fc.loc[idx, f"q{int(round(q * 100)):02d}"] = fc.loc[idx, "point"] + norm.ppf(q) * sd
                fc.loc[idx, "sd"] = sd
                src = "gaussian_change_sd"
            fc.loc[idx, "interval_source"] = src
    qcols = [f"q{int(round(q * 100)):02d}" for q in levels]
    if "sd" in fc.columns:  # native Gaussian (eb_kalman): every requested level from mean / sd
        gauss = fc["method"].isin(NATIVE_DISTRIBUTION) & fc["sd"].notna()
        for q in levels:
            c = f"q{int(round(q * 100)):02d}"
            if c not in fc.columns:
                fc[c] = np.nan
            fc.loc[gauss, c] = fc.loc[gauss, "point"] + norm.ppf(q) * fc.loc[gauss, "sd"]
    have = [c for c in qcols if c in fc.columns]
    for cols in (have, ["q10", "q50", "q90"]):
        cols = [c for c in cols if c in fc.columns]
        if not cols:
            continue
        v = fc[cols].to_numpy(dtype=float, copy=True)
        ok = ~np.isnan(v).any(axis=1)
        v[ok] = np.sort(v[ok], axis=1)
        fc[cols] = v
    return fc


def pinball(y, q, tau):
    d = y - q
    return np.maximum(tau * d, (tau - 1) * d)


def score(df: pd.DataFrame) -> dict:
    y, m = df["actual"].values, df["q50"].values
    e = m - y
    cov = (df["q10"].values <= y) & (y <= df["q90"].values)
    pb = {f"pinball_q{int(t * 100):02d}": float(np.mean(pinball(y, df[f"q{int(t * 100):02d}"].values, t)))
          for t in QUANTILES}
    return {"n": int(len(df)), "mae": float(np.mean(np.abs(e))), "rmse": float(np.sqrt(np.mean(e ** 2))),
            "bias": float(np.mean(e)), "coverage80": float(np.mean(cov)),
            "width80": float(np.mean(df["q90"].values - df["q10"].values)), **pb,
            "pinball_mean": float(np.mean(list(pb.values())))}


def subsets(df: pd.DataFrame, cfg: BacktestConfig) -> dict[str, pd.Series]:
    ts, os_ = df["target_season"], df["origin_season"]
    scored = ts >= cfg.first_scored_target
    dev = scored & (ts <= cfg.dev_last_target)
    clean = ~crosses_era(os_, ts) & (ts != 2020)
    out = {"all_targets": scored, "dev": dev, "dev_same_era": dev & clean,
           "selection": dev & clean & (df["origin_age"] >= cfg.selection_min_age)}
    for s in RETRO_SEASONS:
        out[f"retro_{s}"] = ts == s
    return out


def metrics_table(scored: pd.DataFrame, cfg: BacktestConfig, target_pid: int | None) -> pd.DataFrame:
    """Rows: method, h, breakdown_type, breakdown_key, metrics. Common support across methods per (h)."""
    rows = []
    key = ["pitcher_id", "origin_season", "h"]
    nm = scored["method"].nunique()
    full = scored.groupby(key)["method"].transform("nunique") == nm
    common = scored[full]
    for h in sorted(common["h"].unique()):
        d = common[common["h"] == h]
        sub = subsets(d, cfg)
        groups: list[tuple[str, str, pd.Series]] = [("subset", k, v) for k, v in sub.items()]
        ab = d["origin_age"].map(band)
        for lo, hi, lab in AGE_BANDS:
            groups.append(("age_band_dev_same_era", lab, sub["dev_same_era"] & (ab == lab)))
        for s in sorted(d["target_season"].unique()):
            if s >= cfg.first_scored_target:
                groups.append(("target_season", str(int(s)), d["target_season"] == s))
        if target_pid is not None:
            groups.append(("target_pitcher", str(target_pid), sub["all_targets"] & (d["pitcher_id"] == target_pid)))
            groups.append(("target_pitcher_dev", str(target_pid), sub["dev"] & (d["pitcher_id"] == target_pid)))
        for btype, bkey, mask in groups:
            g = d[mask]
            if g.empty:
                continue
            for m, gm in g.groupby("method"):
                rows.append({"method": m, "h": int(h), "breakdown_type": btype, "breakdown_key": bkey,
                             **score(gm)})
    # each method's own support (no common-support restriction), development rows
    for (m, h), g in scored.groupby(["method", "h"]):
        g = g[subsets(g, cfg)["dev"]]
        if len(g):
            rows.append({"method": m, "h": int(h), "breakdown_type": "own_support", "breakdown_key": "dev",
                         **score(g)})
    return pd.DataFrame(rows)


def select(metrics: pd.DataFrame, min_rows: int = 100) -> dict:
    """One method per target: lowest mean over horizons of (method pinball / no_change pinball) on the selection
    subset (a horizon falls back to dev_same_era when the selection subset has < min_rows rows). Returned per horizon
    (same method for every horizon, so the forecast trajectory and Monte Carlo paths come from one model), with each
    horizon's ranking and its own best method for reference."""
    per_h = {}
    for h, g in metrics[metrics["breakdown_type"] == "subset"].groupby("h"):
        sel = g[g["breakdown_key"] == "selection"]
        basis = "selection"
        if sel.empty or sel["n"].min() < min_rows:
            sel, basis = g[g["breakdown_key"] == "dev_same_era"], "dev_same_era"
        if not sel.empty:
            per_h[int(h)] = (sel.set_index("method"), basis)
    if not per_h:
        return {}
    common = set.intersection(*[set(v[0].index) for v in per_h.values()])
    if not common:
        return {}

    def ref(t):
        return float(t.loc["no_change", "pinball_mean"]) if "no_change" in t.index else float(t["pinball_mean"].mean())
    rel = {m: float(np.mean([v[0].loc[m, "pinball_mean"] / ref(v[0]) for v in per_h.values()])) for m in common}
    best = min(sorted(rel), key=rel.get)
    out = {}
    for h, (t, basis) in per_h.items():
        t = t.sort_values(["pinball_mean", "mae"])
        nc = float(t.loc["no_change", "pinball_mean"]) if "no_change" in t.index else None
        out[h] = {"method": best, "basis": basis, "n": int(t.loc[best, "n"]),
                  "pinball_mean": float(t.loc[best, "pinball_mean"]), "mae": float(t.loc[best, "mae"]),
                  "coverage80": float(t.loc[best, "coverage80"]), "no_change_pinball_mean": nc,
                  "relative_pinball_mean_over_horizons": rel[best], "horizon_best_method": str(t.index[0]),
                  "relative_pinball_by_method": rel,
                  "ranking": [{"method": m, "pinball_mean": float(r["pinball_mean"]), "mae": float(r["mae"]),
                               "coverage80": float(r["coverage80"])} for m, r in t.iterrows()]}
    return out


def _attach_truth(fc: pd.DataFrame, d: pd.DataFrame) -> pd.DataFrame:
    truth = d.set_index(["pitcher_id", "season"])
    idx = pd.MultiIndex.from_arrays([fc["pitcher_id"], fc["target_season"]])
    fc = fc.copy()
    fc["actual"] = truth["value"].reindex(idx).values
    fc["actual_n"] = truth["n"].reindex(idx).values
    return fc


def backtest_target(panel_t: pd.DataFrame, target: Target, cfg: BacktestConfig,
                    target_pid: int | None = None) -> dict:
    """Backtest one target and (if cfg.final_cutoff) forecast the target pitcher from the final cutoff.

    `panel_t`: qualified rows of this target (all seasons). Returns metrics, selection, the target pitcher's backtest
    rows, the final forecast rows (all methods) and the fitted state at the final cutoff."""
    t0 = time.time()
    d = panel_t.sort_values(["pitcher_id", "season"]).reset_index(drop=True)
    seasons_loaded = set(d["season"].unique())
    pool = pd.DataFrame(columns=["method", "h", "origin_season", "target_season", "origin_age", "band", "resid",
                                 "cqr"])
    scored_all, tp_rows, failures = [], [], []
    for c in cfg.origins:
        if c not in seasons_loaded:
            continue
        od = OriginData(target, c, d[d["season"] <= c], tuple(cfg.horizons))
        parts = []
        for m in cfg.methods:
            try:
                parts.append(run_method(m, od))
            except Exception as e:  # noqa: BLE001 - recorded; a failed fit leaves that method out at this origin
                failures.append({"origin": c, "method": m, "error": f"{type(e).__name__}: {e}"[:300]})
        if not parts:
            continue
        fc = pd.concat([p for p in parts if not p.empty], ignore_index=True)
        fc = fc[np.isfinite(fc["point"].astype(float))]
        fc = add_intervals(fc, pool, od.hist, cutoff_known=c)
        fc = conformalize_quantiles(fc, pool, cutoff_known=c)
        fc = _attach_truth(fc, d)
        if target_pid is not None:
            tp_rows.append(fc[fc["pitcher_id"] == target_pid])
        sc = fc[fc["actual"].notna()].copy()
        sc["resid"] = sc["actual"] - sc["point"]
        lo = sc["q10_raw"] if "q10_raw" in sc.columns else sc["q10"]
        hi = sc["q90_raw"] if "q90_raw" in sc.columns else sc["q90"]
        lo, hi = lo.where(lo.notna(), sc["q10"]), hi.where(hi.notna(), sc["q90"])
        sc["cqr"] = np.where(sc["method"].isin(CQR_METHODS), np.maximum(lo - sc["actual"], sc["actual"] - hi),
                             np.nan)
        sc["band"] = sc["origin_age"].map(lambda a: band(a, CONFORMAL_BANDS))
        pool = pd.concat([pool, sc[["method", "h", "origin_season", "target_season", "origin_age", "band", "resid",
                                    "cqr"]]],
                         ignore_index=True)
        scored_all.append(sc)
    scored = pd.concat(scored_all, ignore_index=True) if scored_all else pd.DataFrame()
    metrics = metrics_table(scored, cfg, target_pid) if not scored.empty else pd.DataFrame()
    selection = select(metrics) if not metrics.empty else {}
    out = {"target": target.key, "metrics": metrics, "selection": selection,
           "target_pitcher_rows": pd.concat(tp_rows, ignore_index=True) if tp_rows else pd.DataFrame(),
           "failures": failures, "n_scored": int(len(scored)), "final": None, "fitted_state": None,
           "residual_pool_dev_n": int((pool["target_season"] <= cfg.dev_last_target).sum())}
    if cfg.final_cutoff is not None and target_pid is not None and cfg.final_cutoff in seasons_loaded:
        out.update(final_forecast(d, target, cfg, target_pid, pool))
    out["seconds"] = round(time.time() - t0, 1)
    log(logger, "aging_backtest_target", target=target.key, n_scored=out["n_scored"], seconds=out["seconds"])
    return out


def final_forecast(d: pd.DataFrame, target: Target, cfg: BacktestConfig, target_pid: int,
                   pool: pd.DataFrame) -> dict:
    c = int(cfg.final_cutoff)
    od = OriginData(target, c, d[d["season"] <= c], tuple(cfg.horizons), pitchers={target_pid})
    # Tuning (Marcel k) and interval calibration use development outcomes only (target seasons <= 2025).
    od.cache["tuning_max_origin"] = cfg.dev_last_target - 1
    if od.origin.empty:
        return {"final": pd.DataFrame(), "fitted_state": None, "final_failures": []}
    parts, fails = [], []
    for m in cfg.methods:
        try:
            parts.append(run_method(m, od))
        except Exception as e:  # noqa: BLE001
            fails.append({"origin": c, "method": m, "error": f"{type(e).__name__}: {e}"[:300]})
    fc = pd.concat([p for p in parts if not p.empty], ignore_index=True)
    fc = add_intervals(fc, pool, od.hist[od.hist["season"] <= cfg.dev_last_target], cutoff_known=cfg.dev_last_target,
                       levels=FORECAST_QUANTILES)
    fc = conformalize_quantiles(fc, pool, cutoff_known=cfg.dev_last_target)
    state = fitted_state(od)
    lgbm_models = od.cache.get("lgbm")
    return {"final": fc, "fitted_state": state, "final_failures": fails,
            "lgbm_models": {str(q): m.booster_.model_to_string() for q, m in (lgbm_models or {}).items()}}


# --------------------------------------------------------------------------- target pitcher's location consistency
LOCATION_METHODS = ("no_change", "recent3", "career_mean")


def _loc_point(g: pd.DataFrame, c: int, method: str) -> float | None:
    h = g[g["season"] <= c]
    if h.empty or c not in set(h["season"]):
        return None
    if method == "no_change":
        return float(h.loc[h["season"] == c, "value"].iloc[0])
    if method == "career_mean":
        return float(np.average(h["value"], weights=h["n"]))
    r = h[h["season"] >= c - 2]
    w = r["season"].map({c: 5.0, c - 1: 4.0, c - 2: 3.0}) * r["n"]
    return float(np.average(r["value"], weights=w))


def location_backtest(loc: pd.DataFrame, origins=tuple(range(2016, 2026)), horizons=(1, 2, 3),
                      dev_last_target: int = DEV_LAST_TARGET, final_cutoff: int | None = 2026) -> dict:
    """Own-history forecasts of the target pitcher's plate-location dispersion (no population pitch-level data).

    Interval: Gaussian with sd = sqrt(h) x SD of his one-season changes of the metric (pooled over families, same
    era, no 2020) with target season <= the outcome-known season. Selection per metric (pooled over families and
    horizons) by MAE on development targets."""
    from scipy.stats import norm
    q = loc[loc["qualified"].astype(bool)].copy()
    rows = []

    def sd_for(metric: str, known: int) -> float | None:
        ch = []
        for _, g in q[(q["metric"] == metric) & (q["season"] <= known)].groupby("unit"):
            s = g.set_index("season")["value"]
            for t in s.index:
                if t + 1 in s.index and not crosses_era(t, t + 1) and 2020 not in (t, t + 1):
                    ch.append(s[t + 1] - s[t])
        return float(np.std(ch, ddof=1)) if len(ch) >= 3 else None

    for (unit, metric), g in q.groupby(["unit", "metric"]):
        tv = g.set_index("season")["value"]
        for c in list(origins) + ([final_cutoff] if final_cutoff else []):
            final = final_cutoff is not None and c == final_cutoff
            known = dev_last_target if final else c
            sd1 = sd_for(metric, known)
            for m in LOCATION_METHODS:
                p = _loc_point(g, c, m)
                if p is None:
                    continue
                for h in horizons:
                    sd = None if sd1 is None else sd1 * np.sqrt(h)
                    r = {"unit": unit, "metric": metric, "method": m, "origin_season": c, "h": h,
                         "target_season": c + h, "point": p, "sd": sd, "final": final,
                         "actual": float(tv[c + h]) if (c + h) in tv.index else None}
                    for lev in FORECAST_QUANTILES:
                        r[f"q{int(round(lev * 100)):02d}"] = None if sd is None else p + norm.ppf(lev) * sd
                    rows.append(r)
    df = pd.DataFrame(rows)
    if df.empty:
        return {"rows": df, "metrics": pd.DataFrame(), "selection": {}}
    bt = df[~df["final"] & df["actual"].notna() & df["q10"].notna()]
    mrows = []
    for (metric, m, h), g in bt.groupby(["metric", "method", "h"]):
        for lab, mask in (("dev", g["target_season"] <= dev_last_target),
                          ("retro_2026", g["target_season"] == 2026)):
            if mask.any():
                mrows.append({"metric": metric, "method": m, "h": int(h), "subset": lab, **score(g[mask])})
    metrics = pd.DataFrame(mrows)
    selection = {}
    if not metrics.empty:
        dev = metrics[metrics["subset"] == "dev"]
        for metric, g in dev.groupby("metric"):
            agg = g.groupby("method").apply(lambda x: np.average(x["mae"], weights=x["n"]), include_groups=False)
            selection[metric] = {"method": str(agg.idxmin()), "mae_by_method": {k: float(v) for k, v in agg.items()},
                                 "n": int(g[g["method"] == agg.idxmin()]["n"].sum())}
    return {"rows": df, "metrics": metrics, "selection": selection}
