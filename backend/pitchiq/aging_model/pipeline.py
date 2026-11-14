"""Run the aging forecast engine end to end: as-of panel -> rolling-origin backtest per target -> selection ->
forecasts from the historical cutoff -> survivor / confounding analyses -> Postgres (+ MLflow mirror).

Leakage rules enforced here and in backtest.py:
* the panel is read with `season <= historical_cutoff`; backtest origin c fits on seasons <= c only;
* method selection, Marcel's k and the empirical interval residuals used for the final forecast come from
  development target seasons (<= 2025); target season 2026 is scored and stored as
  "retrospective_2026_previously_viewed" only;
* model parameters of the final forecast (curves, regressions, variance components, LightGBM) are fitted on all
  seasons <= the cutoff (history, not tuning).
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import uuid
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import joblib
import numpy as np
import pandas as pd
from scipy.stats import norm
from sqlalchemy import select, text
from sqlalchemy.orm import Session

from ..aging.families import age_completed_years, age_years, season_midpoint
from ..config import get_settings
from ..logging_setup import log
from ..models_aging import PitcherBio
from ..models_aging_forecast import (AgingAnalysis, AgingBacktest, AgingBacktestMetric, AgingBacktestPrediction,
                                     AgingBacktestRun, AgingModelEntry, PitchCharacteristicForecast)
from . import backtest as bt
from .analysis import analyze_target
from .methods import METHODS, NATIVE_DISTRIBUTION
from .panel import (ALL_METRICS, FAMILY_METRICS, LOCATION_METRICS, Target, load_panel, location_panel,
                    location_targets, target_frame, targets)

logger = logging.getLogger(__name__)
Progress = Callable[..., None]

FEATURE_VERSION = "aging_panel_v1"
N_DRAWS = 200
RETRO_LABEL = "retrospective_2026_previously_viewed"
DEV_LABEL = "development"
METHOD_DESCRIPTIONS = {
    "no_change": "y(t+h) = y(t)",
    "marcel": "Marcel-style 5/4/3 x sample-size weighted last 3 seasons regressed to the season-t population mean "
              "(k pseudo-pitches tuned on h=1 pseudo-forecasts ending <= the tuning season)",
    "age_curve": "population delta method on survivors (restricted cubic spline in age, same-era non-2020 pairs)",
    "age_curve_ipw": "population delta method with inverse-probability-of-continuation weights",
    "mixedlm": "statsmodels MixedLM on levels: age spline + Hawk-Eye + 2020 + log workload + starter share FE; "
               "random intercept + random age slope per pitcher",
    "lgbm_quantile": "LightGBM quantile regression (10/50/90) of y(t+h)-y(t) on the history at t",
    "eb_kalman": "empirical-Bayes local level: IPW population age curve + random walk, season + sampling noise "
                 "(variogram), Kalman filter, Gaussian predictive",
}


def _units(t: Target) -> str:
    if t.unit == "ALL":
        return ALL_METRICS[t.metric][1]
    if t.metric in LOCATION_METRICS:
        return LOCATION_METRICS[t.metric][1]
    return FAMILY_METRICS[t.metric][1]


def _qcols(df: pd.DataFrame) -> list[tuple[float, str]]:
    out = []
    for q in bt.FORECAST_QUANTILES:
        c = f"q{int(round(q * 100)):02d}"
        if c in df.columns:
            out.append((q, c))
    return out


def _seed(*parts) -> int:
    return int(hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:12], 16)


def sample_draws(dist: dict, n: int, seed: int) -> list[float]:
    """Monte Carlo draws from a stored distribution (normal, or piecewise-linear quantile function with Gaussian
    tails beyond the outermost stored quantiles)."""
    rng = np.random.default_rng(seed)
    if dist["type"] == "normal":
        return [float(x) for x in rng.normal(dist["mean"], dist["sd"], n)]
    lv = np.array(sorted(float(k) for k in dist["quantiles"]))
    qv = np.array([dist["quantiles"][k] for k in sorted(dist["quantiles"], key=float)], float)
    u = rng.uniform(0.0, 1.0, n)
    x = np.interp(u, lv, qv)
    sd = dist.get("tail_sd") or 0.0
    lo, hi = u < lv[0], u > lv[-1]
    x[lo] = qv[0] + (norm.ppf(u[lo]) - norm.ppf(lv[0])) * sd
    x[hi] = qv[-1] + (norm.ppf(u[hi]) - norm.ppf(lv[-1])) * sd
    return [float(v) for v in x]


def _distribution(row: pd.Series, qmap: dict) -> dict:
    native_gauss = row.get("method") in NATIVE_DISTRIBUTION and pd.notna(row.get("sd"))
    if native_gauss or str(row.get("interval_source", "")).startswith("gaussian"):
        return {"type": "normal", "mean": float(row["point"]), "sd": float(row["sd"])}
    q10, q90 = qmap.get("0.1"), qmap.get("0.9")
    tail = (q90 - q10) / (2 * bt.Z90) if q10 is not None and q90 is not None else None
    return {"type": "quantile_interp", "quantiles": qmap, "tail": "gaussian beyond outermost quantiles",
            "tail_sd": tail}


def _fingerprint(panel: pd.DataFrame) -> dict:
    cols = ["pitcher_id", "season", "unit", "metric", "value", "n"]
    d = panel[cols].sort_values(cols[:4])
    h = hashlib.sha256(pd.util.hash_pandas_object(d, index=False).values.tobytes()).hexdigest()
    return {"panel": {"n": int(len(panel)), "rows_hash": h, "seasons": sorted(int(s) for s in panel["season"].unique()),
                      "pitchers": int(panel["pitcher_id"].nunique()),
                      "qualified_rows": int(panel["qualified"].sum())}}


def _run_target(args):
    panel_t, target, cfg, pid, do_analysis, cutoff = args
    out = bt.backtest_target(panel_t, target, cfg, pid)
    if do_analysis:
        try:
            out["analysis"] = analyze_target(panel_t, cutoff)
        except Exception as e:  # noqa: BLE001
            out["analysis"] = {"error": f"{type(e).__name__}: {e}"[:300]}
    return out


def _map(fn, jobs, workers: int, progress: Progress | None, lo: float, hi: float):
    out = []
    if workers <= 1:
        for i, j in enumerate(jobs):
            out.append(fn(j))
            if progress:
                progress(lo + (hi - lo) * (i + 1) / len(jobs), f"backtest {out[-1]['target']}")
        return out
    import multiprocessing as mp
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("fork")) as ex:
        for i, r in enumerate(ex.map(fn, jobs)):
            out.append(r)
            if progress:
                progress(lo + (hi - lo) * (i + 1) / len(jobs), f"backtest {r['target']}")
    return out


def _clean(v):
    if v is None:
        return None
    if isinstance(v, (np.floating, float)):
        return None if not np.isfinite(v) else float(v)
    if isinstance(v, (np.integer,)):
        return int(v)
    return v


def _json(o):
    return json.loads(json.dumps(o, default=lambda x: x.item() if isinstance(x, np.generic) else str(x)),
                      parse_constant=lambda c: None)


def _artifact(path: Path, obj) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_aging_model(session: Session, pitcher_id: int | None = None, *, families: list[str] | None = None,
                    cutoff: int | None = None, origins: list[int] | None = None, horizons=(1, 2, 3),
                    methods: list[str] | None = None, workers: int | None = None, analyses: bool = True,
                    mirror: bool = True, n_draws: int = N_DRAWS, min_unit_pitches: int | None = None,
                    min_season_pitches: int | None = None, progress: Progress | None = None) -> dict:
    s = get_settings()
    pid = int(pitcher_id or s.default_pitcher_id)
    loaded = sorted(int(x) for x in session.execute(text("SELECT DISTINCT season FROM pitcher_season_stats")).scalars())
    if not loaded:
        raise LookupError("aging dataset is empty: run `pitchiq aging-etl` first")
    cutoff = int(cutoff or max(loaded))
    if origins is None:
        origins = [x for x in loaded if loaded[0] < x < cutoff]
    origins = sorted(int(o) for o in origins if int(o) < cutoff)
    methods = list(methods or METHODS)
    workers = int(workers or min(4, os.cpu_count() or 1))
    dev_last = min(bt.DEV_LAST_TARGET, cutoff - 1)
    cfg = bt.BacktestConfig(origins=tuple(origins), horizons=tuple(horizons), methods=tuple(methods),
                            dev_last_target=dev_last, final_cutoff=cutoff)
    key = f"ag{cutoff}-{datetime.now(timezone.utc):%Y%m%d%H%M%S}-{uuid.uuid4().hex[:6]}"
    tgts = targets(families)
    run = AgingBacktest(backtest_key=key, pitcher_id=pid, historical_cutoff=cutoff, status="running",
                        config={**cfg.to_dict(), "families": families, "min_unit_pitches": min_unit_pitches,
                                "min_season_pitches": min_season_pitches, "n_draws": n_draws, "workers": workers},
                        targets=[t.key for t in tgts])
    session.add(run)
    session.flush()
    if progress:
        progress(0.02, "loading as-of panel")
    panel = load_panel(session, cutoff=cutoff, min_unit_pitches=min_unit_pitches,
                       min_season_pitches=min_season_pitches)
    run.data_fingerprint = {**_fingerprint(panel), "etl_run_id": session.execute(text(
        "SELECT max(etl_run_id) FROM aging_etl_runs WHERE status = 'succeeded'")).scalar()}
    jobs = [(target_frame(panel, t), t, cfg, pid, analyses, cutoff) for t in tgts]
    results = _map(_run_target, jobs, workers, progress, 0.05, 0.85)
    loc = location_panel(session, pid, cutoff=cutoff, min_unit_pitches=min_unit_pitches)
    loc_res = bt.location_backtest(loc, origins=tuple(origins), horizons=tuple(horizons), dev_last_target=dev_last,
                                   final_cutoff=cutoff)
    if progress:
        progress(0.88, "storing")
    out = _persist(session, run, pid, cfg, tgts, results, loc, loc_res, n_draws)
    if mirror:
        out["mlflow"] = mirror_backtest(session, key)
    run.status = "succeeded"
    run.finished_at = datetime.now(timezone.utc)
    session.flush()
    if progress:
        progress(1.0, "done")
    return out


def _entry_metrics(results: list[dict], method: str) -> dict:
    """Headline development metrics of one method: mean skill vs no_change (1 - pinball / no_change pinball) on the
    selection subset per horizon, and the number of (target, horizon) pairs where it was selected."""
    vm: dict = {}
    for h in (1, 2, 3):
        skills, maes = [], []
        for r in results:
            m = r["metrics"]
            if m is None or len(m) == 0:
                continue
            sel = m[(m["breakdown_type"] == "subset") & (m["breakdown_key"] == "selection") & (m["h"] == h)]
            a, b = sel[sel["method"] == method], sel[sel["method"] == "no_change"]
            if len(a) and len(b) and b["pinball_mean"].iloc[0] > 0:
                skills.append(1 - a["pinball_mean"].iloc[0] / b["pinball_mean"].iloc[0])
                maes.append(a["coverage80"].iloc[0])
        if skills:
            vm[f"h{h}_mean_pinball_skill_vs_no_change"] = float(np.mean(skills))
            vm[f"h{h}_mean_coverage80"] = float(np.mean(maes))
            vm[f"h{h}_n_targets"] = len(skills)
        vm[f"h{h}_n_selected"] = sum(1 for r in results if (r["selection"].get(h) or {}).get("method") == method)
    return vm


def _summary_for(results: list[dict], method: str, subset_key: str) -> tuple[dict, int, dict]:
    summ, by_target, n = {}, {}, 0
    for r in results:
        m = r["metrics"]
        if m is None or len(m) == 0:
            continue
        g = m[(m["method"] == method) & (m["breakdown_type"] == "subset") & (m["breakdown_key"] == subset_key)]
        for _, row in g.iterrows():
            tk = r["target"].replace("|", ".")
            for k in ("mae", "rmse", "coverage80", "pinball_mean"):
                summ[f"{tk}.h{int(row['h'])}.{k}"] = float(row[k])
            by_target.setdefault(r["target"], {})[f"h{int(row['h'])}"] = {
                k: _clean(row[k]) for k in ("n", "mae", "rmse", "bias", "coverage80", "width80", "pinball_q10",
                                            "pinball_q50", "pinball_q90", "pinball_mean")}
            n += int(row["n"])
    return summ, n, by_target


def _persist(session: Session, run: AgingBacktest, pid: int, cfg: bt.BacktestConfig, tgts: list[Target],
             results: list[dict], loc: pd.DataFrame, loc_res: dict, n_draws: int) -> dict:
    s = get_settings()
    key, cutoff = run.backtest_key, run.historical_cutoff
    adir = Path(s.artifact_dir) / "aging" / key
    by_key = {t.key: t for t in tgts}
    selection = {r["target"]: {str(h): v for h, v in r["selection"].items()} for r in results}
    failures = [{"target": r["target"], **f} for r in results for f in r["failures"] + r.get("final_failures", [])]
    fp = run.data_fingerprint or {}
    common_cfg = {"backtest_key": key, "historical_cutoff": cutoff, "origins": list(cfg.origins),
                  "horizons": list(cfg.horizons), "selection_rule": bt.__doc__.split("Selection")[1].strip()[:900],
                  "dev_last_target_season": cfg.dev_last_target, "retrospective_seasons": list(bt.RETRO_SEASONS),
                  "split_policy": (f"rolling origin {cfg.origins[0]}-{cfg.origins[-1]}; scored targets >= "
                                   f"{cfg.first_scored_target}; development targets <= {cfg.dev_last_target}; "
                                   f"2026 retrospective (previously viewed)") if cfg.origins else "none"}
    entries: dict[str, AgingModelEntry] = {}
    schema = {"panel": "pitchiq.aging_model.panel", "targets": [t.key for t in tgts],
              "qualification": {"unit_pitches": s.aging_min_pitch_type_pitches,
                                "season_pitches": s.aging_min_season_pitches, "scaled_for_2020": True}}
    suffix = key.split("-", 1)[1]
    for m in cfg.methods:
        state = {r["target"]: (r.get("fitted_state") or {}).get(m) for r in results}
        art = {"method": m, "historical_cutoff": cutoff, "fitted_state_by_target": state}
        if m == "lgbm_quantile":
            art["lgbm_models_by_target"] = {r["target"]: r.get("lgbm_models") for r in results}
        path = adir / f"{m}.joblib"
        sha = _artifact(path, art)
        e = AgingModelEntry(model_version=f"aging-{m}-c{cutoff}-{suffix}"[:80], backtest_key=key, pitcher_id=pid,
                            model_family=f"aging_{m}", feature_version=FEATURE_VERSION, feature_schema=schema,
                            classes=[t.key for t in tgts],
                            config=_json({**common_cfg, "method": m, "description": METHOD_DESCRIPTIONS.get(m)}),
                            dataset_fingerprint=fp, artifact_path=str(path), artifact_sha256=sha,
                            calibration={"method": "native" if m in NATIVE_DISTRIBUTION else
                                         "empirical residual quantiles from earlier origins (fallback gaussian)",
                                         "fit_split": f"development targets <= {cfg.dev_last_target}"},
                            validation_metrics=_json(_entry_metrics(results, m)), status="candidate",
                            historical_cutoff=cutoff)
        session.add(e)
        entries[m] = e
    loc_path = adir / "location_own_history.joblib"
    loc_entry = AgingModelEntry(
        model_version=f"aging-location-c{cutoff}-{suffix}"[:80], backtest_key=key, pitcher_id=pid,
        model_family="aging_location_own_history", feature_version=FEATURE_VERSION,
        feature_schema={"panel": "pitchiq.aging_model.panel.location_panel (target pitcher pitch-level rows)",
                        "targets": [t.key for t in location_targets()]},
        classes=sorted({f"{u}|{m}" for u, m in zip(loc["unit"], loc["metric"])}) if len(loc) else [],
        config=_json({**common_cfg, "methods": list(bt.LOCATION_METHODS),
                      "description": "own-history forecasts of plate-location dispersion (no population pitch-level "
                                     "data); selection per metric by development MAE"}),
        dataset_fingerprint={"location_rows": int(len(loc))},
        artifact_path=str(loc_path),
        artifact_sha256=_artifact(loc_path, {"selection": loc_res["selection"],
                                             "history": loc.to_dict("records")}),
        calibration={"method": "gaussian, sd = sqrt(h) x SD of own one-season changes (targets <= dev season)"},
        validation_metrics=_json({f"{k}.selected": v["method"] for k, v in loc_res["selection"].items()}),
        status="candidate", historical_cutoff=cutoff)
    session.add(loc_entry)
    # the composite model that the forecasts' `selected` rows come from
    prev = session.scalars(select(AgingModelEntry).where(AgingModelEntry.pitcher_id == pid,
                                                         AgingModelEntry.model_family == "aging_selected",
                                                         AgingModelEntry.status == "promoted")).all()
    for p in prev:
        p.status = "archived"
    sel_path = adir / "selected.joblib"
    sel_entry = AgingModelEntry(
        model_version=f"aging-selected-c{cutoff}-{suffix}"[:80], backtest_key=key, pitcher_id=pid,
        model_family="aging_selected", feature_version=FEATURE_VERSION, feature_schema=schema,
        classes=[t.key for t in tgts],
        config=_json({**common_cfg, "selection": selection, "location_selection": loc_res["selection"],
                      "member_models": {m: e.model_version for m, e in entries.items()},
                      "promotion_source": "aging backtest selection (development targets only)"}),
        dataset_fingerprint=fp, artifact_path=str(sel_path),
        artifact_sha256=_artifact(sel_path, {"selection": selection, "location_selection": loc_res["selection"],
                                             "members": {m: e.model_version for m, e in entries.items()}}),
        calibration={"method": "per selected member"},
        validation_metrics=_json(_selected_metrics(results)), status="promoted", historical_cutoff=cutoff)
    session.add(sel_entry)
    session.flush()

    # evaluation rows (development + retrospective) per method
    evals = []
    for m, e in entries.items():
        for split, subset in ((DEV_LABEL, "selection"), (DEV_LABEL + "_same_era_all_ages", "dev_same_era"),
                              (RETRO_LABEL, "retro_2026")):
            summ, n, by_target = _summary_for(results, m, subset)
            if not n:
                continue
            r = AgingBacktestRun(backtest_key=key, pitcher_id=pid, model_version=e.model_version,
                                 kind="aging_backtest", split=split,
                                 dataset_definition=_json({"subset": subset, **common_cfg,
                                                           "evaluation_label": RETRO_LABEL if "retro" in subset
                                                           else DEV_LABEL}),
                                 n_samples=n, metrics=_json({"summary": summ, "by_target": by_target,
                                                             "evaluation_label": RETRO_LABEL if "retro" in subset
                                                             else DEV_LABEL}))
            session.add(r)
            evals.append(r)
    lm = loc_res["metrics"]
    if len(lm):
        for split, sub in ((DEV_LABEL, "dev"), (RETRO_LABEL, "retro_2026")):
            g = lm[lm["subset"] == sub]
            if g.empty:
                continue
            summ = {f"{r.metric}.{r.method}.h{int(r.h)}.mae": float(r.mae) for r in g.itertuples()}
            summ.update({f"{r.metric}.{r.method}.h{int(r.h)}.coverage80": float(r.coverage80) for r in g.itertuples()})
            ev = AgingBacktestRun(backtest_key=key, pitcher_id=pid, model_version=loc_entry.model_version,
                                  kind="aging_location_backtest", split=split,
                                  dataset_definition=_json({"subset": sub, **common_cfg}), n_samples=int(g["n"].sum()),
                                  metrics=_json({"summary": summ, "table": g.to_dict("records")}))
            session.add(ev)
            evals.append(ev)
    session.flush()

    # metric rows
    rows = []
    for r in results:
        t = by_key[r["target"]]
        m = r["metrics"]
        if m is None or len(m) == 0:
            continue
        for x in m.to_dict("records"):
            rows.append(dict(backtest_key=key, method=x["method"], unit=t.unit, metric=t.metric, horizon=int(x["h"]),
                             breakdown_type=x["breakdown_type"], breakdown_key=x["breakdown_key"], n=int(x["n"]),
                             **{k: float(x[k]) for k in ("mae", "rmse", "bias", "coverage80", "width80",
                                                         "pinball_q10", "pinball_q50", "pinball_q90",
                                                         "pinball_mean")}))
    for x in (lm.to_dict("records") if len(lm) else []):
        rows.append(dict(backtest_key=key, method=f"loc_{x['method']}", unit="ALL_FAMILIES", metric=x["metric"],
                         horizon=int(x["h"]), breakdown_type="location_subset", breakdown_key=x["subset"],
                         n=int(x["n"]), **{k: float(x[k]) for k in ("mae", "rmse", "bias", "coverage80", "width80",
                                                                    "pinball_q10", "pinball_q50", "pinball_q90",
                                                                    "pinball_mean")}))
    if rows:
        session.execute(AgingBacktestMetric.__table__.insert(), rows)

    # target pitcher's backtest forecasts
    prow = []
    for r in results:
        t = by_key[r["target"]]
        d = r["target_pitcher_rows"]
        for x in (d.to_dict("records") if len(d) else []):
            ts = int(x["target_season"])
            prow.append(dict(backtest_key=key, pitcher_id=pid, method=x["method"], unit=t.unit, metric=t.metric,
                             origin_season=int(x["origin_season"]), target_season=ts, horizon=int(x["h"]),
                             origin_value=_clean(x["origin_value"]), origin_age=_clean(x["origin_age"]),
                             point=float(x["point"]), q10=_clean(x.get("q10")), q50=_clean(x.get("q50")),
                             q90=_clean(x.get("q90")), actual=_clean(x.get("actual")),
                             interval_source=x.get("interval_source"),
                             evaluation_label=RETRO_LABEL if ts in bt.RETRO_SEASONS else
                             (DEV_LABEL if ts <= cfg.dev_last_target else "future")))
    lr = loc_res["rows"]
    for x in (lr[~lr["final"]].to_dict("records") if len(lr) else []):
        ts = int(x["target_season"])
        prow.append(dict(backtest_key=key, pitcher_id=pid, method=f"loc_{x['method']}", unit=x["unit"],
                         metric=x["metric"], origin_season=int(x["origin_season"]), target_season=ts,
                         horizon=int(x["h"]), origin_value=None, origin_age=None, point=float(x["point"]),
                         q10=_clean(x.get("q10")), q50=_clean(x["point"]), q90=_clean(x.get("q90")),
                         actual=_clean(x.get("actual")), interval_source="gaussian_own_change_sd",
                         evaluation_label=RETRO_LABEL if ts in bt.RETRO_SEASONS else
                         (DEV_LABEL if ts <= cfg.dev_last_target else "future")))
    if prow:
        session.execute(AgingBacktestPrediction.__table__.insert(), prow)

    # forecasts
    bio = session.get(PitcherBio, pid)
    birth = bio.birth_date if bio else None
    frows = []
    sale_headline = {}
    for r in results:
        t = by_key[r["target"]]
        fc = r.get("final")
        if fc is None or len(fc) == 0:
            continue
        for x in fc.to_dict("records"):
            h = int(x["h"])
            sel = (r["selection"].get(h) or {})
            chosen = sel.get("method")
            row = pd.Series(x)
            qmap = {str(q): _clean(x.get(c)) for q, c in _qcols(fc) if _clean(x.get(c)) is not None}
            dist = _distribution(row, qmap)
            seed = _seed(pid, cutoff, t.key, x["method"], h)
            ts = int(x["target_season"])
            frows.append(dict(
                backtest_key=key, pitcher_id=pid, historical_cutoff=cutoff, target_season=ts, horizon=h,
                projected_age=age_years(birth, season_midpoint(ts)), projected_age_completed=age_completed_years(
                    birth, season_midpoint(ts)), unit=t.unit, metric=t.metric, value_units=_units(t),
                model_version=entries[x["method"]].model_version, method=x["method"],
                selected=x["method"] == chosen, selection_model_version=sel_entry.model_version,
                point=float(x["point"]), mean=float(x["point"]) if dist["type"] == "normal" else None,
                sd=_clean(x.get("sd")), lo80=_clean(x.get("q10")), hi80=_clean(x.get("q90")), quantiles=qmap,
                distribution=_json(dist), sample_draws=sample_draws(dist, n_draws, seed) if n_draws else None,
                draws_seed=seed, last_observed_season=int(x["origin_season"]),
                last_observed_value=_clean(x["origin_value"]),
                assumptions=_json(_assumptions(t, x, cfg, sel, r, birth))))
            if x["method"] == chosen:
                sale_headline.setdefault(t.key, {})[str(ts)] = {
                    "method": chosen, "point": round(float(x["point"]), 3),
                    "lo80": _clean(x.get("q10")), "hi80": _clean(x.get("q90"))}
    for x in (lr[lr["final"]].to_dict("records") if len(lr) else []):
        if x.get("q10") is None:
            continue
        chosen = (loc_res["selection"].get(x["metric"]) or {}).get("method")
        ts = int(x["target_season"])
        qmap = {str(q): float(x[f"q{int(round(q * 100)):02d}"]) for q in bt.FORECAST_QUANTILES}
        dist = {"type": "normal", "mean": float(x["point"]), "sd": float(x["sd"])}
        seed = _seed(pid, cutoff, x["unit"], x["metric"], x["method"], x["h"])
        frows.append(dict(
            backtest_key=key, pitcher_id=pid, historical_cutoff=cutoff, target_season=ts, horizon=int(x["h"]),
            projected_age=age_years(birth, season_midpoint(ts)),
            projected_age_completed=age_completed_years(birth, season_midpoint(ts)), unit=x["unit"],
            metric=x["metric"], value_units=LOCATION_METRICS[x["metric"]][1], model_version=loc_entry.model_version,
            method=f"loc_{x['method']}", selected=x["method"] == chosen, selection_model_version=sel_entry.model_version,
            point=float(x["point"]), mean=float(x["point"]), sd=float(x["sd"]), lo80=float(x["q10"]),
            hi80=float(x["q90"]), quantiles=qmap, distribution=dist,
            sample_draws=sample_draws(dist, n_draws, seed) if n_draws else None, draws_seed=seed,
            last_observed_season=cutoff, last_observed_value=None,
            assumptions=_json({"historical_cutoff": cutoff, "own_history_only": True,
                               "why": "location dispersion needs pitch-level data, available only for the target "
                                      "pitcher; no population model or age curve",
                               "interval": "gaussian, sd = sqrt(h) x SD of own one-season changes (dev targets)",
                               "selection": loc_res["selection"].get(x["metric"]),
                               "rest_of_season": "not produced: the cutoff season is complete",
                               "conditional_on": "the family is thrown >= 100 pitches in the target season"})))
    if frows:
        session.execute(PitchCharacteristicForecast.__table__.insert(), frows)

    # analyses
    arows = []
    for r in results:
        if r.get("analysis") is not None:
            t = by_key[r["target"]]
            arows.append(dict(backtest_key=key, kind="survivor_confounding", data_cutoff=cutoff, unit=t.unit,
                              metric=t.metric, results=_json(r["analysis"])))
    if arows:
        session.execute(AgingAnalysis.__table__.insert(), arows)

    run.selection = _json(selection)
    run.failures = _json(failures)
    run.summary = _json({"n_targets": len(results), "n_scored_forecasts": sum(r["n_scored"] for r in results),
                         "seconds_by_target": {r["target"]: r["seconds"] for r in results},
                         "selected_counts": _selected_counts(results), "target_pitcher_forecast": sale_headline,
                         "location_selection": loc_res["selection"], "n_forecast_rows": len(frows),
                         "models": {m: e.model_version for m, e in entries.items()},
                         "location_model": loc_entry.model_version, "selected_model": sel_entry.model_version,
                         "evaluation_run_ids": [e.evaluation_run_id for e in evals]})
    session.flush()
    log(logger, "aging_model_stored", backtest_key=key, forecasts=len(frows), metrics=len(rows))
    return {"backtest_key": key, "historical_cutoff": cutoff, "selected_model": sel_entry.model_version,
            "models": {m: e.model_version for m, e in entries.items()}, "n_forecast_rows": len(frows),
            "n_metric_rows": len(rows), "n_target_pitcher_backtest_rows": len(prow),
            "evaluation_run_ids": [e.evaluation_run_id for e in evals], "failures": len(failures),
            "selected_counts": _selected_counts(results)}


def _selected_counts(results) -> dict:
    out: dict = {}
    for r in results:
        for h, v in r["selection"].items():
            out.setdefault(f"h{h}", {}).setdefault(v["method"], 0)
            out[f"h{h}"][v["method"]] += 1
    return out


def _selected_metrics(results) -> dict:
    vm = {}
    for h in (1, 2, 3):
        sk = []
        for r in results:
            v = r["selection"].get(h)
            if v and v.get("no_change_pinball_mean"):
                sk.append(1 - v["pinball_mean"] / v["no_change_pinball_mean"])
        if sk:
            vm[f"h{h}_mean_pinball_skill_vs_no_change"] = float(np.mean(sk))
            vm[f"h{h}_n_targets"] = len(sk)
    return vm


def _assumptions(t: Target, x: dict, cfg: bt.BacktestConfig, sel: dict, r: dict, birth) -> dict:
    state = (r.get("fitted_state") or {})
    return {
        "historical_cutoff": cfg.final_cutoff, "data_used": f"seasons <= {cfg.final_cutoff}",
        "tuning_and_interval_calibration_through_target_season": cfg.dev_last_target,
        "selection": {"chosen_method": sel.get("method"), "basis": sel.get("basis"), "n": sel.get("n"),
                      "dev_pinball_mean": sel.get("pinball_mean"), "dev_coverage80": sel.get("coverage80"),
                      "relative_pinball_vs_no_change": sel.get("relative_pinball_mean_over_horizons"),
                      "rule": "one method per target: lowest mean over horizons of pinball loss (q10/q50/q90) "
                              "relative to no_change; development targets 2018-2025, no 2020, same measurement "
                              "era, origin age >= 32"},
        "interval_source": x.get("interval_source"),
        "conditional_on": ("the pitcher qualifies in the target season (>= 500 pitches)" if t.unit == "ALL" else
                           "the pitch family is thrown >= 100 pitches in the target season"),
        "workload_and_role": "season-t workload and starter share carried forward (mixedlm, lgbm features)",
        "measurement_era": "Hawk-Eye; no new measurement change assumed after the cutoff",
        "season_length": "162 games",
        "no_forced_decline": True, "no_injury_inference": True,
        "age_from": "MLB Stats API birth date; age on June 30 of the target season" if birth else "birth date missing",
        "rest_of_season": "not produced: the cutoff season is complete (forecasts start with the next season)",
        "population_support": "few pitcher-seasons at ages >= 38 (see docs/aging-model.md); forecasts at 38-40 "
                              "lean on the age-curve tail and on the pitcher's own history",
        "method_state": state.get(x["method"]) if x["method"] in ("marcel", "eb_kalman") else None,
    }


# --------------------------------------------------------------------------- MLflow mirror (reuses ml.tracking)
def aging_tracker():
    from ..ml.tracking import Tracker
    s = get_settings()
    return Tracker(prefix=f"{s.mlflow_experiment_prefix}-aging")


def mirror_backtest(session: Session, backtest_key: str, tracker=None) -> dict:
    """Mirror the run's model entries and evaluation rows into MLflow experiment `<prefix>-aging-<pitcher_id>`
    via the existing Tracker (Postgres stays the source of truth). Writes run ids back into config / metrics."""
    tr = tracker or aging_tracker()
    if not tr.enabled:
        return {"enabled": False, "reason": tr.disabled_reason}
    entries = session.scalars(select(AgingModelEntry).where(AgingModelEntry.backtest_key == backtest_key)).all()
    runs = session.scalars(select(AgingBacktestRun).where(AgingBacktestRun.backtest_key == backtest_key)).all()
    by_v = {e.model_version: e for e in entries}
    done = {"models": 0, "evaluations": 0, "experiment": None}
    pol = (entries[0].config or {}).get("split_policy") if entries else None
    for e in entries:
        rid = tr.log_model_entry(e)
        if rid:
            e.config = {**(e.config or {}), "mlflow_run_id": rid}
            _retag(tr, rid, pol)
            done["models"] += 1
    session.flush()
    for r in runs:
        rid = tr.log_evaluation(r, by_v.get(r.model_version))
        if rid:
            r.metrics = {**(r.metrics or {}), "mlflow_run_id": rid}
            _retag(tr, rid, pol)
            done["evaluations"] += 1
    session.flush()
    if entries:
        done["experiment"] = tr.experiment_name(entries[0].pitcher_id)
    return done


def _retag(tr, rid: str, policy: str | None) -> None:
    """The shared mirror tags next-pitch split settings; replace them with the aging backtest policy."""
    try:
        tr.client.set_tag(rid, "pitchiq.split_policy", policy or "aging rolling origin")
        tr.client.set_tag(rid, "pitchiq.system", "aging_forecast")
    except Exception as e:  # noqa: BLE001 - mirror is best-effort
        log(logger, "mlflow_mirror_failed", op="aging_retag", error=f"{type(e).__name__}: {e}"[:300])


def latest_backtest(session: Session, pitcher_id: int, cutoff: int | None = None) -> AgingBacktest | None:
    q = select(AgingBacktest).where(AgingBacktest.pitcher_id == pitcher_id, AgingBacktest.status == "succeeded")
    if cutoff is not None:
        q = q.where(AgingBacktest.historical_cutoff == int(cutoff))
    return session.scalars(q.order_by(AgingBacktest.finished_at.desc())).first()


__all__ = ["run_aging_model", "mirror_backtest", "latest_backtest", "sample_draws"]
