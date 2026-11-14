"""Aging forecast engine API (System B, read-only): a pitcher's characteristic history, stored forecasts and the
rolling-origin backtest behind them. All numbers are stored outputs of `pitchiq aging-model` runs."""
from __future__ import annotations

from fastapi import APIRouter, Depends, Query
from sqlalchemy import select
from sqlalchemy.orm import Session

from ...aging_model.panel import ALL_METRICS, FAMILY_METRICS, LOCATION_METRICS, load_panel, location_panel
from ...aging_model.pipeline import RETRO_LABEL, latest_backtest
from ...db import get_db
from ...models_aging import PitcherBio
from ...models_aging_forecast import (AgingAnalysis, AgingBacktestMetric, AgingBacktestPrediction, AgingBacktestRun,
                                      AgingModelEntry, PitchCharacteristicForecast)

router = APIRouter(prefix="/pitchers", tags=["aging-forecast"])


def _row(obj, skip=frozenset()) -> dict:
    return {c.name: (str(v) if hasattr(v, "isoformat") else v)
            for c in obj.__table__.columns if c.name not in skip for v in [getattr(obj, c.name)]}


def _label(unit: str, metric: str) -> tuple[str, str | None]:
    if metric in LOCATION_METRICS:
        return f"{unit} {LOCATION_METRICS[metric][0]}", LOCATION_METRICS[metric][1]
    if unit == "ALL" and metric in ALL_METRICS:
        return ALL_METRICS[metric][0], ALL_METRICS[metric][1]
    if metric in FAMILY_METRICS:
        return f"{unit} {FAMILY_METRICS[metric][0]}", FAMILY_METRICS[metric][1]
    return f"{unit} {metric}", None


def _run_view(run) -> dict:
    return {"backtest_key": run.backtest_key, "historical_cutoff": run.historical_cutoff, "status": run.status,
            "started_at": str(run.started_at), "finished_at": str(run.finished_at), "config": run.config,
            "data_fingerprint": run.data_fingerprint}


def _bt(db: Session, pid: int, cutoff: int | None, backtest_key: str | None):
    if backtest_key:
        from ...models_aging_forecast import AgingBacktest
        run = db.get(AgingBacktest, backtest_key)
        if run is None or run.pitcher_id != pid:
            raise LookupError("unknown backtest_key for this pitcher")
        return run
    run = latest_backtest(db, pid, cutoff)
    if run is None:
        raise LookupError("no aging forecast run for this pitcher (run `pitchiq aging-model`)")
    return run


@router.get("/{pid}/aging/history")
def aging_history(pid: int, cutoff: int | None = None, db: Session = Depends(get_db)):
    """Season-by-season values of every forecast target for the pitcher (qualified and below-threshold seasons)."""
    bio = db.get(PitcherBio, pid)
    panel = load_panel(db, cutoff=cutoff, pitcher_id=pid)
    loc = location_panel(db, pid, cutoff=cutoff)
    if bio is None and panel.empty:
        raise LookupError("pitcher not in aging dataset")
    series = []
    for frame, kind in ((panel, "population_panel"), (loc, "pitch_level_location")):
        if frame.empty:
            continue
        for (unit, metric), g in frame.groupby(["unit", "metric"]):
            lab, units = _label(unit, metric)
            series.append({"unit": unit, "metric": metric, "key": f"{unit}|{metric}", "label": lab, "units": units,
                           "source": kind,
                           "seasons": [{"season": int(r.season), "value": float(r.value), "n": int(r.n),
                                        "age": None if r.age is None else float(r.age),
                                        "qualified": bool(r.qualified)}
                                       for r in g.sort_values("season").itertuples()]})
    return {"pitcher_id": pid,
            "bio": None if bio is None else {"full_name": bio.full_name, "birth_date": str(bio.birth_date),
                                             "throws": bio.throws},
            "cutoff": cutoff, "series": series,
            "notes": {"qualification": "families >= 100 pitches, pitcher-season metrics >= 500 pitches (scaled for "
                                       "the 60-game 2020 season); only qualified seasons enter the models",
                      "break_x_in": "magnitude of horizontal break (Savant pitcher_break_x, unsigned)",
                      "release_pos_x_armside_ft": "release side mirrored so arm side is positive for both hands"}}


@router.get("/{pid}/aging/forecast")
def aging_forecast(pid: int, cutoff: int | None = None, backtest_key: str | None = None,
                   all_methods: bool = False, include_draws: bool = False, unit: str | None = None,
                   metric: str | None = None, db: Session = Depends(get_db)):
    run = _bt(db, pid, cutoff, backtest_key)
    q = select(PitchCharacteristicForecast).where(PitchCharacteristicForecast.backtest_key == run.backtest_key,
                                                  PitchCharacteristicForecast.pitcher_id == pid)
    if not all_methods:
        q = q.where(PitchCharacteristicForecast.selected.is_(True))
    if unit:
        q = q.where(PitchCharacteristicForecast.unit == unit)
    if metric:
        q = q.where(PitchCharacteristicForecast.metric == metric)
    skip = frozenset() if include_draws else frozenset({"sample_draws"})
    rows = []
    for f in db.scalars(q.order_by(PitchCharacteristicForecast.unit, PitchCharacteristicForecast.metric,
                                   PitchCharacteristicForecast.target_season, PitchCharacteristicForecast.method)):
        d = _row(f, skip)
        d["label"], _ = _label(f.unit, f.metric)
        rows.append(d)
    sel = db.scalars(select(AgingModelEntry).where(AgingModelEntry.backtest_key == run.backtest_key,
                                                   AgingModelEntry.model_family == "aging_selected")).first()
    return {"run": _run_view(run), "selected_model": None if sel is None else sel.model_version,
            "selection": run.selection, "location_selection": (run.summary or {}).get("location_selection"),
            "forecasts": rows,
            "notes": {"conditional": "values are conditional on the unit qualifying in the target season",
                      "rest_of_season": "not produced: the cutoff season is complete",
                      "intervals": "lo80/hi80 = 10th/90th percentiles of the stored predictive distribution",
                      "monte_carlo": "sample_draws (include_draws=true) or distribution params per row"}}


@router.get("/{pid}/aging/backtest")
def aging_backtest(pid: int, cutoff: int | None = None, backtest_key: str | None = None, unit: str | None = None,
                   metric: str | None = None, breakdown_type: str = Query("subset"),
                   include_predictions: bool = False, include_analyses: bool = False,
                   db: Session = Depends(get_db)):
    run = _bt(db, pid, cutoff, backtest_key)
    q = select(AgingBacktestMetric).where(AgingBacktestMetric.backtest_key == run.backtest_key)
    if breakdown_type != "all":
        q = q.where(AgingBacktestMetric.breakdown_type == breakdown_type)
    if unit:
        q = q.where(AgingBacktestMetric.unit == unit)
    if metric:
        q = q.where(AgingBacktestMetric.metric == metric)
    metrics = [_row(m, {"backtest_key"}) for m in db.scalars(q.order_by(
        AgingBacktestMetric.unit, AgingBacktestMetric.metric, AgingBacktestMetric.horizon,
        AgingBacktestMetric.breakdown_key, AgingBacktestMetric.pinball_mean))]
    models = [_row(e, {"feature_schema", "dataset_fingerprint"}) for e in db.scalars(
        select(AgingModelEntry).where(AgingModelEntry.backtest_key == run.backtest_key))]
    evals = [{"evaluation_run_id": r.evaluation_run_id, "model_version": r.model_version, "kind": r.kind,
              "split": r.split, "n_samples": r.n_samples, "mlflow_run_id": (r.metrics or {}).get("mlflow_run_id"),
              "evaluation_label": (r.metrics or {}).get("evaluation_label")}
             for r in db.scalars(select(AgingBacktestRun).where(AgingBacktestRun.backtest_key == run.backtest_key))]
    out = {"run": _run_view(run), "selection": run.selection, "summary": run.summary, "failures": run.failures,
           "models": models, "evaluation_runs": evals, "metrics": metrics,
           "labels": {"development": "target seasons 2018-2025 (selection, tuning, interval calibration)",
                      RETRO_LABEL: "target season 2026: retrospective, previously viewed; never used for selection"}}
    if include_predictions:
        pq = select(AgingBacktestPrediction).where(AgingBacktestPrediction.backtest_key == run.backtest_key,
                                                   AgingBacktestPrediction.pitcher_id == pid)
        if unit:
            pq = pq.where(AgingBacktestPrediction.unit == unit)
        if metric:
            pq = pq.where(AgingBacktestPrediction.metric == metric)
        out["target_pitcher_predictions"] = [_row(p, {"backtest_key"}) for p in db.scalars(pq.order_by(
            AgingBacktestPrediction.unit, AgingBacktestPrediction.metric, AgingBacktestPrediction.origin_season,
            AgingBacktestPrediction.horizon, AgingBacktestPrediction.method))]
    if include_analyses:
        aq = select(AgingAnalysis).where(AgingAnalysis.backtest_key == run.backtest_key)
        if unit:
            aq = aq.where(AgingAnalysis.unit == unit)
        if metric:
            aq = aq.where(AgingAnalysis.metric == metric)
        out["analyses"] = [_row(a, {"backtest_key"}) for a in db.scalars(aq)]
    return out
