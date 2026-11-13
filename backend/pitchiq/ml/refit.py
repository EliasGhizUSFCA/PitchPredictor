"""Deployment refit and scheduled retraining for the next-pitch models.

1. **Selection** (`protocol.rolling_dev_evaluate`): each family's grid is scored on rolling-origin development folds
   (2023-2025 only); the lowest pooled raw log loss picks the hyperparameters (incl. LightGBM boosting rounds).
2. **Final fit** (`final_fit`): the selected recipe is refit on ALL labeled pitches from games dated before the
   deployment cutoff (today: 2023-2026). Calibration is a temperature fit on the recipe's pooled chronological
   out-of-sample development predictions - never on the data the final model was fit on.
3. **Gated promotion** (`lifecycle.promote_with_gates`): the family with the lowest chronologically calibrated dev
   log loss is the production candidate; it is promoted only if every predetermined gate passes.
4. **Scheduled retraining** (`retrain`): policies after_each_start | every_6_starts | weekly | manual decide when a
   refit with the frozen recipe is due; the new model goes through the same gates.

Data cutoff, training sample counts, selected hyperparameters and dev metrics are stored in `model_lifecycle`,
`model_registry.config`, and mirrored to MLflow through `tracking.mirror_model_entry` / `mirror_evaluation`.
"""
from __future__ import annotations

import logging
from datetime import date, datetime, timezone
from typing import Callable

import numpy as np
import pandas as pd
from sqlalchemy.orm import Session

from ..config import get_settings
from ..features.schema import FEATURE_VERSION, schema_dict
from ..features.store import get_classes
from ..logging_setup import log
from ..models import EvaluationRun, ModelRegistry
from ..models_lifecycle import ModelLifecycle
from . import lifecycle, registry, tracking
from .evaluate import full_report
from .models import TemperatureCalibrator
from .protocol import (DevResult, game_table, load_dated_frame, make_model, per_fold_summaries, role_label,
                       rolling_dev_evaluate)
from .train import fingerprint

logger = logging.getLogger(__name__)

POLICIES = ("after_each_start", "every_6_starts", "weekly", "manual")
FAMILY_ORDER = ("majority", "frequency", "logreg", "lightgbm")


def today_utc() -> date:
    return datetime.now(timezone.utc).date()


# ------------------------------------------------------------------ selection records
def selection_from_dev(res: DevResult) -> dict:
    return {"family": res.family, "params": res.params, "temperature": res.temperature,
            "dev_metrics": res.dev_metrics(), "grid_results": res.grid_results,
            "raw_oos_log_loss": res.summary_raw["log_loss"],
            "calibrated_oos_log_loss_in_sample_T": float(
                -np.mean(np.log(np.clip(TemperatureCalibrator._apply(np.log(np.clip(res.P_raw, 1e-12, 1)),
                                                                      res.temperature)[np.arange(len(res.y)), res.y],
                                        1e-15, 1)))),
            "oos_fingerprint": fingerprint(res.eval_frame), "dev_evaluation_run_id": None}


def selection_from_entry(session: Session, entry: ModelRegistry) -> dict:
    """Frozen recipe of a lifecycle model (used by scheduled retrains without re-selection)."""
    lc = lifecycle.lifecycle_row(session, entry.model_version)
    if lc is None:
        raise LookupError(f"{entry.model_version} has no lifecycle record; re-select instead")
    cfg = entry.config or {}
    return {"family": entry.model_family, "params": lc.selected_params, "temperature": entry.calibration["temperature"],
            "dev_metrics": lc.dev_metrics, "grid_results": cfg.get("grid_results") or [],
            "raw_oos_log_loss": entry.calibration.get("val_log_loss_before"),
            "calibrated_oos_log_loss_in_sample_T": entry.calibration.get("val_log_loss_after"),
            "oos_fingerprint": (entry.dataset_fingerprint or {}).get("development_oos"),
            "dev_evaluation_run_id": lc.dev_evaluation_run_id, "inherited_from": entry.model_version}


# ------------------------------------------------------------------ final fit
def _unique_version(session: Session, pitcher_id: int, family: str, tag: str) -> str:
    v = registry.new_version(pitcher_id, family, FEATURE_VERSION, tag)
    base, i = v, 2
    while session.get(ModelRegistry, v) is not None:
        v, i = f"{base}-{i}", i + 1
    return v


def final_fit(session: Session, pitcher_id: int, selection: dict, deployment_cutoff: date | None = None,
              scope: str = "deployment_refit", policy: str | None = None, trigger: str | None = None,
              tag: str = "refit", df: pd.DataFrame | None = None, classes: list[str] | None = None) -> ModelRegistry:
    started_at = datetime.now(timezone.utc)
    family = selection["family"]
    classes = classes or get_classes(session, pitcher_id)
    df = df if df is not None else load_dated_frame(session, pitcher_id)
    deployment_cutoff = deployment_cutoff or today_utc()
    train = df[df.game_date < deployment_cutoff]
    if train.empty:
        raise ValueError("no labeled pitches before the deployment cutoff")
    model = make_model(family, classes, selection["params"]).fit(train, train["label"])
    model.calibrator = TemperatureCalibrator()
    model.calibrator.T = float(selection["temperature"])
    cutoff = max(train.game_date)
    version = _unique_version(session, pitcher_id, family, tag)
    path, sha = registry.save_artifact(model, version)
    dev = selection["dev_metrics"]
    seasons = sorted({int(x) for x in train.game_season.unique()})
    calib = {"method": "temperature_scaling", "fit_split": "development_rolling_oos",
             "temperature": float(selection["temperature"]),
             "val_log_loss_before": selection.get("raw_oos_log_loss"),
             "val_log_loss_after": selection.get("calibrated_oos_log_loss_in_sample_T"),
             "note": "temperature fit on pooled rolling-origin development out-of-sample predictions of this recipe"}
    cfg = {**model.config(), "grid_results": selection.get("grid_results") or [],
           "selection": "min rolling-origin development log loss", "training_scope": scope,
           "data_cutoff": str(cutoff), "deployment_cutoff": str(deployment_cutoff), "n_train": int(len(train)),
           "n_train_games": int(train.game_pk.nunique()), "train_seasons_present": seasons,
           "selected_params": selection["params"], "dev_protocol": {k: dev.get(k) for k in
                                                                    ("protocol", "eval_seasons", "block_games",
                                                                     "n_folds", "calibration")},
           "retrain_policy": policy, "trigger": trigger, "inherited_recipe_from": selection.get("inherited_from")}
    fp = {"train": fingerprint(train)}
    if selection.get("oos_fingerprint"):
        fp["development_oos"] = selection["oos_fingerprint"]
    entry = ModelRegistry(model_version=version, pitcher_id=pitcher_id, model_family=family,
                          feature_version=FEATURE_VERSION,
                          feature_schema={**schema_dict(classes), "used_features": model.features}, classes=classes,
                          config=cfg, dataset_fingerprint=fp, artifact_path=path, artifact_sha256=sha,
                          calibration=calib, validation_metrics=dict(dev), status="candidate")
    session.add(entry)
    session.flush()
    session.add(ModelLifecycle(model_version=version, pitcher_id=pitcher_id, training_scope=scope,
                               data_cutoff=cutoff, deployment_cutoff=deployment_cutoff,
                               n_train_samples=int(len(train)), n_train_games=int(train.game_pk.nunique()),
                               selected_params=selection["params"], dev_metrics=dev,
                               dev_evaluation_run_id=selection.get("dev_evaluation_run_id"),
                               retrain_policy=policy, trigger=trigger))
    session.flush()
    tracking.mirror_model_entry(entry, started_at=started_at)
    _tag_run(entry)
    session.flush()
    log(logger, "final_fit", version=version, family=family, n_train=len(train), data_cutoff=str(cutoff),
        dev_log_loss=dev.get("log_loss"))
    return entry


def _tag_run(entry: ModelRegistry) -> None:
    """Extra MLflow tags/metrics for lifecycle models (best effort; Postgres is the record)."""
    t = tracking.get_tracker()
    rid = (entry.config or {}).get("mlflow_run_id")
    if not rid or not t.enabled:
        return
    try:
        cfg = entry.config
        for k in ("training_scope", "data_cutoff", "deployment_cutoff", "retrain_policy"):
            if cfg.get(k) is not None:
                t.client.set_tag(rid, f"pitchiq.{k}", str(cfg[k]))
        t.client.set_tag(rid, "pitchiq.evaluation_protocol", "rolling_origin_development")
        t.client.log_metric(rid, "n_train", float(cfg["n_train"]))
        t.client.log_metric(rid, "n_train_games", float(cfg["n_train_games"]))
        for k, v in (entry.validation_metrics or {}).items():
            if isinstance(v, (int, float)) and not isinstance(v, bool) and np.isfinite(v):
                t.client.log_metric(rid, f"dev_{k}", float(v))
    except Exception as e:  # noqa: BLE001
        log(logger, "mlflow_mirror_failed", op="lifecycle_tags", error=f"{type(e).__name__}: {e}"[:500])


# ------------------------------------------------------------------ stored dev evaluations
def store_dev_run(session: Session, entry: ModelRegistry, res: DevResult, recipe: str) -> EvaluationRun:
    classes = entry.classes
    report = full_report(res.P_cal, res.y, res.eval_frame, classes)
    report.update(uncalibrated_summary=res.summary_raw, dev_metrics=res.dev_metrics(),
                  evaluation_role="development", evaluation_label=role_label("development"),
                  grid_results=res.grid_results, selected_params=res.params, folds=per_fold_summaries(res, classes),
                  fold_temperatures=res.fold_temperatures, pooled_temperature=res.temperature,
                  recipe=recipe, seconds=res.seconds,
                  note=("Pooled out-of-sample predictions of rolling-origin development folds. 'summary' is "
                        "chronologically calibrated (fold f uses a temperature fit on folds before f); "
                        "'uncalibrated_summary' is the raw selection objective. The selected configuration's score "
                        "is the minimum over the grid, so it is optimistic by the selection effect."))
    run = EvaluationRun(pitcher_id=entry.pitcher_id, model_version=entry.model_version, kind="rolling_dev_evaluation",
                        split="development", n_samples=int(len(res.y)), metrics=report,
                        dataset_definition={"protocol": "rolling_origin_expanding_window",
                                            "eval_seasons": res.extra.get("eval_seasons"),
                                            "block_games": res.extra.get("block_games"),
                                            "n_folds": sum(f.skipped is None for f in res.folds),
                                            "feature_version": entry.feature_version,
                                            "evaluation_role": "development",
                                            "folds": [f.meta() for f in res.folds]})
    session.add(run)
    session.flush()
    tracking.mirror_evaluation(run, entry)
    session.flush()
    lc = lifecycle.lifecycle_row(session, entry.model_version)
    if lc is not None:
        lc.dev_evaluation_run_id = run.evaluation_run_id
        session.flush()
    return run


def evaluate_recipe_on_dev(session: Session, entry: ModelRegistry, progress: Callable | None = None) -> EvaluationRun:
    """Score an existing (e.g. legacy season-split) model's frozen recipe on the same rolling development folds,
    so it can be compared with lifecycle candidates by the promotion gates."""
    base = registry.load_model(entry)
    params = dict(base.params)
    if entry.model_family == "lightgbm":
        params["n_estimators"] = int(getattr(base, "best_iteration_", None) or params.get("n_estimators"))
    df = load_dated_frame(session, entry.pitcher_id, entry.feature_version)
    res = rolling_dev_evaluate(df, entry.classes, entry.model_family, grid=[params], features=base.features,
                               fixed_rounds=True, progress=progress)
    return store_dev_run(session, entry, res, recipe=f"frozen recipe of {entry.model_version}")


# ------------------------------------------------------------------ full refit
def refit_all(session: Session, pitcher_id: int, families=FAMILY_ORDER, deployment_cutoff: date | None = None,
              grids: dict | None = None, block_games: int | None = None, promote: bool = True,
              progress: Callable | None = None, tag: str = "refit") -> dict:
    classes = get_classes(session, pitcher_id)
    df = load_dated_frame(session, pitcher_id)
    out = {"families": {}, "deployment_cutoff": str(deployment_cutoff or today_utc())}
    entries = {}
    for i, fam in enumerate(families):
        sub = (lambda f, m="", r=None, i=i: progress((i + f) / len(families), m, r)) if progress else None
        res = rolling_dev_evaluate(df, classes, fam, grid=(grids or {}).get(fam), block_games=block_games,
                                   progress=sub)
        sel = selection_from_dev(res)
        e = final_fit(session, pitcher_id, sel, deployment_cutoff, tag=tag, df=df, classes=classes)
        run = store_dev_run(session, e, res, recipe="selected on rolling development folds")
        entries[fam] = e
        out["families"][fam] = {"model_version": e.model_version, "selected_params": res.params,
                                "dev_evaluation_run_id": run.evaluation_run_id, "dev": res.dev_metrics(),
                                "n_train": e.config["n_train"], "data_cutoff": e.config["data_cutoff"]}
    # production candidate: lowest chronologically calibrated dev log loss (the deployed output is calibrated)
    ranked = sorted(entries.values(), key=lambda e: e.validation_metrics["log_loss"])
    out["ranking"] = [{"family": e.model_family, "model_version": e.model_version,
                       "dev_log_loss": e.validation_metrics["log_loss"]} for e in ranked]
    out["promotion"] = None
    if promote:
        for e in ranked:
            res = lifecycle.promote_with_gates(session, e.model_version,
                                               source="refit_all: lowest rolling development log loss passing gates")
            out.setdefault("gate_attempts", []).append({"model_version": e.model_version,
                                                        "passed": res["gates"]["passed"] if "gates" in res else True,
                                                        "failed": [g["gate"] for g in res.get("gates", {})
                                                                   .get("gates", []) if not g["passed"]]})
            if res["promoted"]:
                out["promotion"] = res
                break
    return out


# ------------------------------------------------------------------ scheduled retraining
def policy_due(policy: str, new_games: pd.DataFrame, last_fit_at: datetime, today: date, force: bool) -> tuple:
    n = len(new_games)
    if policy not in POLICIES:
        raise ValueError(f"Unknown retraining policy {policy}. Known: {POLICIES}")
    if force:
        return True, "forced"
    if policy == "manual":
        return True, "manual trigger (operator ran the job)"
    if policy == "after_each_start":
        return n >= 1, f"{n} completed start(s) since the production data cutoff"
    if policy == "every_6_starts":
        return n >= 6, f"{n} of 6 completed starts since the production data cutoff"
    days = (today - last_fit_at.date()).days
    return (days >= 7 and n >= 1), f"{days} day(s) since the last fit, {n} new start(s)"


def retrain(session: Session, pitcher_id: int, policy: str, force: bool = False, reselect: bool = False,
            today: date | None = None, promote: bool = True, progress: Callable | None = None) -> dict:
    """Retrain the production recipe when the policy says a refit is due; gate and promote the result."""
    today = today or today_utc()
    prod = registry.promoted(session, pitcher_id)
    if prod is None:
        raise LookupError("No production model; run `pitchiq refit` first")
    df = load_dated_frame(session, pitcher_id)
    games = game_table(df)
    cut = lifecycle.data_cutoff(session, prod)
    new = games[(games.game_date > cut) & (games.game_date < today)] if cut else games.iloc[0:0]
    lc = lifecycle.lifecycle_row(session, prod.model_version)
    last_fit = lc.created_at if lc is not None else prod.created_at
    due, why = policy_due(policy, new, last_fit, today, force)
    rep = {"policy": policy, "production": prod.model_version, "production_data_cutoff": str(cut),
           "new_completed_starts": int(len(new)), "due": bool(due), "reason": why, "retrained": False}
    if not due:
        return rep
    classes = get_classes(session, pitcher_id)
    res = None
    if reselect or lc is None:
        res = rolling_dev_evaluate(df, classes, prod.model_family, progress=progress)
        sel = selection_from_dev(res)
    else:
        sel = selection_from_entry(session, prod)
    e = final_fit(session, pitcher_id, sel, today, scope="scheduled_retrain", policy=policy,
                  trigger=f"retrain_next_pitch policy={policy}: {why}", tag=f"rt-{policy[:12]}", df=df,
                  classes=classes)
    if res is not None:
        store_dev_run(session, e, res, recipe="re-selected on rolling development folds")
    rep.update(retrained=True, model_version=e.model_version, n_train=e.config["n_train"],
               data_cutoff=e.config["data_cutoff"], recipe_inherited_from=sel.get("inherited_from"))
    if promote:
        rep["promotion"] = lifecycle.promote_with_gates(session, e.model_version,
                                                        source=f"retrain_next_pitch policy={policy}")
    return rep
