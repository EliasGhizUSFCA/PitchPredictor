"""v2 feature-group and time-decay ablation on the rolling-origin development evaluation (2024-2025 folds).

For one family's dev-selected v2 recipe (hyperparameters, LightGBM rounds and decay half-life fixed), every
configuration is scored on the same rolling folds as the selection objective (`protocol.rolling_dev_evaluate`,
expanding window, no fold trains on a date at or after its block):

* ``recipe`` / ``recipe_no_decay``: the recipe's own feature list with / without its decay (reference when present)
* ``v2_all``: every v2 model feature, recipe decay (reference when the recipe uses all of them)
* ``v2_all_no_decay``: same features, no time-decay weights
* ``v1_features`` / ``v1_features_decay``: the v1 model features without / with the recipe's decay
* ``v2_without_<group>``: one new v2 group removed ("off")
* ``v1_plus_<group>``: one new v2 group added to the v1 features ("on")

Reported per configuration (mean over seeds for LightGBM, whose row/column subsampling is seeded): pooled raw log loss,
Brier, top-1, and for every class the class-conditional log loss (mean -log p(c) over pitches whose actual type is c),
the one-vs-rest Brier score, mean predicted probability vs observed frequency and recall. The question asked of the
changeup (CH) and sinker (SI) columns is whether their probability estimates improve, not their argmax recall: no
configuration re-weights classes or forces rare-class predictions.
"""
from __future__ import annotations

import logging
import time
from typing import Callable

import numpy as np
import pandas as pd
from sqlalchemy.orm import Session

from ..features import v2 as fv2
from ..features.schema import FEATURE_VERSION, default_model_features, feature_list
from ..features.store import get_classes
from ..logging_setup import log
from ..models import EvaluationRun, ModelRegistry
from . import lifecycle, tracking
from .decay import DECAY_KEY
from .metrics import summarize
from .protocol import load_dated_frame, role_label, rolling_dev_evaluate

logger = logging.getLogger(__name__)
KIND = "feature_ablation_v2"


def per_class_scores(P: np.ndarray, y: np.ndarray, classes: list[str]) -> dict:
    P = np.clip(P, 1e-15, 1.0)
    top = P.argmax(1)
    out = {}
    for j, c in enumerate(classes):
        m = y == j
        onehot = m.astype(float)
        out[c] = {"n": int(m.sum()),
                  "log_loss": float(-np.log(P[m, j]).mean()) if m.any() else None,
                  "brier_ovr": float(np.mean((P[:, j] - onehot) ** 2)),
                  "mean_pred": float(P[:, j].mean()), "observed_freq": float(onehot.mean()),
                  "mean_pred_when_actual": float(P[m, j].mean()) if m.any() else None,
                  "recall": float((top[m] == j).mean()) if m.any() else None}
    return out


def _configs(classes: list[str], half_life, recipe_features: list[str] | None = None
             ) -> list[tuple[str, list[str], float | None]]:
    groups = {f.name: f.group for f in feature_list(classes, FEATURE_VERSION)}
    v2_all = all_v2_model_features(classes)
    v1 = default_model_features(classes, "v1")
    out = []
    if recipe_features is not None and list(recipe_features) != list(v2_all):
        out.append(("recipe", list(recipe_features), half_life))
        if half_life is not None:
            out.append(("recipe_no_decay", list(recipe_features), None))
    out.append(("v2_all", v2_all, half_life))
    if half_life is not None:
        out.append(("v2_all_no_decay", v2_all, None))
    out.append(("v1_features", v1, None))
    if half_life is not None:
        out.append(("v1_features_decay", v1, half_life))
    for g in fv2.GROUPS:
        members = [f for f in v2_all if groups.get(f) == g]
        if not members:
            continue
        out.append((f"v2_without_{g}", [f for f in v2_all if groups.get(f) != g], half_life))
        out.append((f"v1_plus_{g}", v1 + members, half_life))
    return out


def all_v2_model_features(classes: list[str]) -> list[str]:
    """Every v2 model input (no group exclusion): v1 inputs + all v2 additions, minus MODEL_EXCLUDED_BY_VERSION."""
    from ..features.schema import MODEL_EXCLUDED_BY_VERSION
    return [f.name for f in feature_list(classes, FEATURE_VERSION) if f.name not in MODEL_EXCLUDED_BY_VERSION["v2"]]


def recipe_entry(session: Session, pitcher_id: int, family: str) -> ModelRegistry:
    """Newest lifecycle (rolling-dev-selected) model of the family built on the current feature version."""
    from sqlalchemy import select
    rows = session.scalars(select(ModelRegistry).where(ModelRegistry.pitcher_id == pitcher_id,
                                                       ModelRegistry.model_family == family,
                                                       ModelRegistry.feature_version == FEATURE_VERSION)
                           .order_by(ModelRegistry.created_at.desc())).all()
    for r in rows:
        if lifecycle.is_lifecycle_model(r):
            return r
    raise LookupError(f"No lifecycle {family} {FEATURE_VERSION} model; run `pitchiq refit --decay` first")


def run_ablation(session: Session, pitcher_id: int, family: str = "lightgbm", seeds=(0, 1, 2),
                 configs: list[str] | None = None, block_games: int | None = None,
                 progress: Callable | None = None, store: bool = True, params: dict | None = None,
                 params_from: str | None = None) -> dict:
    """Default: the newest v2 lifecycle recipe of `family` (its hyperparameters, decay half-life and feature list;
    reference config "recipe" when its features differ from v2_all). With `params` (+ the model version they came
    from, `params_from`), those hyperparameters are used instead and the reference is v2_all."""
    from . import registry
    if params is None:
        entry = recipe_entry(session, pitcher_id, family)
        lc = lifecycle.lifecycle_row(session, entry.model_version)
        params = dict(lc.selected_params)
        recipe_features = (entry.feature_schema or {}).get("used_features")
    else:
        entry = registry.get_entry(session, pitcher_id, params_from)
        params = dict(params)
        recipe_features = None
    half_life = params.pop(DECAY_KEY, None)
    classes = get_classes(session, pitcher_id, FEATURE_VERSION)
    df = load_dated_frame(session, pitcher_id, FEATURE_VERSION)
    cfgs = _configs(classes, half_life, recipe_features)
    if configs:
        cfgs = [c for c in cfgs if c[0] in set(configs) | {cfgs[0][0]}]
    seeds = tuple(seeds) if family == "lightgbm" else (0,)
    t0 = time.perf_counter()
    results = []
    for i, (name, feats, h) in enumerate(cfgs):
        lls, briers, top1s, pcs = [], [], [], []
        n = None
        for seed in seeds:
            p = {**params, DECAY_KEY: h}
            if family == "lightgbm":
                p["random_state"] = seed
            res = rolling_dev_evaluate(df, classes, family, grid=[p], features=feats, block_games=block_games,
                                       fixed_rounds=True)
            s = res.summary_raw
            lls.append(s["log_loss"]); briers.append(s["brier"]); top1s.append(s["top1_accuracy"])
            pcs.append(per_class_scores(res.P_raw, res.y, classes))
            n = int(len(res.y))
        per_class = {c: {k: (float(np.mean([pc[c][k] for pc in pcs])) if pcs[0][c][k] is not None else None)
                         for k in pcs[0][c]} for c in classes}
        results.append({"config": name, "n_features": len(feats), "decay_half_life_days": h,
                        "val_log_loss": float(np.mean(lls)), "val_log_loss_sd": float(np.std(lls)),
                        "val_brier": float(np.mean(briers)), "val_top1": float(np.mean(top1s)),
                        "per_seed_log_loss": lls, "per_class": per_class, "n": n})
        log(logger, "feature_ablation_config", family=family, config=name, log_loss=round(float(np.mean(lls)), 4),
            ch_ll=round(per_class.get("CH", {}).get("log_loss") or 0, 4),
            si_ll=round(per_class.get("SI", {}).get("log_loss") or 0, 4))
        if progress:
            progress((i + 1) / len(cfgs), f"ablation {name}", i + 1)
    ref = results[0]
    for r in results:
        r["delta_log_loss"] = r["val_log_loss"] - ref["val_log_loss"]
        r["delta_top1"] = r["val_top1"] - ref["val_top1"]
        r["delta_brier"] = r["val_brier"] - ref["val_brier"]
        for c in classes:
            a, b = r["per_class"][c], ref["per_class"][c]
            if a["log_loss"] is not None and b["log_loss"] is not None:
                a["delta_log_loss"] = a["log_loss"] - b["log_loss"]
            a["delta_brier_ovr"] = a["brier_ovr"] - b["brier_ovr"]
    v1 = next((r for r in results if r["config"] == "v1_features"), None)
    if v1 is not None:
        for r in results:
            r["delta_log_loss_vs_v1"] = r["val_log_loss"] - v1["val_log_loss"]
    metrics = {"results": results, "reference": ref["config"], "seeds": len(seeds), "family": family,
               "recipe_model": entry.model_version, "hyperparameters": params,
               "decay_half_life_days": half_life, "evaluation_role": "development",
               "evaluation_label": role_label("development"), "seconds": time.perf_counter() - t0,
               "note": ("Rolling-origin development folds (2024-2025 blocks, expanding window), raw probabilities, "
                        "recipe hyperparameters fixed. delta_* = config - reference (first config): a positive "
                        "delta_log_loss for v2_without_<group> means the group helps; delta_log_loss_vs_v1 < 0 means "
                        "better than the v1 "
                        "features. per_class.log_loss is the class-conditional log loss (mean -log p over pitches of "
                        "that actual class); brier_ovr is the one-vs-rest Brier score over all pitches. The recipe was "
                        "selected on these folds, so v2_all is optimistic by the selection effect; the other configs "
                        "reuse its hyperparameters.")}
    out = {"metrics": metrics, "evaluation_run_id": None}
    if store:
        run = EvaluationRun(pitcher_id=pitcher_id, model_version=entry.model_version, kind=KIND, split="development",
                            n_samples=int(ref["n"]), metrics=metrics,
                            dataset_definition={"protocol": "rolling_origin_expanding_window",
                                                "feature_version": FEATURE_VERSION, "evaluation_role": "development",
                                                "configs": [c[0] for c in cfgs], "seeds": list(seeds)})
        session.add(run)
        session.flush()
        tracking.mirror_evaluation(run, entry)
        _log_class_metrics(run, classes)
        session.flush()
        out["evaluation_run_id"] = run.evaluation_run_id
    return out


def _log_class_metrics(run: EvaluationRun, classes: list[str]) -> None:
    t = tracking.get_tracker()
    rid = (run.metrics or {}).get("mlflow_run_id")
    if not rid or not t.enabled:
        return
    try:
        from .refit import tag_evaluation_run
        tag_evaluation_run(run)
        for r in run.metrics["results"]:
            t.client.log_metric(rid, f"ablation/{r['config']}/val_brier", float(r["val_brier"]))
            for c in classes:
                pc = r["per_class"][c]
                for k in ("log_loss", "brier_ovr", "mean_pred", "recall"):
                    if pc.get(k) is not None:
                        t.client.log_metric(rid, f"ablation/{r['config']}/{c}_{k}", float(pc[k]))
    except Exception as e:  # noqa: BLE001
        log(logger, "mlflow_mirror_failed", op="feature_ablation_metrics", error=f"{type(e).__name__}: {e}"[:500])


def summary_table(metrics: dict, classes=("CH", "SI")) -> pd.DataFrame:
    rows = []
    for r in metrics["results"]:
        row = {"config": r["config"], "n_features": r["n_features"], "log_loss": r["val_log_loss"],
               "sd": r["val_log_loss_sd"], "delta": r["delta_log_loss"], "brier": r["val_brier"],
               "top1": r["val_top1"]}
        for c in classes:
            pc = r["per_class"].get(c, {})
            row[f"{c}_ll"] = pc.get("log_loss")
            row[f"{c}_brier"] = pc.get("brier_ovr")
        rows.append(row)
    return pd.DataFrame(rows)
