"""Automated benchmarks: holdout (test = 2026, retrospective, previously viewed), walk-forward, and feature ablation
(validation only)."""
from __future__ import annotations

import logging
import time
import warnings
from typing import Callable

import numpy as np
import pandas as pd
from sqlalchemy import select, text
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from ..features.schema import FEATURE_GROUPS, feature_list
from ..features.store import get_classes, load_feature_frame
from ..logging_setup import log
from ..models import EvaluationRun, ModelRegistry, PitchPrediction
from . import registry, tracking
from .metrics import breakdowns, cluster_bootstrap, confusion, reliability, summarize
from .models import FAMILIES

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", message=".*eval_set.*")


def _attach_meta(session: Session, df: pd.DataFrame) -> pd.DataFrame:
    meta = pd.read_sql(text("""SELECT pitch_uid, game_pk,
        CASE WHEN inning_topbot='Top' THEN away_team ELSE home_team END AS opponent
        FROM pitches WHERE pitch_uid = ANY(:ids)"""), session.connection(),
                       params={"ids": df["pitch_uid"].astype(int).tolist()})
    return df.merge(meta, on="pitch_uid", how="left")


def full_report(P: np.ndarray, y_idx: np.ndarray, feats: pd.DataFrame, classes: list[str]) -> dict:
    return {
        "summary": summarize(P, y_idx, classes),
        "bootstrap": cluster_bootstrap(P, y_idx, feats["game_pk"].to_numpy()) if "game_pk" in feats else None,
        "pitch_distribution": {c: int((y_idx == j).sum()) for j, c in enumerate(classes)},
        "predicted_distribution": {c: int((P.argmax(1) == j).sum()) for j, c in enumerate(classes)},
        "mean_predicted_probability": {c: float(P[:, j].mean()) for j, c in enumerate(classes)},
        "confusion": confusion(P, y_idx, classes),
        "calibration": reliability(P, y_idx, classes),
        "breakdowns": breakdowns(P, y_idx, feats, classes),
    }


def holdout_benchmark(session: Session, pitcher_id: int, model_version: str | None = None, split: str = "test",
                      store_predictions: bool = True, progress: Callable | None = None) -> EvaluationRun:
    """Predict EVERY eligible pitch of the split with a frozen model; the primary source of official metrics."""
    entry = registry.get_entry(session, pitcher_id, model_version)
    from . import lifecycle  # refit models are trained on every season up to their cutoff: refuse in-sample scoring
    if lifecycle.is_lifecycle_model(entry):
        first = lifecycle._first_date(session, pitcher_id, split=split)
        cut = lifecycle.data_cutoff(session, entry)
        if first is not None and cut is not None and first <= cut:
            raise ValueError(f"{entry.model_version} was trained on data through {cut}; split '{split}' starts "
                             f"{first}, so a holdout benchmark would be in-sample")
    model = registry.load_model(entry)
    classes = entry.classes
    df = _attach_meta(session, load_feature_frame(session, pitcher_id, [split], entry.feature_version))
    t0 = time.perf_counter()
    P = model.predict_proba(df)
    P_raw = model.predict_proba(df, calibrated=False)
    infer_ms = (time.perf_counter() - t0) * 1000
    y = model.y_index(df["label"])
    if np.isnan(y.astype(float)).any():
        raise ValueError("Evaluation labels outside the model's class set")
    y = y.astype(int)
    report = full_report(P, y, df, classes)
    report["uncalibrated_summary"] = summarize(P_raw, y, classes)
    report["inference_ms_total"] = infer_ms
    report["inference_ms_per_pitch"] = infer_ms / max(len(df), 1)
    run = EvaluationRun(pitcher_id=pitcher_id, model_version=entry.model_version, kind="holdout_benchmark",
                        split=split, n_samples=len(df), metrics=report,
                        dataset_definition={"split": split, "feature_version": entry.feature_version,
                                            "n_games": int(df["game_pk"].nunique()),
                                            "pitch_uid_min": int(df.pitch_uid.min()),
                                            "pitch_uid_max": int(df.pitch_uid.max())})
    session.add(run)
    session.flush()
    if store_predictions:
        feat_cols = [f.name for f in feature_list(classes)]
        rows = []
        for i in range(len(df)):
            probs = {c: float(P[i, j]) for j, c in enumerate(classes)}
            top = classes[int(P[i].argmax())]
            rows.append(dict(pitch_uid=int(df.pitch_uid.iloc[i]), model_version=entry.model_version,
                             context="benchmark", evaluation_run_id=run.evaluation_run_id,
                             input_features={k: _py(df[k].iloc[i]) for k in feat_cols if k in df},
                             probabilities=probs, top_pitch=top, actual_pitch=df.label.iloc[i],
                             is_correct_top1=top == df.label.iloc[i], prob_actual=probs[df.label.iloc[i]]))
        for i in range(0, len(rows), 1000):
            session.execute(insert(PitchPrediction).values(rows[i:i + 1000]))
    tracking.mirror_evaluation(run, entry)
    session.flush()
    log(logger, "benchmark", version=entry.model_version, split=split, **report["summary"])
    return run


def benchmark_all(session: Session, pitcher_id: int, split: str = "test", progress: Callable | None = None) -> list[EvaluationRun]:
    """Evaluate the latest model of every family on the SAME held-out examples."""
    runs = []
    fams = registry.latest_by_family(session, pitcher_id)
    for i, (fam, e) in enumerate(sorted(fams.items())):
        runs.append(holdout_benchmark(session, pitcher_id, e.model_version, split))
        if progress:
            progress((i + 1) / len(fams), f"Benchmarked {fam}", i + 1)
    return runs


def walk_forward(session: Session, pitcher_id: int, family: str = "lightgbm", split: str = "test",
                 block_games: int = 6, progress: Callable | None = None) -> EvaluationRun:
    """Expanding-window evaluation: for each block of `split` games, refit on ALL labeled games before the
    block (hyperparameters + temperature frozen from the registered model) and predict the block."""
    entry = registry.latest_by_family(session, pitcher_id)[family]
    base = registry.load_model(entry)
    classes = entry.classes
    allf = _attach_meta(session, load_feature_frame(session, pitcher_id, None, entry.feature_version))
    dates = pd.read_sql(text("SELECT pitch_uid, game_date FROM pitches WHERE pitcher_id=:p"),
                        session.connection(), params={"p": pitcher_id})
    allf = allf.merge(dates, on="pitch_uid")
    target = allf[allf.split == split]
    games = target.groupby("game_pk")["game_date"].first().sort_values()
    blocks = [games.index[i:i + block_games] for i in range(0, len(games), block_games)]
    Ps, ys, parts = [], [], []
    for b, gids in enumerate(blocks):
        start = games.loc[gids].min()
        tr = allf[(allf.game_date < start) & allf.split.isin(["train", "validation", split])]
        te = target[target.game_pk.isin(gids)]
        m = FAMILIES[family](classes, features=base.features, **{k: v for k, v in base.params.items()})
        if family == "lightgbm":
            m.params["n_estimators"] = base.best_iteration_
            m = FAMILIES[family](classes, features=base.features, **m.params)
        m.fit(tr, tr["label"])
        m.calibrator = base.calibrator
        Ps.append(m.predict_proba(te))
        ys.append(m.y_index(te["label"]))
        parts.append(te)
        if progress:
            progress((b + 1) / len(blocks), f"Walk-forward block {b + 1}/{len(blocks)}", b + 1)
    P, y, feats = np.vstack(Ps), np.concatenate(ys).astype(int), pd.concat(parts)
    report = full_report(P, y, feats, classes)
    report["blocks"] = len(blocks)
    run = EvaluationRun(pitcher_id=pitcher_id, model_version=entry.model_version, kind="walk_forward", split=split,
                        n_samples=len(feats), metrics=report,
                        dataset_definition={"block_games": block_games, "refit": "expanding window",
                                            "hyperparameters": "frozen from registered model"})
    session.add(run)
    session.flush()
    tracking.mirror_evaluation(run, entry)
    session.flush()
    return run


def ablation(session: Session, pitcher_id: int, family: str = "lightgbm", progress: Callable | None = None) -> EvaluationRun:
    """Drop-one-group ablation scored on VALIDATION (feature selection must not use the 2026 retrospective season)."""
    entry = registry.latest_by_family(session, pitcher_id)[family]
    base = registry.load_model(entry)
    classes = entry.classes
    df = load_feature_frame(session, pitcher_id, ["train", "validation"], entry.feature_version)
    tr, va = df[df.split == "train"], df[df.split == "validation"]
    groups = {f.name: f.group for f in feature_list(classes)}
    params = {k: v for k, v in base.params.items()}
    results = []
    configs = [("all_features", base.features)] + [
        (f"without_{g}", [f for f in base.features if groups.get(f) != g]) for g in FEATURE_GROUPS
        if any(groups.get(f) == g for f in base.features)]
    for i, (name, feats) in enumerate(configs):
        accs, lls = [], []
        for seed in (0, 1, 2):
            m = FAMILIES[family](classes, features=feats, **params)
            if family == "lightgbm":
                m.params["random_state"] = seed
            m.fit(tr, tr["label"], va, va["label"])
            P = m.predict_proba(va, calibrated=False)
            s = summarize(P, m.y_index(va["label"]), classes)
            accs.append(s["top1_accuracy"]); lls.append(s["log_loss"])
        results.append({"config": name, "n_features": len(feats), "val_top1": float(np.mean(accs)),
                        "val_log_loss": float(np.mean(lls)), "val_log_loss_sd": float(np.std(lls))})
        if progress:
            progress((i + 1) / len(configs), f"Ablation {name}", i + 1)
    ref = results[0]
    for r in results:
        r["delta_log_loss"] = r["val_log_loss"] - ref["val_log_loss"]  # >0 => removing the group hurts
        r["delta_top1"] = r["val_top1"] - ref["val_top1"]
    run = EvaluationRun(pitcher_id=pitcher_id, model_version=entry.model_version, kind="ablation", split="validation",
                        n_samples=len(va), metrics={"results": results, "seeds": 3, "note":
                        "Uncalibrated validation metrics; positive delta_log_loss means the group helps."},
                        dataset_definition={"train": "train", "score": "validation"})
    session.add(run)
    session.flush()
    tracking.mirror_evaluation(run, entry)
    session.flush()
    return run


def _py(v):
    if v is None:
        return None
    if isinstance(v, float) and np.isnan(v):
        return None
    if hasattr(v, "item"):
        v = v.item()
        if isinstance(v, float) and np.isnan(v):
            return None
    return v
