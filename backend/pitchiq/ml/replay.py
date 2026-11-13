"""Historical replay of retraining policies.

For a replay season, every game is predicted by the model a policy would have had deployed on that game's date:

* ``manual`` - fit once on all games before the season's first game, never refit during the season;
* ``after_each_start`` - refit after every start (game g uses all games before g);
* ``every_6_starts`` - refit after starts 6, 12, ... of the season;
* ``weekly`` - a refit runs each Monday 00:00 on all games before that Monday (game on date d uses the latest
  Monday <= d, or the season-start fit); runs without new games produce the same model and are not counted as refits.

Each fit uses the family's frozen recipe (hyperparameters selected on rolling development folds) and only games dated
strictly before the cutoff; identical training sets are fit once and shared across policies. Probabilities are the
raw model outputs (no temperature), so policies are compared on the fit cadence alone.
"""
from __future__ import annotations

import logging
from datetime import timedelta
from typing import Callable

import numpy as np
import pandas as pd
from sqlalchemy.orm import Session

from ..features.store import get_classes
from ..logging_setup import log
from ..models import EvaluationRun, ModelRegistry
from . import lifecycle, tracking
from .metrics import cluster_bootstrap, confusion, reliability, summarize
from .protocol import evaluation_role, game_table, load_dated_frame, make_model, role_label
from .refit import POLICIES

logger = logging.getLogger(__name__)


def policy_cutoffs(policy: str, games: pd.DataFrame) -> list:
    """Training cutoff date (exclusive) for each game of the replay season, in order."""
    dates = list(games.game_date)
    first = dates[0]
    if policy == "manual":
        return [first] * len(dates)
    if policy == "after_each_start":
        return dates
    if policy == "every_6_starts":
        return [dates[(i // 6) * 6] for i in range(len(dates))]
    if policy == "weekly":
        return [max(first, d - timedelta(days=d.weekday())) for d in dates]
    raise ValueError(f"Unknown policy {policy}")


def _recipe(session: Session, pitcher_id: int, family: str) -> ModelRegistry:
    """Newest lifecycle (rolling-dev-selected) model of the family; its recipe is replayed."""
    from sqlalchemy import select
    rows = session.scalars(select(ModelRegistry).where(ModelRegistry.pitcher_id == pitcher_id,
                                                       ModelRegistry.model_family == family)
                           .order_by(ModelRegistry.created_at.desc())).all()
    for r in rows:
        if lifecycle.is_lifecycle_model(r):
            return r
    raise LookupError(f"No lifecycle {family} model; run `pitchiq refit` first")


def replay_season(df: pd.DataFrame, classes: list[str], family: str, params: dict, season: int,
                  policies=POLICIES, cache: dict | None = None) -> dict:
    games = game_table(df)
    target = games[games.season == season].reset_index(drop=True)
    if target.empty:
        raise ValueError(f"no games in season {season}")
    cache = cache if cache is not None else {}
    idx = {c: i for i, c in enumerate(classes)}
    dates = df["game_date"].to_numpy()
    out = {}
    for pol in policies:
        cuts = policy_cutoffs(pol, target)
        Ps, ys, gpks, fits = [], [], [], []
        for gpk, cut in zip(target.game_pk, cuts):
            n_before = int((dates < cut).sum())
            key = (family, n_before)
            if key not in cache:
                tr = df.iloc[:n_before]  # df is sorted chronologically, so these are exactly the rows before `cut`
                assert (tr.game_date < cut).all()
                cache[key] = make_model(family, classes, params).fit(tr, tr["label"])
            te = df[df.game_pk == gpk]
            Ps.append(cache[key].predict_proba(te, calibrated=False))
            ys.append(te["label"].map(idx).to_numpy().astype(int))
            gpks.append(te["game_pk"].to_numpy())
            fits.append((str(cut), n_before))
        P, y, g = np.vstack(Ps), np.concatenate(ys), np.concatenate(gpks)
        distinct = []
        for f in fits:
            if f not in distinct:
                distinct.append(f)
        s = summarize(P, y, classes)
        out[pol] = {**s, "ece": reliability(P, y, classes)["ece_top_label"],
                    "per_class": confusion(P, y, classes)["per_class"],
                    "bootstrap": cluster_bootstrap(P, y, g, n_boot=300),
                    "n_fits": len(distinct), "n_refits": len(distinct) - 1,
                    "scheduled_runs": len(set(cuts)) - 1 if pol == "weekly" else None,
                    "fit_cutoffs": [{"train_before": c, "n_train": n} for c, n in distinct],
                    "n_games": int(len(target))}
    return out


def replay_policies(session: Session, pitcher_id: int, families=("frequency", "logreg", "lightgbm"),
                    seasons=(2025, 2026), policies=POLICIES, progress: Callable | None = None) -> list[EvaluationRun]:
    classes = get_classes(session, pitcher_id)
    df = load_dated_frame(session, pitcher_id)
    runs = []
    steps = len(families) * len(seasons)
    k = 0
    for fam in families:
        entry = _recipe(session, pitcher_id, fam)
        params = lifecycle.lifecycle_row(session, entry.model_version).selected_params
        cache: dict = {}
        for season in seasons:
            role = evaluation_role(season)
            res = replay_season(df, classes, fam, params, season, policies, cache)
            split = df[df.game_season == season]["split"].iloc[0]
            best = min(res, key=lambda p: res[p]["log_loss"])
            metrics = {"evaluation_role": role, "evaluation_label": role_label(role), "season": season,
                       "family": fam, "hyperparameters": params, "policies": res, "lowest_log_loss_policy": best,
                       "probabilities": "raw (uncalibrated) model outputs",
                       "note": ("Hyperparameters were selected on rolling development folds that include 2025 games, "
                                "so absolute 2025 numbers are optimistic; every policy shares that bias. 2026 is "
                                "retrospective (previously viewed) evidence and was not used to choose anything.")}
            run = EvaluationRun(pitcher_id=pitcher_id, model_version=entry.model_version,
                                kind="retraining_policy_replay", split=split,
                                n_samples=int(res[policies[0]]["n"]), metrics=metrics,
                                dataset_definition={"season": season, "evaluation_role": role,
                                                    "policies": list(policies), "feature_version":
                                                    entry.feature_version, "recipe_model": entry.model_version,
                                                    "training": "all labeled games dated before each cutoff"})
            session.add(run)
            session.flush()
            tracking.mirror_evaluation(run, entry)
            _log_policy_metrics(run)
            session.flush()
            runs.append(run)
            log(logger, "policy_replay", family=fam, season=season,
                **{p: round(res[p]["log_loss"], 4) for p in policies})
            k += 1
            if progress:
                progress(k / steps, f"replayed {fam} {season}", k)
    return runs


def _log_policy_metrics(run: EvaluationRun) -> None:
    t = tracking.get_tracker()
    rid = (run.metrics or {}).get("mlflow_run_id")
    if not rid or not t.enabled:
        return
    try:
        for pol, d in run.metrics["policies"].items():
            for k in ("log_loss", "brier", "top1_accuracy", "top2_accuracy", "ece", "n_refits"):
                if d.get(k) is not None:
                    t.client.log_metric(rid, f"policy/{pol}/{k}", float(d[k]))
        t.client.set_tag(rid, "pitchiq.evaluation_role", run.metrics["evaluation_role"])
    except Exception as e:  # noqa: BLE001
        log(logger, "mlflow_mirror_failed", op="policy_metrics", error=f"{type(e).__name__}: {e}"[:500])
