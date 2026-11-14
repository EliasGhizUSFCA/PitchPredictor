"""Historical replay of retraining policies.

The replay simulates the live retraining state machine (`refit.policy_due`, the rule job `retrain_next_pitch` and
`refit.after_ingest` apply) day by day, continuously across all replayed seasons:

* At the first game of the first replay season a model is deployed, fit on all games dated before that day
  (``initial_fit``; not counted as a refit).
* On every calendar day t after that, before that day's game, the policy is checked exactly as live: new completed
  starts are the games dated after the deployed model's data cutoff and before t, the last-fit time is the day of the
  last (re)fit, and ``refit.policy_due(policy, new, last_fit, t, force=False)`` decides. A due refit trains on all
  games dated before t and becomes the deployed model.
* ``manual``: live, the job only runs when an operator triggers it. The replay assumes no operator trigger, so the
  initial deployment serves throughout (a lower bound: no retraining at all).

State (deployed model, data cutoff, last-fit day, starts since the cutoff) carries across the offseason, so e.g.
``every_6_starts`` counts late-2025 starts toward its first 2026 refit and nothing gets a free season-start refit.

Each fit uses the family's frozen recipe (hyperparameters selected on rolling development folds) and only games dated
strictly before its day; identical training sets are fit once and shared across policies. Probabilities are the raw
model outputs (no temperature), so policies are compared on the fit cadence alone.
"""
from __future__ import annotations

import logging
from datetime import datetime, time as dtime, timedelta, timezone
from typing import Callable

import numpy as np
import pandas as pd
from sqlalchemy.orm import Session

from ..features.store import get_classes
from ..logging_setup import log
from ..models import EvaluationRun, ModelRegistry
from . import lifecycle, tracking
from .metrics import cluster_bootstrap, confusion, reliability, summarize
from . import decay
from .protocol import evaluation_role, game_table, load_dated_frame, make_model, role_label
from .refit import POLICIES, policy_due, tag_evaluation_run

logger = logging.getLogger(__name__)

SIMULATION = ("live state machine refit.policy_due checked daily before each day's game, continuous across the "
              "replayed seasons; initial deployment at the first replayed game fit on all earlier games (not a "
              "refit); manual = no operator trigger (initial deployment serves throughout)")


def simulate_policy(policy: str, games: pd.DataFrame, start, end) -> tuple[list, list[dict]]:
    """Replay `refit.policy_due` day by day from `start` (first replayed game date) to `end` (last one).

    `games` is the full chronological game table (all seasons). Returns (train_before date of the model deployed for
    each game with start <= game_date <= end, in order; fit events). A model with train_before d was fit on all games
    dated strictly before d."""
    if policy not in POLICIES:
        raise ValueError(f"Unknown policy {policy}")
    gdates = np.array(list(games.game_date))
    prior = gdates[gdates < start]
    train_before, cutoff, last_fit = start, (prior.max() if len(prior) else None), start
    events = [{"date": start, "train_before": start, "kind": "initial_fit", "reason": "deployment at replay start"}]
    target = list(games[(games.game_date >= start) & (games.game_date <= end)].game_date)
    out, i, day = [], 0, start
    while day <= end:
        if day > start and policy != "manual":
            new = games[games.game_date < day]
            if cutoff is not None:
                new = new[new.game_date > cutoff]
            due, why = policy_due(policy, new, datetime.combine(last_fit, dtime(0), tzinfo=timezone.utc), day,
                                  False)
            if due:
                train_before, last_fit = day, day
                before = gdates[gdates < day]
                cutoff = before.max() if len(before) else None
                events.append({"date": day, "train_before": day, "kind": "refit", "reason": why})
        while i < len(target) and target[i] == day:
            out.append(train_before)
            i += 1
        day += timedelta(days=1)
    return out, events


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


def replay_seasons(df: pd.DataFrame, classes: list[str], family: str, params: dict, seasons,
                   policies=POLICIES, cache: dict | None = None,
                   features: list[str] | None = None) -> dict[int, dict]:
    """Continuous replay over `seasons` (state carried across the offseason); metrics reported per season."""
    games = game_table(df)
    seasons = sorted(int(x) for x in seasons)
    tg = games[games.season.isin(seasons)].reset_index(drop=True)
    if tg.empty:
        raise ValueError(f"no games in seasons {seasons}")
    missing = [x for x in seasons if x not in set(tg.season)]
    if missing:
        raise ValueError(f"no games in season(s) {missing}")
    start, end = tg.game_date.min(), tg.game_date.max()
    cache = cache if cache is not None else {}
    idx = {c: i for i, c in enumerate(classes)}
    dates = df["game_date"].to_numpy()
    out: dict[int, dict] = {x: {} for x in seasons}
    for pol in policies:
        cuts, events = simulate_policy(pol, games, start, end)
        by_game = dict(zip(tg.game_pk, cuts))
        prev_end = None
        for season in seasons:
            sg = tg[tg.season == season]
            Ps, ys, gpks, used = [], [], [], []
            for gpk in sg.game_pk:
                cut = by_game[gpk]
                n_before = int((dates < cut).sum())
                key = (family, n_before)
                if key not in cache:
                    tr = df.iloc[:n_before]  # df is sorted chronologically: exactly the rows dated before `cut`
                    assert (tr.game_date < cut).all()
                    cache[key] = decay.fit(make_model(family, classes, params, features), tr, cut)
                te = df[df.game_pk == gpk]
                Ps.append(cache[key].predict_proba(te, calibrated=False))
                ys.append(te["label"].map(idx).to_numpy().astype(int))
                gpks.append(te["game_pk"].to_numpy())
                if (str(cut), n_before) not in used:
                    used.append((str(cut), n_before))
            P, y, g = np.vstack(Ps), np.concatenate(ys), np.concatenate(gpks)
            s_end = sg.game_date.max()
            in_season = [e for e in events if e["kind"] == "refit" and e["date"] <= s_end
                         and (prev_end is None or e["date"] > prev_end)]
            s = summarize(P, y, classes)
            out[season][pol] = {
                **s, "ece": reliability(P, y, classes)["ece_top_label"],
                "per_class": confusion(P, y, classes)["per_class"],
                "bootstrap": cluster_bootstrap(P, y, g, n_boot=300),
                "n_refits": len(in_season), "n_models_used": len(used),
                "refit_dates": [str(e["date"]) for e in in_season],
                "models_used": [{"train_before": c, "n_train": n} for c, n in used],
                "initial_fit": ({"train_before": str(start), "n_train": int((dates < start).sum())}
                                if prev_end is None else None),
                "n_refits_whole_replay": sum(e["kind"] == "refit" for e in events),
                "n_games": int(len(sg))}
            prev_end = s_end
    return out


def replay_season(df: pd.DataFrame, classes: list[str], family: str, params: dict, season: int,
                  policies=POLICIES, cache: dict | None = None) -> dict:
    """Single-season replay (deployment at that season's first game)."""
    return replay_seasons(df, classes, family, params, [season], policies, cache)[int(season)]


def replay_policies(session: Session, pitcher_id: int, families=("frequency", "logreg", "lightgbm"),
                    seasons=(2025, 2026), policies=POLICIES, progress: Callable | None = None) -> list[EvaluationRun]:
    classes = get_classes(session, pitcher_id)
    frames: dict = {}
    runs = []
    steps = len(families) * len(seasons)
    k = 0
    for fam in families:
        entry = _recipe(session, pitcher_id, fam)
        params = lifecycle.lifecycle_row(session, entry.model_version).selected_params
        if entry.feature_version not in frames:  # the recipe's own feature version and feature list
            frames[entry.feature_version] = load_dated_frame(session, pitcher_id, entry.feature_version)
        df = frames[entry.feature_version]
        allres = replay_seasons(df, classes, fam, params, seasons, policies, {},
                                features=(entry.feature_schema or {}).get("used_features"))
        replay_start = str(min(d for d in df[df.game_season == min(seasons)].game_date))
        for season in sorted(seasons):
            role = evaluation_role(season)
            res = allres[int(season)]
            split = df[df.game_season == season]["split"].iloc[0]
            best = min(res, key=lambda p: res[p]["log_loss"])
            metrics = {"evaluation_role": role, "evaluation_label": role_label(role), "season": season,
                       "family": fam, "hyperparameters": params, "policies": res, "lowest_log_loss_policy": best,
                       "probabilities": "raw (uncalibrated) model outputs",
                       "simulation": SIMULATION, "replay_seasons": sorted(int(x) for x in seasons),
                       "replay_start": replay_start,
                       "note": ("Hyperparameters were selected on rolling development folds that include 2025 games, "
                                "so absolute 2025 numbers are optimistic; every policy shares that bias. 2026 is "
                                "retrospective (previously viewed) evidence and was not used to choose anything.")}
            run = EvaluationRun(pitcher_id=pitcher_id, model_version=entry.model_version,
                                kind="retraining_policy_replay", split=split,
                                n_samples=int(res[policies[0]]["n"]), metrics=metrics,
                                dataset_definition={"season": season, "evaluation_role": role,
                                                    "policies": list(policies), "feature_version":
                                                    entry.feature_version, "recipe_model": entry.model_version,
                                                    "training": "all labeled games dated before each (re)fit day",
                                                    "simulation": SIMULATION,
                                                    "replay_seasons": sorted(int(x) for x in seasons),
                                                    "replay_start": replay_start})
            session.add(run)
            session.flush()
            tracking.mirror_evaluation(run, entry)
            tag_evaluation_run(run)
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
            for k in ("log_loss", "brier", "top1_accuracy", "top2_accuracy", "ece", "n_refits", "n_models_used"):
                if d.get(k) is not None:
                    t.client.log_metric(rid, f"policy/{pol}/{k}", float(d[k]))
        t.client.set_tag(rid, "pitchiq.evaluation_role", run.metrics["evaluation_role"])
    except Exception as e:  # noqa: BLE001
        log(logger, "mlflow_mirror_failed", op="policy_metrics", error=f"{type(e).__name__}: {e}"[:500])
