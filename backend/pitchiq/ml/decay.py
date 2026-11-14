"""Training-time exponential time-decay sample weights: weight = exp(-lambda * age_days), lambda = ln 2 / half-life.

`age_days` is measured from the fit's reference date (the first day the model is used: a rolling fold's block
start, a final fit's deployment cutoff, a replay's refit day) back to each training pitch's game date, so the
weights only ever depend on dates before the scored games. Weights are normalized to mean 1 over the training set,
which keeps the effective regularization of logistic regression and the smoothing of the frequency baseline on the
same scale as unweighted fits.

The half-life is a recipe parameter (`decay_half_life_days` in the selected params; None = no decay). It is selected
on the rolling-origin development evaluation (`protocol.rolling_dev_evaluate`) by `select_half_life`, after the
family's hyperparameters, and stored with the model so scheduled retrains reproduce it.
"""
from __future__ import annotations

import math
import time
from datetime import date

import numpy as np
import pandas as pd

DECAY_KEY = "decay_half_life_days"
DEFAULT_HALF_LIVES: tuple = (None, 45.0, 90.0, 180.0, 365.0, 730.0)


def split_params(params: dict) -> tuple[dict, float | None]:
    p = dict(params or {})
    h = p.pop(DECAY_KEY, None)
    return p, (float(h) if h is not None else None)


def decay_weights(dates, ref_date: date, half_life_days: float | None) -> np.ndarray | None:
    if half_life_days is None:
        return None
    d = pd.to_datetime(pd.Series(list(dates))).dt.date
    age = np.array([(ref_date - x).days for x in d], dtype=float)
    if (age < 0).any():
        raise ValueError("decay weights: a training pitch is dated after the reference date")
    w = np.exp(-math.log(2.0) * age / float(half_life_days))
    return w / w.mean()


def fit(model, train: pd.DataFrame, ref_date: date):
    """Fit `model` on `train` (needs `label`, and `game_date` when the model has a half-life)."""
    h = getattr(model, "decay_half_life_days", None)
    if h is None:
        return model.fit(train, train["label"])
    return model.fit(train, train["label"], sample_weight=decay_weights(train["game_date"], ref_date, h))


def select_half_life(df: pd.DataFrame, classes: list[str], family: str, base, half_lives=DEFAULT_HALF_LIVES,
                     block_games: int | None = None, features: list[str] | None = None, progress=None):
    """Second selection stage on the same rolling development folds: keep the hyperparameters chosen in `base`
    (a DevResult without decay) and pick the half-life (None = no decay) with the lowest pooled raw log loss.
    LightGBM boosting rounds are searched again for each half-life. Returns the winning DevResult with both
    stages' grid results."""
    from .protocol import rolling_dev_evaluate
    t0 = time.perf_counter()
    params = {k: v for k, v in base.params.items() if k != DECAY_KEY}
    if family == "lightgbm":
        params.pop("n_estimators", None)
    grid = [{**params, DECAY_KEY: h} for h in half_lives]
    res = rolling_dev_evaluate(df, classes, family, grid=grid, features=features, block_games=block_games,
                               progress=progress)
    res.grid_results = ([{**r, "stage": "hyperparameters (no decay)"} for r in base.grid_results] +
                        [{**r, "stage": "decay half-life (hyperparameters fixed)"} for r in res.grid_results])
    res.extra["selection_stages"] = ["hyperparameters without decay", "decay half-life with those hyperparameters"]
    res.extra["half_lives"] = [h for h in half_lives]
    res.seconds += base.seconds
    res.extra["decay_stage_seconds"] = time.perf_counter() - t0
    return res
