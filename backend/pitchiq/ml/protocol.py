"""Next-pitch evaluation protocol: evaluation roles + rolling-origin chronological development evaluation.

Roles (see `Settings`):

* **development** (2023-2025): everything used to choose or tune a model — hyperparameter grids, the model family,
  the calibration temperature and the promotion gates — is computed here, chronologically.
* **retrospective_viewed** (2026): the former "locked test" season. It has been viewed, so numbers on it are reported
  as "retrospective (previously viewed)" and never feed a decision.
* **prospective**: games newer than every configured season (e.g. 2027) and the prospective ledger.

`pre_pitch_states.split` keeps its train / validation / test values; the role is derived from the season.

Rolling-origin development evaluation: the games of `dev_eval_seasons` are cut into consecutive blocks of
`dev_block_games` whole games. For each block the model is fit on **every labeled pitch from games played on dates
strictly before the block's first game** (expanding window) and scores the block. Games are never split and no
fold ever trains on a date at or after the block it scores. The pooled out-of-sample predictions give the selection
objective (multiclass log loss) plus Brier, top-1/2, ECE and per-class metrics.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd
from sqlalchemy import text
from sqlalchemy.orm import Session

from ..config import get_settings
from ..features.schema import FEATURE_VERSION
from ..features.store import load_feature_frame
from ..logging_setup import log
from .metrics import log_loss, reliability, summarize
from .models import FAMILIES, FrequencyBaseline, TemperatureCalibrator

logger = logging.getLogger(__name__)

ROLE_LABELS = {
    "development": "development (chronological, used for selection/tuning/calibration/gates)",
    "retrospective_viewed": "retrospective (previously viewed)",
    "prospective": "prospective (not used by any decision)",
}

# Hyperparameter grids searched on the rolling development objective. LightGBM's number of boosting rounds is part
# of the search: each fold model is fit once with LGBM_MAX_ROUNDS rounds and scored at every checkpoint, which is
# exactly the score of the same model trained with n_estimators=checkpoint (no early stopping on scored data).
LGBM_MAX_ROUNDS = 800
LGBM_CHECKPOINT_STEP = 25
DEV_GRIDS: dict[str, list[dict]] = {
    "majority": [{}],
    "frequency": [{"alpha": a, "levels": lv} for a in (5.0, 20.0, 50.0)
                  for lv in (FrequencyBaseline.DEFAULT_LEVELS[:2], FrequencyBaseline.DEFAULT_LEVELS)],
    "logreg": [{"C": c} for c in (0.01, 0.03, 0.1, 0.3, 1.0)],
    "lightgbm": [{"num_leaves": nl, "min_child_samples": mc} for nl in (7, 15, 31) for mc in (20, 60, 120)],
}


CALIBRATION_RULE = ("apply the pooled out-of-sample temperature only if chronologically calibrated rolling "
                    "development log loss < raw rolling development log loss; otherwise deploy raw outputs (T=1)")


# ------------------------------------------------------------------ roles
def evaluation_role(season: int) -> str:
    s = get_settings()
    if season in s.development_seasons:
        return "development"
    if season in s.retrospective_viewed_seasons:
        return "retrospective_viewed"
    return "prospective"


def split_role(split: str) -> str:
    """Role of a pre_pitch_states split value under the current configuration."""
    s = get_settings()
    seasons = {"train": s.train_seasons, "validation": s.validation_seasons, "test": s.test_seasons}.get(split)
    if not seasons:
        return "development" if split == "development" else "prospective"
    roles = {evaluation_role(x) for x in seasons}
    return roles.pop() if len(roles) == 1 else "mixed"


def role_label(role: str) -> str:
    return ROLE_LABELS.get(role, role)


def protocol_dict() -> dict:
    s = get_settings()
    return {
        "roles": {"development": s.development_seasons, "retrospective_viewed": s.retrospective_viewed_seasons},
        "labels": ROLE_LABELS,
        "split_roles": {sp: split_role(sp) for sp in ("train", "validation", "test")},
        "split_seasons": {"train": s.train_seasons, "validation": s.validation_seasons, "test": s.test_seasons},
        "rolling_development": {"eval_seasons": s.dev_eval_seasons, "block_games": s.dev_block_games,
                                "window": "expanding; train on all labeled games dated before the block",
                                "primary_metric": "multiclass log loss",
                                "also": ["brier", "top1", "top2", "ece", "per-class precision/recall"]},
        "gates": gate_thresholds(),
        "note": ("2026 was the 'locked test' season and has been viewed twice; it is retrospective evidence only. "
                 "A clean confirmation needs data no decision has seen (the prospective ledger)."),
    }


def gate_thresholds() -> dict:
    s = get_settings()
    return {"max_dev_log_loss_regression": s.gate_max_dev_log_loss_regression, "max_dev_ece": s.gate_max_dev_ece,
            "min_training_samples": s.gate_min_training_samples, "prob_tolerance": s.gate_prob_tolerance}


# ------------------------------------------------------------------ data
def load_dated_frame(session: Session, pitcher_id: int, feature_version: str = FEATURE_VERSION) -> pd.DataFrame:
    """All labeled pre-pitch rows with game_pk, game_date, season and opponent, in chronological order."""
    df = load_feature_frame(session, pitcher_id, None, feature_version)
    if df.empty:
        return pd.DataFrame(columns=["pitch_uid", "split", "label", "game_pk", "game_date", "game_season",
                                     "opponent", "season"])
    meta = pd.read_sql(text("""SELECT pitch_uid, game_pk, game_date, season AS game_season,
        CASE WHEN inning_topbot='Top' THEN away_team ELSE home_team END AS opponent
        FROM pitches WHERE pitcher_id = :p"""), session.connection(), params={"p": pitcher_id})
    meta["game_date"] = pd.to_datetime(meta["game_date"]).dt.date
    df = df.merge(meta, on="pitch_uid", how="left")
    if "season" not in df:
        df["season"] = df["game_season"]
    df = df.sort_values(["game_date", "game_pk", "pitch_uid"], kind="stable").reset_index(drop=True)
    return df


def game_table(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("game_pk").agg(game_date=("game_date", "first"), season=("game_season", "first"),
                                  n=("pitch_uid", "size")).reset_index()
    return g.sort_values(["game_date", "game_pk"]).reset_index(drop=True)


@dataclass
class Fold:
    index: int
    eval_games: list[int]
    start_date: object  # first game date of the block; training uses dates strictly before it
    end_date: object
    train_idx: np.ndarray
    eval_idx: np.ndarray
    skipped: str | None = None

    def meta(self) -> dict:
        return {"fold": self.index, "eval_games": [int(g) for g in self.eval_games], "start": str(self.start_date),
                "end": str(self.end_date), "n_train": int(len(self.train_idx)), "n_eval": int(len(self.eval_idx)),
                "skipped": self.skipped}


def rolling_folds(df: pd.DataFrame, classes: list[str], eval_seasons: list[int] | None = None,
                  block_games: int | None = None) -> list[Fold]:
    s = get_settings()
    eval_seasons = eval_seasons or s.dev_eval_seasons
    block_games = int(block_games or s.dev_block_games)
    dev = set(s.development_seasons)
    if not set(eval_seasons) <= dev:
        raise ValueError(f"Rolling development folds must stay inside development seasons {sorted(dev)}")
    games = game_table(df)
    target = games[games.season.isin(eval_seasons)]
    dates = df["game_date"].to_numpy()
    folds = []
    for b, i in enumerate(range(0, len(target), block_games)):
        blk = target.iloc[i:i + block_games]
        start, end = blk.game_date.min(), blk.game_date.max()
        train_idx = np.where(dates < start)[0]
        eval_idx = np.where(df["game_pk"].isin(blk.game_pk).to_numpy())[0]
        f = Fold(b, blk.game_pk.tolist(), start, end, train_idx, eval_idx)
        seen = set(df["label"].iloc[train_idx])
        # every class must be present: LightGBM only emits columns for classes it saw, and a fold whose model
        # cannot represent a class would be scored on an incomparable output space
        if len(train_idx) == 0 or not set(classes) <= seen:
            f.skipped = "training window does not contain every pitch class yet"
        folds.append(f)
    if not any(f.skipped is None for f in folds):
        raise ValueError("No usable rolling development fold")
    return folds


# ------------------------------------------------------------------ fitting
def make_model(family: str, classes: list[str], params: dict, features: list[str] | None = None):
    p = dict(params)
    if family == "lightgbm":
        p.setdefault("n_estimators", LGBM_MAX_ROUNDS)
    return FAMILIES[family](classes, features=features, **p)


def lgbm_checkpoints(max_rounds: int | None = None, step: int | None = None) -> list[int]:
    max_rounds, step = max_rounds or LGBM_MAX_ROUNDS, step or LGBM_CHECKPOINT_STEP
    return list(range(step, max_rounds + 1, step))


def _fold_predictions(df: pd.DataFrame, folds: list[Fold], family: str, classes: list[str], params: dict,
                      features: list[str] | None, checkpoints: list[int] | None) -> dict:
    """Returns {checkpoint_or_None: [P_fold, ...]} for the usable folds."""
    out: dict = {}
    for f in folds:
        if f.skipped:
            continue
        tr, te = df.iloc[f.train_idx], df.iloc[f.eval_idx]
        p = dict(params)
        if family == "lightgbm" and checkpoints:
            p["n_estimators"] = max(checkpoints)
        m = make_model(family, classes, p, features).fit(tr, tr["label"])  # never sees the scored block
        if family == "lightgbm" and checkpoints:
            Xte = m._X(te)
            for k in checkpoints:
                P = np.clip(m.model.predict_proba(Xte, num_iteration=k), 1e-12, None)
                out.setdefault(k, []).append(P / P.sum(1, keepdims=True))
        else:
            out.setdefault(None, []).append(m.predict_proba(te, calibrated=False))
    return out


def chronological_calibration(P_folds: list[np.ndarray], y_folds: list[np.ndarray]) -> tuple[np.ndarray, list]:
    """Calibrate fold f with a temperature fit only on the out-of-sample predictions of folds before f (T=1 for the
    first fold). The result is an honest estimate of what a calibrated deployed model would have produced."""
    out, temps = [], []
    for i, P in enumerate(P_folds):
        if i == 0:
            T = 1.0
        else:
            T = TemperatureCalibrator().fit(np.vstack(P_folds[:i]), np.concatenate(y_folds[:i])).T
        temps.append(float(T))
        out.append(TemperatureCalibrator._apply(np.log(np.clip(P, 1e-12, 1)), T))
    return np.vstack(out), temps


@dataclass
class DevResult:
    family: str
    params: dict
    grid_results: list[dict]
    folds: list[Fold]
    P_raw: np.ndarray
    P_cal: np.ndarray
    y: np.ndarray
    eval_frame: pd.DataFrame
    fold_temperatures: list[float]
    temperature: float  # fit on ALL pooled dev OOS predictions
    summary_raw: dict
    summary_cal: dict
    ece_cal: float
    ece_raw: float
    seconds: float
    extra: dict = field(default_factory=dict)

    @property
    def calibration_applied(self) -> bool:
        """Predetermined rule: deploy the temperature only if chronological calibration lowered the rolling
        development log loss relative to the raw outputs (same folds, same pitches)."""
        return self.summary_cal["log_loss"] < self.summary_raw["log_loss"]

    @property
    def deployed_temperature(self) -> float:
        return float(self.temperature) if self.calibration_applied else 1.0

    @property
    def P_deployed(self) -> np.ndarray:
        return self.P_cal if self.calibration_applied else self.P_raw

    def dev_metrics(self) -> dict:
        """Compact dict stored with the model (lifecycle.dev_metrics / registry validation_metrics).

        The headline keys (log_loss, brier, top1_accuracy, ..., ece) describe the output the deployed model produces
        under the calibration rule: chronologically calibrated predictions if calibration is applied, else raw."""
        s = get_settings()
        head = self.summary_cal if self.calibration_applied else self.summary_raw
        return {**head, "ece": self.ece_cal if self.calibration_applied else self.ece_raw,
                "raw_log_loss": self.summary_raw["log_loss"],
                "raw_brier": self.summary_raw["brier"], "raw_top1_accuracy": self.summary_raw["top1_accuracy"],
                "raw_ece": self.ece_raw, "chrono_calibrated_log_loss": self.summary_cal["log_loss"],
                "chrono_calibrated_brier": self.summary_cal["brier"], "chrono_calibrated_ece": self.ece_cal,
                "evaluation_role": "development",
                "protocol": "rolling_origin_expanding_window",
                "eval_seasons": list(self.extra.get("eval_seasons") or s.dev_eval_seasons),
                "block_games": int(self.extra.get("block_games") or s.dev_block_games),
                "n_folds": sum(f.skipped is None for f in self.folds),
                "calibration": "chronological (fold f uses a temperature fit on folds < f)",
                "calibration_rule": CALIBRATION_RULE, "calibration_applied": bool(self.calibration_applied),
                "pooled_temperature": self.temperature, "deployed_temperature": self.deployed_temperature}


def _stack(folds: list[Fold], df: pd.DataFrame, model_y) -> tuple[list[np.ndarray], pd.DataFrame]:
    ys, parts = [], []
    for f in folds:
        if f.skipped:
            continue
        te = df.iloc[f.eval_idx]
        ys.append(model_y(te["label"]))
        parts.append(te)
    return ys, pd.concat(parts)


def rolling_dev_evaluate(df: pd.DataFrame, classes: list[str], family: str, grid: list[dict] | None = None,
                         features: list[str] | None = None, eval_seasons: list[int] | None = None,
                         block_games: int | None = None, checkpoints: list[int] | None = None,
                         fixed_rounds: bool = False, progress: Callable | None = None) -> DevResult:
    """Score every grid point on the rolling development folds and select the lowest pooled raw log loss.

    `fixed_rounds=True` evaluates LightGBM params as given (their own n_estimators) instead of searching rounds."""
    t0 = time.perf_counter()
    folds = rolling_folds(df, classes, eval_seasons, block_games)
    grid = grid if grid is not None else DEV_GRIDS[family]
    cks = None
    if family == "lightgbm" and not fixed_rounds:
        cks = checkpoints or lgbm_checkpoints()
    idx = {c: i for i, c in enumerate(classes)}
    ys, eval_frame = _stack(folds, df, lambda s: s.map(idx).to_numpy().astype(int))
    y = np.concatenate(ys)
    results, best = [], None
    for gi, params in enumerate(grid):
        preds = _fold_predictions(df, folds, family, classes, params, features, cks)
        for k, Ps in preds.items():
            P = np.vstack(Ps)
            ll = log_loss(P, y)
            if best is None or ll < best[0] - 1e-12:
                best = (ll, {**params, **({"n_estimators": int(k)} if k is not None else {})}, Ps)
        if cks:
            curve = [(int(k), log_loss(np.vstack(preds[k]), y)) for k in cks]
            kbest, llbest = min(curve, key=lambda t: t[1])
            P = np.vstack(preds[kbest])
            s = summarize(P, y, classes)
            results.append({"params": {**params, "n_estimators": kbest}, "val_log_loss": llbest,
                            "val_top1": s["top1_accuracy"], "val_brier": s["brier"],
                            "rounds_curve": [{"n_estimators": k, "log_loss": v} for k, v in curve]})
        else:
            P = np.vstack(preds[None])
            s = summarize(P, y, classes)
            results.append({"params": dict(params), "val_log_loss": s["log_loss"], "val_top1": s["top1_accuracy"],
                            "val_brier": s["brier"]})
        if progress:
            progress((gi + 1) / len(grid), f"{family} dev grid {gi + 1}/{len(grid)}", gi + 1)
    _, params, Ps = best
    P_raw = np.vstack(Ps)
    P_cal, temps = chronological_calibration(Ps, ys)
    T = TemperatureCalibrator().fit(P_raw, y).T
    s_raw, s_cal = summarize(P_raw, y, classes), summarize(P_cal, y, classes)
    res = DevResult(family=family, params=params, grid_results=results, folds=folds, P_raw=P_raw, P_cal=P_cal, y=y,
                    eval_frame=eval_frame, fold_temperatures=temps, temperature=float(T), summary_raw=s_raw,
                    summary_cal=s_cal, ece_cal=reliability(P_cal, y, classes)["ece_top_label"],
                    ece_raw=reliability(P_raw, y, classes)["ece_top_label"], seconds=time.perf_counter() - t0,
                    extra={"eval_seasons": list(eval_seasons or get_settings().dev_eval_seasons),
                           "block_games": int(block_games or get_settings().dev_block_games)})
    log(logger, "rolling_dev_evaluated", family=family, params=params, raw_log_loss=s_raw["log_loss"],
        cal_log_loss=s_cal["log_loss"], folds=sum(f.skipped is None for f in folds), seconds=res.seconds)
    return res


def per_fold_summaries(res: DevResult, classes: list[str]) -> list[dict]:
    out, off, used = [], 0, 0
    for f in res.folds:
        if f.skipped:
            out.append(f.meta())
            continue
        n = len(f.eval_idx)
        s = summarize(res.P_cal[off:off + n], res.y[off:off + n], classes)
        r = summarize(res.P_raw[off:off + n], res.y[off:off + n], classes)
        out.append({**f.meta(), "log_loss": s["log_loss"], "top1_accuracy": s["top1_accuracy"],
                    "brier": s["brier"], "raw_log_loss": r["log_loss"], "raw_top1_accuracy": r["top1_accuracy"],
                    "temperature": res.fold_temperatures[used]})
        off += n
        used += 1
    return out
