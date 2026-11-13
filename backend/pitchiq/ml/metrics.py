"""Proper scoring rules, calibration and breakdowns for multiclass pitch predictions."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

EPS = 1e-15


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def onehot(y_idx: np.ndarray, k: int) -> np.ndarray:
    o = np.zeros((len(y_idx), k))
    o[np.arange(len(y_idx)), y_idx] = 1.0
    return o


def log_loss(P: np.ndarray, y_idx: np.ndarray) -> float:
    return float(-np.mean(np.log(np.clip(P[np.arange(len(y_idx)), y_idx], EPS, 1.0))))


def brier(P: np.ndarray, y_idx: np.ndarray) -> float:
    """Multiclass Brier score: mean over samples of sum_k (p_k - y_k)^2 (range 0..2)."""
    return float(np.mean(np.sum((P - onehot(y_idx, P.shape[1])) ** 2, axis=1)))


def topk_acc(P: np.ndarray, y_idx: np.ndarray, k: int) -> float:
    top = np.argsort(-P, axis=1)[:, :k]
    return float(np.mean([y in t for y, t in zip(y_idx, top)]))


def reliability(P: np.ndarray, y_idx: np.ndarray, classes: list[str], n_bins: int = 10) -> dict:
    """Top-label reliability (for ECE) and per-class one-vs-rest reliability curves."""
    edges = np.linspace(0, 1, n_bins + 1)
    conf = P.max(axis=1)
    correct = (P.argmax(axis=1) == y_idx).astype(float)
    bins, ece = [], 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi) if lo > 0 else (conf >= lo) & (conf <= hi)
        if m.sum():
            gap = abs(conf[m].mean() - correct[m].mean())
            ece += m.mean() * gap
            bins.append({"lo": lo, "hi": hi, "n": int(m.sum()), "mean_conf": float(conf[m].mean()),
                         "accuracy": float(correct[m].mean())})
    per_class = {}
    for j, c in enumerate(classes):
        pts = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (P[:, j] > lo) & (P[:, j] <= hi) if lo > 0 else (P[:, j] >= lo) & (P[:, j] <= hi)
            if m.sum():
                pts.append({"lo": lo, "hi": hi, "n": int(m.sum()), "mean_pred": float(P[m, j].mean()),
                            "observed": float((y_idx[m] == j).mean())})
        per_class[c] = pts
    return {"ece_top_label": float(ece), "top_label_bins": bins, "per_class": per_class}


def confusion(P: np.ndarray, y_idx: np.ndarray, classes: list[str]) -> dict:
    k = len(classes)
    M = np.zeros((k, k), dtype=int)
    for t, p in zip(y_idx, P.argmax(axis=1)):
        M[t, p] += 1
    per = {}
    for j, c in enumerate(classes):
        tp, fp, fn = M[j, j], M[:, j].sum() - M[j, j], M[j, :].sum() - M[j, j]
        prec = tp / (tp + fp) if tp + fp else None
        rec = tp / (tp + fn) if tp + fn else None
        f1 = 2 * prec * rec / (prec + rec) if prec and rec else None
        per[c] = {"support": int(M[j, :].sum()), "predicted": int(M[:, j].sum()), "precision": prec,
                  "recall": rec, "f1": f1}
    return {"labels": classes, "matrix": M.tolist(), "rows": "actual", "cols": "predicted", "per_class": per}


def summarize(P: np.ndarray, y_idx: np.ndarray, classes: list[str]) -> dict:
    n = len(y_idx)
    k1 = int((P.argmax(axis=1) == y_idx).sum())
    lo, hi = wilson(k1, n)
    return {
        "n": n,
        "top1_correct": k1,
        "top1_accuracy": k1 / n if n else None,
        "top1_ci95": [lo, hi],
        "top2_accuracy": topk_acc(P, y_idx, 2) if n else None,
        "log_loss": log_loss(P, y_idx) if n else None,
        "brier": brier(P, y_idx) if n else None,
        "mean_prob_actual": float(P[np.arange(n), y_idx].mean()) if n else None,
    }


def cluster_bootstrap(P: np.ndarray, y_idx: np.ndarray, groups: np.ndarray, n_boot: int = 500, seed: int = 0) -> dict:
    """95% CIs for accuracy / log loss resampling whole games (pitches within a game are correlated)."""
    rng = np.random.default_rng(seed)
    ug = np.unique(groups)
    idx_by = {g: np.where(groups == g)[0] for g in ug}
    acc, ll = [], []
    for _ in range(n_boot):
        idx = np.concatenate([idx_by[g] for g in rng.choice(ug, size=len(ug), replace=True)])
        acc.append(float((P[idx].argmax(1) == y_idx[idx]).mean()))
        ll.append(log_loss(P[idx], y_idx[idx]))
    q = lambda a: [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]  # noqa: E731
    return {"n_boot": n_boot, "unit": "game", "top1_accuracy_ci95": q(acc), "log_loss_ci95": q(ll)}


def pitch_count_bucket(x) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "unknown"
    x = int(x)
    for hi in (25, 50, 75, 100):
        if x < hi:
            return f"{hi - 25}-{hi - 1}"
    return "100+"


def breakdowns(P: np.ndarray, y_idx: np.ndarray, feats: pd.DataFrame, classes: list[str]) -> dict:
    keys = {
        "count": feats["count_str"],
        "batter_hand": feats["stand"],
        "batting_order": feats["bat_order_slot"].astype("Int64").astype(str),
        "inning": feats["inning"].clip(upper=9).astype(int).astype(str).replace({"9": "9+"}),
        "pitch_count": feats["pitch_count_before"].map(pitch_count_bucket),
        "times_through_order": feats["times_through_order"].astype("Int64").astype(str),
        "season": feats["season"].astype(str),
        "actual_pitch": pd.Series([classes[i] for i in y_idx], index=feats.index),
    }
    if "opponent" in feats:
        keys["opponent"] = feats["opponent"]
    out = {}
    pred = P.argmax(axis=1)
    for name, ser in keys.items():
        rows = []
        for val in sorted(ser.dropna().unique(), key=str):
            m = (ser == val).to_numpy()
            n = int(m.sum())
            k = int((pred[m] == y_idx[m]).sum())
            lo, hi = wilson(k, n)
            rows.append({"value": str(val), "n": n, "top1_accuracy": k / n, "ci95": [lo, hi],
                         "log_loss": log_loss(P[m], y_idx[m]), "brier": brier(P[m], y_idx[m])})
        out[name] = rows
    return out
