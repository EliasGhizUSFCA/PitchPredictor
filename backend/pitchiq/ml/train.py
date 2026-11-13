"""Training with chronological splits. Hyperparameters + calibration use TRAIN and VALIDATION only.

The locked test split is never read here.
"""
from __future__ import annotations

import hashlib
import itertools
import logging
from typing import Callable

import pandas as pd
from sqlalchemy.orm import Session

from ..features.schema import FEATURE_VERSION, schema_dict
from ..features.store import get_classes, load_feature_frame
from ..logging_setup import log
from ..models import ModelRegistry
from . import registry
from .metrics import summarize
from .models import FAMILIES, FrequencyBaseline

logger = logging.getLogger(__name__)

GRIDS: dict[str, list[dict]] = {
    "majority": [{}],
    "frequency": [{"alpha": a, "levels": lv} for a in (5.0, 20.0, 50.0)
                  for lv in (FrequencyBaseline.DEFAULT_LEVELS[:2], FrequencyBaseline.DEFAULT_LEVELS)],
    "logreg": [{"C": c} for c in (0.03, 0.1, 0.3, 1.0)],
    "lightgbm": [{"num_leaves": nl, "min_child_samples": mc}
                 for nl, mc in itertools.product((7, 15, 31), (20, 60, 120))],
}


def fingerprint(df: pd.DataFrame) -> dict:
    h = hashlib.sha256(",".join(map(str, df["pitch_uid"].tolist())).encode()).hexdigest()[:16]
    return {"n": int(len(df)), "pitch_uid_hash": h, "label_counts": df["label"].value_counts().to_dict()}


def train_family(session: Session, pitcher_id: int, family: str, features: list[str] | None = None,
                 tag: str = "", grid: list[dict] | None = None, progress: Callable | None = None,
                 extra_config: dict | None = None) -> ModelRegistry:
    classes = get_classes(session, pitcher_id)
    df = load_feature_frame(session, pitcher_id, ["train", "validation"])
    tr, va = df[df.split == "train"], df[df.split == "validation"]
    cls = FAMILIES[family]
    results, best, best_ll = [], None, float("inf")
    for i, params in enumerate(grid or GRIDS[family]):
        m = cls(classes, features=features, **params).fit(tr, tr["label"], va, va["label"])
        P = m.predict_proba(va, calibrated=False)
        s = summarize(P, m.y_index(va["label"]), classes)
        results.append({"params": {k: v for k, v in params.items()}, "val_log_loss": s["log_loss"],
                        "val_top1": s["top1_accuracy"], "best_iteration": getattr(m, "best_iteration_", None)})
        if s["log_loss"] < best_ll:
            best, best_ll = m, s["log_loss"]
        if progress:
            progress((i + 1) / len(grid or GRIDS[family]), f"{family} grid {i + 1}", i + 1)
    calib = best.calibrate(va, va["label"])
    P = best.predict_proba(va)
    val_metrics = summarize(P, best.y_index(va["label"]), classes)
    version = registry.new_version(pitcher_id, family, FEATURE_VERSION, tag)
    path, sha = registry.save_artifact(best, version)
    entry = ModelRegistry(
        model_version=version, pitcher_id=pitcher_id, model_family=family, feature_version=FEATURE_VERSION,
        feature_schema={**schema_dict(classes), "used_features": best.features}, classes=classes,
        config={**best.config(), "grid_results": results, "selection": "min validation log loss",
                **(extra_config or {})},
        dataset_fingerprint={"train": fingerprint(tr), "validation": fingerprint(va)},
        artifact_path=path, artifact_sha256=sha, calibration=calib, validation_metrics=val_metrics,
        status="candidate",
    )
    session.add(entry)
    session.flush()
    log(logger, "model_trained", version=version, val_log_loss=val_metrics["log_loss"],
        val_top1=val_metrics["top1_accuracy"], temperature=calib["temperature"])
    return entry


def train_all(session: Session, pitcher_id: int, families=("majority", "frequency", "logreg", "lightgbm"),
              progress: Callable | None = None, auto_promote: bool = True) -> list[ModelRegistry]:
    entries = []
    for i, fam in enumerate(families):
        entries.append(train_family(session, pitcher_id, fam))
        if progress:
            progress((i + 1) / len(families), f"Trained {fam}", i + 1)
    if auto_promote and registry.promoted(session, pitcher_id) is None:
        best = min(entries, key=lambda e: e.validation_metrics["log_loss"])
        registry.promote(session, best.model_version)  # chosen on VALIDATION, never test
    return entries
