"""File-based artifact store + `model_registry` table."""
from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

import joblib
from sqlalchemy import select, update
from sqlalchemy.orm import Session

from ..config import get_settings
from ..models import ModelRegistry


def save_artifact(model, version: str) -> tuple[str, str]:
    d = get_settings().artifact_dir
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{version}.joblib"
    joblib.dump(model, path)
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    return str(path), sha


@lru_cache(maxsize=32)
def _load(path: str, sha: str):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Model artifact missing: {path}")
    actual = hashlib.sha256(p.read_bytes()).hexdigest()
    if actual != sha:
        raise RuntimeError(f"Artifact checksum mismatch for {path}")
    return joblib.load(p)


def load_model(entry: ModelRegistry):
    return _load(entry.artifact_path, entry.artifact_sha256)


def new_version(pitcher_id: int, family: str, feature_version: str, tag: str = "") -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    return f"{pitcher_id}-{family}-{feature_version}-{ts}{('-' + tag) if tag else ''}"


def promoted(session: Session, pitcher_id: int) -> ModelRegistry | None:
    return session.scalars(select(ModelRegistry).where(ModelRegistry.pitcher_id == pitcher_id,
                                                       ModelRegistry.status == "promoted")
                           .order_by(ModelRegistry.created_at.desc())).first()


def get_entry(session: Session, pitcher_id: int, version: str | None) -> ModelRegistry:
    if version:
        e = session.get(ModelRegistry, version)
        if e is None:
            raise KeyError(f"Unknown model version {version}")
        return e
    e = promoted(session, pitcher_id)
    if e is None:
        raise LookupError("No promoted model. Train models first (pitchiq train).")
    return e


def latest_by_family(session: Session, pitcher_id: int, include_lifecycle: bool = False) -> dict[str, ModelRegistry]:
    """Latest non-archived season-split model per family (benchmarks, walk-forward, ablation, comparisons).

    Lifecycle models (deployment refits / scheduled retrains, see ml/refit.py) are trained on every season up to the
    deployment cutoff, so they are excluded unless `include_lifecycle`: scoring them on a held-out split would be
    in-sample."""
    rows = session.scalars(select(ModelRegistry).where(ModelRegistry.pitcher_id == pitcher_id,
                                                       ModelRegistry.status != "archived")
                           .order_by(ModelRegistry.created_at.desc())).all()
    out = {}
    for r in rows:
        if (r.config or {}).get("ablation"):
            continue
        if not include_lifecycle and (r.config or {}).get("training_scope") in ("deployment_refit",
                                                                                 "scheduled_retrain"):
            continue
        out.setdefault(r.model_family, r)
    return out


def promote(session: Session, version: str, source: str = "manual") -> None:
    """Make `version` the pitcher's promoted model.

    `source` records how the decision was made (e.g. `train_all` picks the lowest *validation* log loss; `manual`
    means a person chose it and nothing here checks their criterion). It is stored in
    `config["promotion_source"]` and mirrored to MLflow."""
    e = session.get(ModelRegistry, version)
    if e is None:
        raise KeyError(f"Unknown model version {version}")
    demoted = session.scalars(select(ModelRegistry).where(ModelRegistry.pitcher_id == e.pitcher_id,
                                                          ModelRegistry.status.in_(("promoted", "previous_production")),
                                                          ModelRegistry.model_version != version)).all()
    # One rollback slot: the outgoing production model becomes previous_production (see ml/lifecycle.rollback);
    # the older previous_production returns to candidate. Nothing is deleted.
    session.execute(update(ModelRegistry).where(ModelRegistry.pitcher_id == e.pitcher_id,
                                                ModelRegistry.status == "previous_production",
                                                ModelRegistry.model_version != version).values(status="candidate"))
    session.execute(update(ModelRegistry).where(ModelRegistry.pitcher_id == e.pitcher_id,
                                                ModelRegistry.status == "promoted",
                                                ModelRegistry.model_version != version)
                    .values(status="previous_production"))
    e.status = "promoted"
    e.config = {**(e.config or {}), "promotion_source": source,
                "promoted_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    session.flush()
    for d in demoted:
        session.refresh(d)
    from . import tracking  # optional MLflow mirror: status tags + registry alias (no-op if disabled)
    tracking.get_tracker().sync_status([*demoted, e])
