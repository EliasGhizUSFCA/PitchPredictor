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


def latest_by_family(session: Session, pitcher_id: int) -> dict[str, ModelRegistry]:
    rows = session.scalars(select(ModelRegistry).where(ModelRegistry.pitcher_id == pitcher_id,
                                                       ModelRegistry.status != "archived")
                           .order_by(ModelRegistry.created_at.desc())).all()
    out = {}
    for r in rows:
        if (r.config or {}).get("ablation"):
            continue
        out.setdefault(r.model_family, r)
    return out


def promote(session: Session, version: str) -> None:
    e = session.get(ModelRegistry, version)
    session.execute(update(ModelRegistry).where(ModelRegistry.pitcher_id == e.pitcher_id,
                                                ModelRegistry.status == "promoted").values(status="candidate"))
    e.status = "promoted"
    session.flush()
