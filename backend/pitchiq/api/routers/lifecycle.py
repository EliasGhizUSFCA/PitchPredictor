"""Model lifecycle API: evaluation protocol, gates, rollback, development evaluation, policy replay, prospective."""
from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.orm import Session

from ...db import get_db
from ...ml import lifecycle
from ...ml.prospective import prospective_metrics
from ...ml.protocol import protocol_dict
from ...models import EvaluationRun, ModelRegistry

router = APIRouter(tags=["lifecycle"])


@router.get("/protocol")
def protocol():
    """Evaluation roles (development 2023-2025, retrospective_viewed 2026), rolling protocol, gate thresholds."""
    return protocol_dict()


@router.get("/pitchers/{pid}/lifecycle")
def overview(pid: int, db: Session = Depends(get_db)):
    return lifecycle.overview(db, pid)


@router.post("/models/{version}/gates")
def gates(version: str, db: Session = Depends(get_db)):
    """Evaluate and record the promotion gates (no status change)."""
    return lifecycle.check_gates(db, version)


@router.post("/models/{version}/rollback")
def rollback(version: str, db: Session = Depends(get_db)):
    """`version` must be the production model; the previous_production model becomes production again."""
    return lifecycle.rollback(db, version, source="manual: POST /api/models/{version}/rollback")


def _run_view(r: EvaluationRun, e: ModelRegistry | None) -> dict:
    m = r.metrics or {}
    return {"evaluation_run_id": r.evaluation_run_id, "kind": r.kind, "split": r.split,
            "model_version": r.model_version, "family": e.model_family if e else None,
            "model_status": e.status if e else None, "n_samples": r.n_samples, "created_at": str(r.created_at),
            "evaluation_role": m.get("evaluation_role"), "evaluation_label": m.get("evaluation_label"),
            "mlflow_run_id": m.get("mlflow_run_id")}


@router.get("/pitchers/{pid}/dev-evaluation")
def dev_evaluation(pid: int, db: Session = Depends(get_db)):
    """Latest rolling-origin development evaluation per model (selection objective + per-class metrics)."""
    rows = db.execute(select(EvaluationRun, ModelRegistry)
                      .join(ModelRegistry, ModelRegistry.model_version == EvaluationRun.model_version)
                      .where(EvaluationRun.pitcher_id == pid, EvaluationRun.kind == "rolling_dev_evaluation")
                      .order_by(EvaluationRun.created_at.desc())).all()
    out, seen = [], set()
    for r, e in rows:
        if e.model_version in seen:
            continue
        seen.add(e.model_version)
        m = r.metrics
        out.append({**_run_view(r, e), "dev_metrics": m.get("dev_metrics"), "summary": m.get("summary"),
                    "uncalibrated_summary": m.get("uncalibrated_summary"), "selected_params": m.get("selected_params"),
                    "per_class": (m.get("confusion") or {}).get("per_class"), "recipe": m.get("recipe"),
                    "grid_size": len(m.get("grid_results") or []), "folds": m.get("folds")})
    return {"protocol": protocol_dict()["rolling_development"], "rows": sorted(
        out, key=lambda x: (x["dev_metrics"] or {}).get("log_loss", 9))}


@router.get("/pitchers/{pid}/retraining-replay")
def retraining_replay(pid: int, db: Session = Depends(get_db)):
    rows = db.execute(select(EvaluationRun, ModelRegistry)
                      .join(ModelRegistry, ModelRegistry.model_version == EvaluationRun.model_version)
                      .where(EvaluationRun.pitcher_id == pid, EvaluationRun.kind == "retraining_policy_replay")
                      .order_by(EvaluationRun.created_at.desc())).all()
    out, seen = [], set()
    for r, e in rows:
        key = (r.metrics.get("family"), r.metrics.get("season"))
        if key in seen:
            continue
        seen.add(key)
        pol = {p: {k: d.get(k) for k in ("n", "log_loss", "brier", "top1_accuracy", "top2_accuracy", "ece",
                                         "n_refits", "n_fits", "bootstrap")}
               for p, d in r.metrics["policies"].items()}
        out.append({**_run_view(r, e), "season": r.metrics.get("season"), "policies": pol,
                    "lowest_log_loss_policy": r.metrics.get("lowest_log_loss_policy"), "note": r.metrics.get("note")})
    return {"rows": sorted(out, key=lambda x: (x["season"], x["family"] or ""))}


@router.get("/pitchers/{pid}/prospective")
def prospective(pid: int, db: Session = Depends(get_db)):
    return prospective_metrics(db, pid)
