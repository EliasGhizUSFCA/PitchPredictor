from __future__ import annotations

import time

import pandas as pd
from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.orm import Session

from ...db import get_db
from ...ml import registry
from ...ml.inference import SimState, _clean, _mixture_predict, sim_feature_mixture, summarize_attribution
from ...models import EvaluationRun, ModelRegistry, PipelineTrace, PrePitchState
from ...pitch_types import pitch_name

router = APIRouter(tags=["models"])


def _entry_view(e: ModelRegistry, full: bool = False) -> dict:
    d = {"model_version": e.model_version, "family": e.model_family, "status": e.status,
         "feature_version": e.feature_version, "classes": e.classes, "created_at": str(e.created_at),
         "validation_metrics": e.validation_metrics, "calibration": e.calibration,
         "n_features": len(e.feature_schema.get("used_features", [])),
         "params": (e.config or {}).get("params"), "artifact_sha256": e.artifact_sha256[:16],
         "dataset_fingerprint": e.dataset_fingerprint, "mlflow_run_id": (e.config or {}).get("mlflow_run_id")}
    if full:
        d.update(config=e.config, feature_schema=e.feature_schema, artifact_path=e.artifact_path)
    return d


@router.get("/pitchers/{pid}/models")
def list_models(pid: int, include_archived: bool = False, db: Session = Depends(get_db)):
    q = select(ModelRegistry).where(ModelRegistry.pitcher_id == pid)
    if not include_archived:
        q = q.where(ModelRegistry.status != "archived")
    return [_entry_view(e) for e in db.scalars(q.order_by(ModelRegistry.created_at.desc()))]


@router.get("/models/{version}")
def get_model(version: str, db: Session = Depends(get_db)):
    e = db.get(ModelRegistry, version)
    if e is None:
        raise LookupError("model not found")
    return _entry_view(e, full=True)


@router.post("/models/{version}/promote")
def promote(version: str, db: Session = Depends(get_db)):
    registry.promote(db, version, source="manual: POST /api/models/{version}/promote")
    return {"promoted": version}


@router.get("/pitchers/{pid}/evaluations")
def list_evals(pid: int, kind: str | None = None, db: Session = Depends(get_db)):
    q = select(EvaluationRun).where(EvaluationRun.pitcher_id == pid)
    if kind:
        q = q.where(EvaluationRun.kind == kind)
    out = []
    for r in db.scalars(q.order_by(EvaluationRun.created_at.desc())):
        e = db.get(ModelRegistry, r.model_version)
        out.append({"evaluation_run_id": r.evaluation_run_id, "kind": r.kind, "split": r.split,
                    "model_version": r.model_version, "family": e.model_family if e else None,
                    "model_status": e.status if e else None, "n_samples": r.n_samples,
                    "created_at": str(r.created_at), "summary": r.metrics.get("summary"),
                    "dataset_definition": r.dataset_definition, "mlflow_run_id": r.metrics.get("mlflow_run_id")})
    return out


@router.get("/evaluations/{run_id}")
def get_eval(run_id: int, db: Session = Depends(get_db)):
    r = db.get(EvaluationRun, run_id)
    if r is None:
        raise LookupError("evaluation not found")
    e = db.get(ModelRegistry, r.model_version)
    return {"evaluation_run_id": r.evaluation_run_id, "kind": r.kind, "split": r.split,
            "model_version": r.model_version, "family": e.model_family if e else None, "n_samples": r.n_samples,
            "created_at": str(r.created_at), "dataset_definition": r.dataset_definition, "metrics": r.metrics,
            "mlflow_run_id": r.metrics.get("mlflow_run_id")}


@router.get("/pitchers/{pid}/leaderboard")
def leaderboard(pid: int, split: str = "test", db: Session = Depends(get_db)):
    """Latest holdout benchmark per non-archived model — all on the same held-out pitches."""
    rows, seen = [], set()
    q = (select(EvaluationRun, ModelRegistry).join(ModelRegistry, ModelRegistry.model_version == EvaluationRun.model_version)
         .where(EvaluationRun.pitcher_id == pid, EvaluationRun.kind == "holdout_benchmark", EvaluationRun.split == split,
                ModelRegistry.status != "archived").order_by(EvaluationRun.created_at.desc()))
    for r, e in db.execute(q).all():
        if e.model_version in seen:
            continue
        seen.add(e.model_version)
        m = r.metrics
        rows.append({"model_version": e.model_version, "family": e.model_family, "status": e.status,
                     "evaluation_run_id": r.evaluation_run_id, "n": r.n_samples, **m["summary"],
                     "bootstrap": m.get("bootstrap"), "ece": m["calibration"]["ece_top_label"],
                     "uncalibrated": m.get("uncalibrated_summary"),
                     "inference_ms_per_pitch": m.get("inference_ms_per_pitch"),
                     "validation": e.validation_metrics, "mlflow_run_id": m.get("mlflow_run_id"),
                     "model_mlflow_run_id": (e.config or {}).get("mlflow_run_id")})
    hashes = {r.dataset_definition.get("pitch_uid_min") for r, _ in db.execute(q).all()}
    return {"split": split, "rows": sorted(rows, key=lambda x: x["log_loss"]),
            "same_examples": len(rows) <= 1 or len({x["n"] for x in rows}) == 1,
            "note": "Lower log loss / Brier is better. Overlapping game-bootstrap CIs mean no reliable ranking."}


@router.get("/tracking")
def tracking_status():
    """MLflow mirror status. Postgres remains the source of truth; run ids appear on model / evaluation views."""
    from ...ml.tracking import get_tracker
    return get_tracker().status()


class ComparePitch(BaseModel):
    pitch_uid: int | None = None
    state: SimState | None = None


@router.post("/pitchers/{pid}/models/compare")
def compare_models(pid: int, body: ComparePitch, db: Session = Depends(get_db)):
    """Every model family's prediction for the same situation (historical pitch or simulator state)."""
    out = []
    for fam, e in sorted(registry.latest_by_family(db, pid).items()):
        m = registry.load_model(e)
        if body.pitch_uid is not None:
            feats = db.scalar(select(PrePitchState.features).where(PrePitchState.pitch_uid == body.pitch_uid))
            mix = [(1.0, feats)]
        else:
            mix, _ = sim_feature_mixture(db, e, body.state or SimState())
        t = time.perf_counter()
        P, X = _mixture_predict(m, mix)
        ms = (time.perf_counter() - t) * 1000
        top = e.classes[int(P[0].argmax())]
        ex = m.explain(X.iloc[[0]]) if len(mix) == 1 else None
        out.append({"model_version": e.model_version, "family": fam, "status": e.status, "inference_ms": ms,
                    "predictions": sorted([{"pitch": c, "pitch_name": pitch_name(c), "probability": float(P[0, j])}
                                           for j, c in enumerate(e.classes)], key=lambda d: -d["probability"]),
                    "attribution": summarize_attribution(ex[0], top, e.classes) if ex else None})
    return out


@router.get("/traces")
def traces(limit: int = 30, kind: str | None = None, db: Session = Depends(get_db)):
    q = select(PipelineTrace)
    if kind:
        q = q.where(PipelineTrace.kind == kind)
    return [{"trace_id": t.trace_id, "kind": t.kind, "model_version": t.model_version, "total_ms": t.total_ms,
             "created_at": str(t.created_at), "n_stages": len(t.stages),
             "status": "error" if any(s["status"] == "error" for s in t.stages) else
             "warning" if any(s["status"] == "warning" for s in t.stages) else "ok"}
            for t in db.scalars(q.order_by(PipelineTrace.created_at.desc()).limit(limit))]


@router.get("/traces/{trace_id}")
def trace(trace_id: str, db: Session = Depends(get_db)):
    t = db.get(PipelineTrace, trace_id)
    if t is None:
        raise LookupError("trace not found")
    return {"trace_id": t.trace_id, "kind": t.kind, "model_version": t.model_version, "total_ms": t.total_ms,
            "created_at": str(t.created_at), "stages": t.stages}


@router.get("/explain/narrative")
def narrative(trace_id: str | None = None, prediction_id: int | None = None, db: Session = Depends(get_db)):
    """Deterministic template summary of a stored attribution (no LLM; never generates probabilities)."""
    from ...models import PitchPrediction
    pp = db.get(PitchPrediction, prediction_id) if prediction_id else None
    if pp is None or not pp.attribution:
        raise LookupError("prediction with attribution not found")
    top = pp.top_pitch
    grp = pp.attribution["per_class"][top]["by_group"]
    pos = [g for g, v in grp.items() if v > 0][:3]
    neg = [g for g, v in grp.items() if v < 0][:2]
    text = (f"The model's highest-probability pitch is {pitch_name(top)} ({pp.probabilities[top]:.0%}). "
            f"Feature groups pushing toward it: {', '.join(pos) or 'none'}. "
            f"Groups pushing away: {', '.join(neg) or 'none'}. "
            f"Attribution space: {pp.attribution['space']}.")
    return {"text": text, "generated_by": "template over stored attribution values", "source": _clean(pp.prediction_id)}
