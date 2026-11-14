"""Next-pitch feature v2 API: feature schema per version, as-of MLB batter priors, v2 ablation runs."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from ...db import get_db
from ...features.schema import FEATURE_VERSION, FEATURE_VERSIONS, default_model_features, schema_dict
from ...models import EvaluationRun, Pitcher
from ...models_batter_priors import BatterPriorAsOf, BatterPriorMeta

router = APIRouter(tags=["features"])


@router.get("/features/schema")
def feature_schema(version: str = FEATURE_VERSION, pitcher_id: int | None = None, db: Session = Depends(get_db)):
    """Feature definitions (name, kind, group, description) of a feature version, for a pitcher's classes."""
    if version not in FEATURE_VERSIONS:
        raise HTTPException(404, f"unknown feature version {version}; known {list(FEATURE_VERSIONS)}")
    from ...config import get_settings
    p = db.get(Pitcher, pitcher_id or get_settings().default_pitcher_id)
    fs = ((p.extra or {}).get("feature_sets") or {}) if p is not None else {}
    classes = (fs.get(version) or next(iter(fs.values()), {})).get("classes") or ["FF", "SL", "CH", "SI"]
    return {**schema_dict(classes, version), "current_version": FEATURE_VERSION,
            "versions": list(FEATURE_VERSIONS), "model_inputs_default": default_model_features(classes, version)}


@router.get("/batters/{batter_id}/priors")
def batter_priors(batter_id: int, season: int, throws: str = "L", db: Session = Depends(get_db)):
    """MLB-wide empirical-Bayes priors used for pitches in `season` (built only from seasons before it)."""
    meta = db.get(BatterPriorMeta, (season, throws))
    if meta is None:
        raise HTTPException(404, f"no batter priors built for as-of season {season} vs {throws}HP")
    row = db.get(BatterPriorAsOf, (batter_id, season, throws))
    feats = row.features if row is not None else {**meta.league_means, "blb_missing": 1, "blb_log_pa": 0.0}
    return {"batter_id": batter_id, "as_of_season": season, "pitcher_throws": throws,
            "source_seasons": meta.source_seasons, "features": feats,
            "n_prior_pa": row.n_prior_pa if row is not None else 0.0, "league_means": meta.league_means,
            "shrinkage": {k: v.get("k") for k, v in (meta.shrinkage or {}).items()}, "method": meta.method,
            "note": "Batter without a prior-season MLB line: league means, blb_missing = 1." if row is None else None}


@router.get("/pitchers/{pid}/feature-ablation")
def feature_ablation(pid: int, db: Session = Depends(get_db)):
    """Latest v2 feature-group / decay ablation per family (rolling development evaluation)."""
    runs = db.scalars(select(EvaluationRun).where(EvaluationRun.pitcher_id == pid,
                                                  EvaluationRun.kind == "feature_ablation_v2")
                      .order_by(EvaluationRun.created_at.desc())).all()
    out = {}
    for r in runs:
        fam = (r.metrics or {}).get("family")
        if fam in out:
            continue
        out[fam] = {"evaluation_run_id": r.evaluation_run_id, "model_version": r.model_version,
                    "created_at": str(r.created_at), "n_samples": r.n_samples,
                    "evaluation_role": "development", **(r.metrics or {})}
    return {"pitcher_id": pid, "families": out}
