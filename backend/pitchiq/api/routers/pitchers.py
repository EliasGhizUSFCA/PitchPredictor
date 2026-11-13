from __future__ import annotations

from fastapi import APIRouter, Depends, Query
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from ...analytics import filter_options, pitcher_analytics
from ...challenge import pre_pitch_situation, reveal_payload
from ...db import get_db
from ...ml import registry
from ...ml.inference import predict_historical
from ...models import (Batter, DataValidationReport, EvaluationRun, Game, Pitch, PitchPrediction, Pitcher,
                       PrePitchState)
from ...pitch_types import pitch_name
from ...video.discovery import coverage

router = APIRouter(tags=["pitchers"])


@router.get("/pitchers")
def list_pitchers(db: Session = Depends(get_db)):
    out = []
    for p in db.scalars(select(Pitcher).order_by(Pitcher.is_active_target.desc(), Pitcher.full_name)):
        n = db.scalar(select(func.count()).select_from(Pitch).where(Pitch.pitcher_id == p.pitcher_id))
        out.append({"pitcher_id": p.pitcher_id, "full_name": p.full_name, "throws": p.throws, "seasons": p.seasons,
                    "pitches": n, "is_active_target": p.is_active_target})
    return out


@router.get("/pitchers/{pid}/summary")
def summary(pid: int, db: Session = Depends(get_db)):
    p = db.get(Pitcher, pid)
    if p is None:
        raise LookupError("pitcher not found")
    promoted = registry.promoted(db, pid)
    bench = None
    if promoted:
        run = db.scalars(select(EvaluationRun).where(EvaluationRun.model_version == promoted.model_version,
                                                     EvaluationRun.kind == "holdout_benchmark")
                         .order_by(EvaluationRun.created_at.desc())).first()
        if run:
            bench = {"evaluation_run_id": run.evaluation_run_id, "split": run.split, **run.metrics["summary"],
                     "bootstrap": run.metrics.get("bootstrap")}
    splits = dict(db.execute(select(PrePitchState.split, func.count()).where(
        PrePitchState.pitcher_id == pid, PrePitchState.label.is_not(None)).group_by(PrePitchState.split)).all())
    val = db.scalars(select(DataValidationReport).where(DataValidationReport.pitcher_id == pid)
                     .order_by(DataValidationReport.created_at.desc())).first()
    return {
        "pitcher": {"pitcher_id": p.pitcher_id, "full_name": p.full_name, "throws": p.throws, "seasons": p.seasons},
        "pitches": db.scalar(select(func.count()).select_from(Pitch).where(Pitch.pitcher_id == pid)),
        "games": db.scalar(select(func.count(func.distinct(Pitch.game_pk))).where(Pitch.pitcher_id == pid)),
        "date_range": [str(x) for x in db.execute(select(func.min(Pitch.game_date), func.max(Pitch.game_date))
                                                  .where(Pitch.pitcher_id == pid)).one()],
        "splits": splits,
        "classes": (p.extra or {}).get("feature_sets", {}).get("v1", {}).get("classes"),
        "promoted_model": None if not promoted else {
            "model_version": promoted.model_version, "family": promoted.model_family,
            "validation": promoted.validation_metrics, "calibration": promoted.calibration},
        "official_benchmark": bench,
        "video_coverage": coverage(db, pid),
        "validation": None if not val else {"created_at": str(val.created_at), **val.summary},
    }


@router.get("/pitchers/{pid}/analytics")
def analytics(pid: int, season: int | None = None, batter_id: int | None = None, count: str | None = None,
              game_pk: int | None = None, stand: str | None = None, db: Session = Depends(get_db)):
    return pitcher_analytics(db, pid, {"season": season, "batter_id": batter_id, "count": count,
                                       "game_pk": game_pk, "stand": stand})


@router.get("/pitchers/{pid}/filters")
def filters(pid: int, db: Session = Depends(get_db)):
    return filter_options(db, pid)


@router.get("/pitchers/{pid}/games")
def games(pid: int, split: str | None = None, db: Session = Depends(get_db)):
    q = (select(Game.game_pk, Game.game_date, Game.home_team, Game.away_team, func.count(Pitch.pitch_uid),
                func.min(PrePitchState.split))
         .join(Pitch, Pitch.game_pk == Game.game_pk).join(PrePitchState, PrePitchState.pitch_uid == Pitch.pitch_uid)
         .where(Pitch.pitcher_id == pid).group_by(Game.game_pk).order_by(Game.game_date.desc()))
    rows = db.execute(q).all()
    return [{"game_pk": r[0], "game_date": str(r[1]), "home_team": r[2], "away_team": r[3], "pitches": r[4],
             "split": r[5]} for r in rows if split is None or r[5] == split]


@router.get("/pitchers/{pid}/games/{game_pk}/pitches")
def game_pitches(pid: int, game_pk: int, db: Session = Depends(get_db)):
    """Navigation list WITHOUT pitch types/outcomes (so the replay stays blind)."""
    rows = db.execute(select(Pitch.pitch_uid, Pitch.game_pitch_seq, Pitch.at_bat_number, Pitch.pitch_number,
                             Pitch.inning, Pitch.inning_topbot, Pitch.balls, Pitch.strikes, Pitch.outs_when_up,
                             Batter.full_name, Pitch.stand, Pitch.pitch_type.is_(None).label("auto"))
                      .join(Batter, Batter.batter_id == Pitch.batter_id)
                      .where(Pitch.pitcher_id == pid, Pitch.game_pk == game_pk).order_by(Pitch.game_pitch_seq)).all()
    return [{"pitch_uid": r[0], "game_pitch_seq": r[1], "at_bat_number": r[2], "pitch_number": r[3], "inning": r[4],
             "half": r[5], "count": f"{r[6]}-{r[7]}", "outs": r[8], "batter": r[9], "stand": r[10],
             "automatic_call": bool(r[11])} for r in rows]


@router.get("/pitches/{uid}/situation")
def situation(uid: int, db: Session = Depends(get_db)):
    return pre_pitch_situation(db, uid)


@router.post("/pitchers/{pid}/pitches/{uid}/predict")
def predict(pid: int, uid: int, model_version: str | None = None, db: Session = Depends(get_db)):
    """Blind prediction for a historical pitch (dashboard replay). Frozen before any reveal."""
    res = predict_historical(db, pid, uid, model_version, context="dashboard", save=True)
    return {**res, "situation": pre_pitch_situation(db, uid)}


@router.post("/predictions/{prediction_id}/reveal")
def reveal_prediction(prediction_id: int, db: Session = Depends(get_db)):
    from ...challenge import best_video
    pp = db.get(PitchPrediction, prediction_id)
    if pp is None or pp.pitch_uid is None:
        raise LookupError("prediction not found")
    label = db.scalar(select(PrePitchState.label).where(PrePitchState.pitch_uid == pp.pitch_uid))
    if pp.actual_pitch is None:
        pp.actual_pitch, pp.is_correct_top1 = label, pp.top_pitch == label
        pp.prob_actual = float(pp.probabilities.get(label, 0.0))
        from datetime import datetime, timezone
        pp.revealed_at = datetime.now(timezone.utc)
    v = best_video(db, pp.pitch_uid)

    class _A:  # minimal adapter for reveal_payload
        pitch_uid, prediction_id, video_asset_id, user_guess = pp.pitch_uid, pp.prediction_id, (v.video_asset_id if v else None), None
    return reveal_payload(db, _A)


@router.get("/pitchers/{pid}/validation")
def validation(pid: int, db: Session = Depends(get_db)):
    r = db.scalars(select(DataValidationReport).where(DataValidationReport.pitcher_id == pid)
                   .order_by(DataValidationReport.created_at.desc())).first()
    return None if r is None else {"created_at": str(r.created_at), "summary": r.summary, "issues": r.issues[:200]}


@router.get("/pitch-types")
def pitch_types():
    from ...pitch_types import PITCH_TYPE_NAMES
    return PITCH_TYPE_NAMES


@router.get("/pitchers/{pid}/predictions")
def prediction_history(pid: int, context: str | None = None, limit: int = Query(50, le=500),
                       db: Session = Depends(get_db)):
    q = select(PitchPrediction).join(Pitch, Pitch.pitch_uid == PitchPrediction.pitch_uid).where(
        Pitch.pitcher_id == pid, PitchPrediction.context != "benchmark")
    if context:
        q = q.where(PitchPrediction.context == context)
    rows = db.scalars(q.order_by(PitchPrediction.created_at.desc()).limit(limit)).all()
    return [{"prediction_id": r.prediction_id, "pitch_uid": r.pitch_uid, "context": r.context,
             "model_version": r.model_version, "top_pitch": r.top_pitch, "top_pitch_name": pitch_name(r.top_pitch),
             "probabilities": r.probabilities, "actual_pitch": r.actual_pitch, "is_correct_top1": r.is_correct_top1,
             "prob_actual": r.prob_actual, "created_at": str(r.created_at), "inference_ms": r.inference_ms}
            for r in rows]
