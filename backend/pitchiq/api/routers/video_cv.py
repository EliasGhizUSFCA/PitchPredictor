"""Automatic video alignment (Mode A, licensed_local only): run, inspect, validation reports, attempt media."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from sqlalchemy import select
from sqlalchemy.orm import Session

from ...db import get_db
from ...models import ChallengeAttempt, VideoAsset
from ...models_video_cv import VideoAlignmentRun, VideoAlignmentValidation
from ...video import align_service as svc

router = APIRouter(tags=["video-cv"])

NO_STORE = {"Cache-Control": "private, no-store"}


def _run_view(r: VideoAlignmentRun, full: bool = False) -> dict:
    d = {"alignment_run_id": r.alignment_run_id, "video_asset_id": r.video_asset_id,
         "detector_version": r.detector_version, "status": r.status, "applied": r.applied,
         "alignment_confidence": r.alignment_confidence, "clip_start_time": r.clip_start_time,
         "onset_time": r.onset_time, "release_time": r.release_time, "pre_pitch_pause_time": r.pre_pitch_pause_time,
         "pitch_end_time": r.pitch_end_time, "clip_duration": r.clip_duration, "fps": r.fps, "n_frames": r.n_frames,
         "error": r.error, "runtime_ms": r.runtime_ms, "created_at": str(r.created_at),
         "reasons": (r.result or {}).get("reasons"),
         "confidence_components": (r.result or {}).get("confidence_components"),
         "scene_cuts": (r.result or {}).get("scene_cuts"), "segments": (r.result or {}).get("segments"),
         "scoreboard_region": (r.result or {}).get("scoreboard_region")}
    if full:
        d.update(params=r.params, roi=(r.result or {}).get("roi"), motion=(r.result or {}).get("motion"))
    return d


@router.post("/videos/{vid}/align")
def align_one(vid: int, force: bool = False, db: Session = Depends(get_db)):
    """Run the detector on one licensed_local clip (force=True also replaces a user correction)."""
    v = db.get(VideoAsset, vid)
    if v is None:
        raise LookupError("video asset not found")
    run = svc.align_asset(db, v, force=force)
    return {**_run_view(run), "synchronization_status": v.synchronization_status}


@router.get("/videos/{vid}/alignment")
def alignment(vid: int, db: Session = Depends(get_db)):
    """Current alignment of an asset plus the latest detector run's diagnostics (licensed_local only).

    Exposes timestamps and diagnostics, never the pitch type or outcome."""
    v = db.get(VideoAsset, vid)
    if v is None:
        raise LookupError("video asset not found")
    if v.playback_method != "licensed_local":
        raise ValueError("Alignment applies only to controllable licensed media")
    r = svc.latest_run(db, vid)
    return {"video_asset_id": vid, "synchronization_status": v.synchronization_status,
            "alignment_confidence": v.alignment_confidence, "clip_start_time": v.clip_start_time,
            "pre_pitch_pause_time": v.pre_pitch_pause_time, "release_frame_timestamp": v.release_frame_timestamp,
            "pitch_end_time": v.pitch_end_time, "clip_duration": v.clip_duration,
            "pre_pitch_segment_available": svc.has_pre_pitch_segment(v),
            "latest_run": _run_view(r, full=True) if r else None}


@router.get("/videos/alignment/validations")
def validations(limit: int = 10, db: Session = Depends(get_db)):
    """Detector accuracy reports. Every report so far is on SYNTHETIC clips (data_kind='synthetic')."""
    rows = db.scalars(select(VideoAlignmentValidation).order_by(VideoAlignmentValidation.validation_id.desc())
                      .limit(min(limit, 100))).all()
    return [{"validation_id": r.validation_id, "detector_version": r.detector_version, "role": r.role,
             "data_kind": r.data_kind, "generator": r.generator, "n_clips": r.n_clips, "seeds": r.seeds,
             "summary": r.summary, "mlflow_run_id": r.mlflow_run_id, "created_at": str(r.created_at)} for r in rows]


@router.get("/attempts/{aid}/pre-pitch-media")
def attempt_pre_pitch_media(aid: int, db: Session = Depends(get_db)):
    """Pre-pitch segment for a challenge attempt (in_app mode). Attempt-scoped so the video asset id (which other
    endpoints can tie to the pitch type) is not exposed before reveal. Never serves frames after the pause."""
    att = db.get(ChallengeAttempt, aid)
    if att is None:
        raise LookupError("attempt not found")
    v = db.get(VideoAsset, att.video_asset_id) if att.video_asset_id else None
    try:
        path = svc.media_for(db, v, "pre_pitch")
    except svc.MediaNotAligned as e:
        raise HTTPException(status_code=409, detail=str(e)) from e
    return FileResponse(path, media_type="video/mp4", headers=NO_STORE)
