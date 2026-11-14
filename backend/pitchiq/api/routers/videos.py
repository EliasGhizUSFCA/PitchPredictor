from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from sqlalchemy import func, select, text
from sqlalchemy.orm import Session

from ...db import get_db
from ...models import Batter, Pitch, VideoAsset
from ...pitch_types import pitch_name
from ...video.align_service import MediaForbidden, MediaNotAligned, has_pre_pitch_segment, media_for
from ...video.discovery import coverage

router = APIRouter(tags=["videos"])


@router.get("/pitchers/{pid}/videos/coverage")
def cov(pid: int, db: Session = Depends(get_db)):
    return coverage(db, pid)


@router.get("/pitchers/{pid}/videos")
def search(pid: int, game_date: str | None = None, opponent: str | None = None, batter_id: int | None = None,
           stand: str | None = None, inning: int | None = None, count: str | None = None,
           pitch_type: str | None = None, pc_min: int | None = None, pc_max: int | None = None,
           availability: str | None = None, match_confidence: str | None = None, include_labels: bool = True,
           limit: int = Query(50, le=500), offset: int = 0, db: Session = Depends(get_db)):
    q = (select(VideoAsset, Pitch, Batter.full_name).join(Pitch, Pitch.pitch_uid == VideoAsset.pitch_uid)
         .join(Batter, Batter.batter_id == Pitch.batter_id).where(VideoAsset.pitcher_id == pid))
    if game_date:
        q = q.where(Pitch.game_date == game_date)
    if opponent:
        q = q.where(((Pitch.inning_topbot == "Top") & (Pitch.away_team == opponent)) |
                    ((Pitch.inning_topbot == "Bot") & (Pitch.home_team == opponent)))
    if batter_id:
        q = q.where(Pitch.batter_id == batter_id)
    if stand:
        q = q.where(Pitch.stand == stand)
    if inning:
        q = q.where(Pitch.inning == inning)
    if count:
        b, s = count.split("-")
        q = q.where(Pitch.balls == int(b), Pitch.strikes == int(s))
    if pitch_type:
        q = q.where(Pitch.pitch_type == pitch_type)
    if pc_min is not None:
        q = q.where(Pitch.game_pitch_seq - 1 >= pc_min)
    if pc_max is not None:
        q = q.where(Pitch.game_pitch_seq - 1 <= pc_max)
    if availability:
        q = q.where(VideoAsset.video_availability_status == availability)
    if match_confidence:
        q = q.where(VideoAsset.match_confidence == match_confidence)
    total = db.scalar(select(func.count()).select_from(q.subquery()))
    rows = db.execute(q.order_by(Pitch.game_date.desc(), Pitch.game_pitch_seq).limit(limit).offset(offset)).all()
    return {"total": total, "rows": [_view(v, p, name, include_labels) for v, p, name in rows]}


def _view(v: VideoAsset, p: Pitch, batter_name: str | None, include_labels: bool) -> dict:
    d = {"video_asset_id": v.video_asset_id, "pitch_uid": v.pitch_uid, "game_pk": v.game_pk,
         "game_date": str(p.game_date), "opponent": p.away_team if p.inning_topbot == "Top" else p.home_team,
         "batter": batter_name, "stand": p.stand, "inning": p.inning, "count": f"{p.balls}-{p.strikes}",
         "at_bat_number": v.at_bat_number, "pitch_number": v.pitch_number, "game_pitch_seq": p.game_pitch_seq,
         "source_provider": v.source_provider, "match_confidence": v.match_confidence,
         "match_evidence": v.match_evidence, "review_status": v.review_status,
         "playback_method": v.playback_method, "availability": v.video_availability_status,
         "rights": v.rights_or_permission_status, "synchronization_status": v.synchronization_status,
         "discovered_at": str(v.discovered_at), "last_verified_at": str(v.last_verified_at) if v.last_verified_at else None,
         "source_page_url": v.source_page_url,
         "media_url": f"/api/videos/{v.video_asset_id}/media" if v.playback_method == "licensed_local" else None,
         # media_url serves the pre-pitch segment only; the full clip needs ?segment=full after a reveal
         "full_media_url": f"/api/videos/{v.video_asset_id}/media?segment=full"
         if v.playback_method == "licensed_local" else None,
         "pre_pitch_segment_available": has_pre_pitch_segment(v), "alignment_confidence": v.alignment_confidence,
         "pre_pitch_pause_time": v.pre_pitch_pause_time}
    if include_labels:
        d.update(pitch_type=p.pitch_type, pitch_name=pitch_name(p.pitch_type), result=p.description)
    return d


@router.get("/pitchers/{pid}/videos/random")
def random_clip(pid: int, split: str = "test", db: Session = Depends(get_db)):
    """Random eligible (verified) clip. Returns only the pitch id so it can be used blind."""
    row = db.execute(text("""SELECT v.video_asset_id, v.pitch_uid FROM video_assets v JOIN pre_pitch_states s USING (pitch_uid)
        WHERE v.pitcher_id=:p AND v.match_confidence='exact_verified' AND s.split=:sp AND s.label IS NOT NULL
        AND v.video_availability_status <> 'unavailable' ORDER BY random() LIMIT 1"""), {"p": pid, "sp": split}).first()
    if row is None:
        raise LookupError("no eligible clips")
    return {"video_asset_id": row[0], "pitch_uid": row[1]}


class ReviewIn(BaseModel):
    decision: str = Field(pattern="^(approved|rejected)$")
    notes: str | None = None


@router.post("/videos/{vid}/review")
def review(vid: int, body: ReviewIn, db: Session = Depends(get_db)):
    v = db.get(VideoAsset, vid)
    if v is None:
        raise LookupError("video asset not found")
    v.review_status = body.decision
    v.verification_notes = body.notes
    if body.decision == "approved":
        v.match_confidence = "exact_verified"
        v.match_evidence = {**(v.match_evidence or {}), "manual_review": "approved"}
    else:
        v.match_confidence = "no_match"
        v.pitch_uid = None
    v.last_verified_at = datetime.now(timezone.utc)
    return {"ok": True}


class AlignmentIn(BaseModel):
    clip_start_time: float | None = None
    pre_pitch_pause_time: float
    release_frame_timestamp: float | None = None
    pitch_end_time: float | None = None


@router.put("/videos/{vid}/alignment")
def set_alignment(vid: int, body: AlignmentIn, db: Session = Depends(get_db)):
    """User correction of pre-pitch pause point (only meaningful for controllable, licensed media)."""
    v = db.get(VideoAsset, vid)
    if v is None:
        raise LookupError("video asset not found")
    if v.playback_method != "licensed_local":
        raise ValueError("Alignment applies only to controllable licensed media; external clips open after reveal")
    for k, val in body.model_dump().items():
        setattr(v, k, val)
    v.synchronization_status, v.alignment_confidence = "user_corrected", 1.0
    return {"ok": True}


@router.get("/videos/{vid}/media")
def media(vid: int, segment: str = Query("pre_pitch", pattern="^(pre_pitch|full)$"), db: Session = Depends(get_db)):
    """Licensed local media with server-side spoiler protection.

    Default: the FFmpeg-trimmed pre-pitch segment [0, pre_pitch_pause_time) (409 if the clip is unaligned).
    segment=full: the whole clip, only after a prediction for this pitch was revealed (403 before)."""
    v = db.get(VideoAsset, vid)
    try:
        p = media_for(db, v, segment)
    except MediaForbidden as e:
        raise HTTPException(status_code=403, detail=str(e)) from e
    except MediaNotAligned as e:
        raise HTTPException(status_code=409, detail=str(e)) from e
    return FileResponse(Path(p), media_type="video/mp4", headers={"Cache-Control": "private, no-store"})
