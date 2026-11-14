"""Apply automatic alignment to licensed_local assets and serve media with server-side spoiler protection.

Rules (Mode A only; external/official links are never downloaded, trimmed or proxied):
* `align_asset` runs the detector on the operator-supplied local file and records a `video_alignment_runs` row.
  A confident result sets synchronization_status='auto_aligned' and the timestamps; otherwise 'unaligned' and no
  pre-pitch timestamps (the clip then only plays in full after reveal). A user correction (PUT
  /api/videos/{id}/alignment, status 'user_corrected') is never overwritten unless force=True.
* `media_for` decides what GET /api/videos/{id}/media may return: by default the FFmpeg-trimmed pre-pitch segment
  [0, pre_pitch_pause_time) (cached on disk, video only); the full clip only with segment=full AND after a
  prediction for that pitch has been revealed (challenge attempt or dashboard replay). There are no user accounts in
  PitchIQ, so the reveal gate is per pitch, not per viewer.
"""
from __future__ import annotations

import hashlib
import logging
import math
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Callable

from sqlalchemy import exists, select
from sqlalchemy.orm import Session

from ..config import get_settings
from ..logging_setup import log
from ..models import ChallengeAttempt, PitchPrediction, VideoAsset
from ..models_video_cv import VideoAlignmentRun
from .alignment import DETECTOR_VERSION, AlignmentParams, align_file, ffprobe

logger = logging.getLogger(__name__)

ALIGNED = ("auto_aligned", "user_corrected")
REVEAL_CONTEXTS = ("challenge", "dashboard")  # user-facing reveals; ledger/benchmark rows do not unlock video


class MediaForbidden(Exception):
    """Full clip requested before any prediction for the pitch was revealed (HTTP 403)."""


class MediaNotAligned(Exception):
    """No pre-pitch segment exists because the clip is not aligned (HTTP 409)."""


def is_licensed_local(a: VideoAsset) -> bool:
    return (a.playback_method == "licensed_local" and bool(a.permitted_media_reference)
            and (a.rights_or_permission_status or "").startswith("licensed:")
            and "://" not in a.permitted_media_reference)


def file_fingerprint(path: str | Path) -> str:
    st = Path(path).stat()
    return hashlib.sha1(f"{Path(path).resolve()}|{st.st_size}|{st.st_mtime_ns}".encode()).hexdigest()


# ------------------------------------------------------------------ alignment
def align_asset(session: Session, asset: VideoAsset, params: AlignmentParams | None = None,
                force: bool = False) -> VideoAlignmentRun:
    if not is_licensed_local(asset):
        raise ValueError("Automatic alignment applies only to licensed_local media with a licence reference")
    path = Path(asset.permitted_media_reference)
    if not path.is_file():
        raise LookupError("licensed file missing")
    p = params or AlignmentParams()
    t0 = time.perf_counter()
    res = align_file(path, p)
    runtime = (time.perf_counter() - t0) * 1000
    keep_user = asset.synchronization_status == "user_corrected" and not force
    run = VideoAlignmentRun(
        video_asset_id=asset.video_asset_id, detector_version=DETECTOR_VERSION, params=p.as_dict(), status=res.status,
        applied=not keep_user, alignment_confidence=res.alignment_confidence, clip_start_time=res.clip_start_time,
        onset_time=res.onset_time, release_time=res.release_time, pre_pitch_pause_time=res.pre_pitch_pause_time,
        pitch_end_time=res.pitch_end_time, clip_duration=res.clip_duration, fps=res.fps, n_frames=res.n_frames,
        file_fingerprint=file_fingerprint(path), result=res.to_dict(), error=res.error, runtime_ms=round(runtime, 1))
    session.add(run)
    if not keep_user:
        asset.clip_duration = res.clip_duration
        asset.alignment_confidence = res.alignment_confidence
        asset.synchronization_status = res.status
        if res.status == "auto_aligned":
            asset.clip_start_time = res.clip_start_time
            asset.pre_pitch_pause_time = res.pre_pitch_pause_time
            asset.release_frame_timestamp = res.release_time
            asset.pitch_end_time = res.pitch_end_time
        else:  # no trustworthy pause point -> no pre-pitch segment
            asset.clip_start_time = asset.pre_pitch_pause_time = None
            asset.release_frame_timestamp = asset.pitch_end_time = None
        asset.verification_notes = ((asset.verification_notes or "").split(" | auto-align:")[0]
                                    + f" | auto-align: {res.status} ({DETECTOR_VERSION}, conf "
                                    f"{res.alignment_confidence:.2f}"
                                    + (f"; {'; '.join(res.reasons)}" if res.reasons else "") + ")")[:2000]
    session.flush()
    if not keep_user and res.status == "auto_aligned":
        try:
            pre_pitch_segment(asset)
        except Exception as e:  # noqa: BLE001 - generated lazily on request otherwise
            log(logger, "pre_pitch_segment_failed", video_asset_id=asset.video_asset_id, error=str(e)[:300])
    return run


def align_videos(session: Session, pitcher_id: int | None = None, force: bool = False,
                 asset_ids: list[int] | None = None, limit: int | None = None,
                 params: AlignmentParams | None = None, progress: Callable | None = None,
                 should_cancel: Callable[[], bool] | None = None) -> dict:
    """Align every licensed_local asset (of a pitcher) that has no current-detector run for its current file."""
    q = select(VideoAsset).where(VideoAsset.playback_method == "licensed_local")
    if pitcher_id is not None:
        q = q.where(VideoAsset.pitcher_id == pitcher_id)
    if asset_ids:
        q = q.where(VideoAsset.video_asset_id.in_(asset_ids))
    assets = session.scalars(q.order_by(VideoAsset.video_asset_id)).all()
    stats = {"licensed_local_assets": len(assets), "aligned": 0, "auto_aligned": 0, "unaligned": 0,
             "skipped_up_to_date": 0, "skipped_user_corrected": 0, "skipped_not_licensed": 0, "missing_file": 0,
             "detector_version": DETECTOR_VERSION, "errors": []}
    todo = []
    for a in assets:
        if not is_licensed_local(a):
            stats["skipped_not_licensed"] += 1
            continue
        if not Path(a.permitted_media_reference).is_file():
            stats["missing_file"] += 1
            continue
        if a.synchronization_status == "user_corrected" and not force:
            stats["skipped_user_corrected"] += 1
            continue
        if not force:
            fp = file_fingerprint(a.permitted_media_reference)
            done = session.scalar(select(exists().where(
                VideoAlignmentRun.video_asset_id == a.video_asset_id,
                VideoAlignmentRun.detector_version == DETECTOR_VERSION,
                VideoAlignmentRun.file_fingerprint == fp, VideoAlignmentRun.applied.is_(True))))
            if done:
                stats["skipped_up_to_date"] += 1
                continue
        todo.append(a)
    if limit:
        todo = todo[:limit]
    for i, a in enumerate(todo):
        if should_cancel and should_cancel():
            stats["cancelled"] = True
            break
        try:
            run = align_asset(session, a, params, force=force)
            stats["aligned"] += 1
            stats[run.status] += 1
        except Exception as e:  # noqa: BLE001 - one bad file must not stop the batch
            stats["errors"].append({"video_asset_id": a.video_asset_id, "error": f"{type(e).__name__}: {e}"[:300]})
        if progress:
            progress((i + 1) / max(len(todo), 1), f"Aligned {i + 1}/{len(todo)}", i + 1)
    log(logger, "align_videos", **{k: v for k, v in stats.items() if k != "errors"}, n_errors=len(stats["errors"]))
    return stats


# ------------------------------------------------------------------ media protection
def pitch_revealed(session: Session, pitch_uid: int | None) -> bool:
    if pitch_uid is None:
        return False
    pred = session.scalar(select(exists().where(PitchPrediction.pitch_uid == pitch_uid,
                                                PitchPrediction.revealed_at.is_not(None),
                                                PitchPrediction.context.in_(REVEAL_CONTEXTS))))
    att = session.scalar(select(exists().where(ChallengeAttempt.pitch_uid == pitch_uid,
                                               ChallengeAttempt.revealed.is_(True))))
    return bool(pred or att)


def has_pre_pitch_segment(a: VideoAsset | None) -> bool:
    return (a is not None and is_licensed_local(a) and a.synchronization_status in ALIGNED
            and a.pre_pitch_pause_time is not None and a.pre_pitch_pause_time > 0)


def segment_dir() -> Path:
    d = get_settings().data_dir / "video_segments"
    d.mkdir(parents=True, exist_ok=True)
    return d


def pre_pitch_segment(a: VideoAsset) -> Path:
    """FFmpeg re-encode of frames [0, pause): exactly floor(pause * fps) frames, video only. Cached per
    (asset, file, pause); a new pause (user correction / re-alignment) produces a new file."""
    if not has_pre_pitch_segment(a):
        raise MediaNotAligned("clip is not aligned: no pre-pitch segment")
    src = Path(a.permitted_media_reference)
    if not src.is_file():
        raise LookupError("licensed file missing")
    pause = float(a.pre_pitch_pause_time)
    key = hashlib.sha1(f"{file_fingerprint(src)}|{pause:.4f}|seg-v1".encode()).hexdigest()[:16]
    out = segment_dir() / f"{a.video_asset_id}-{key}.mp4"
    if out.exists():
        return out
    meta = ffprobe(src)
    fps = meta["fps"]
    if not fps:
        raise RuntimeError("cannot trim: unknown frame rate")
    n = int(math.floor(pause * fps + 1e-6))
    if n < 1:
        raise MediaNotAligned("pre-pitch pause is before the first frame")
    exe = shutil.which("ffmpeg")
    if exe is None:
        raise RuntimeError("ffmpeg not found on PATH")
    tmp = out.with_suffix(f".tmp{os.getpid()}.mp4")
    cmd = [exe, "-y", "-v", "error", "-i", str(src), "-map", "0:v:0", "-an", "-sn", "-dn", "-frames:v", str(n),
           "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
           "-map_metadata", "-1", str(tmp)]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if r.returncode != 0 or not tmp.exists():
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"ffmpeg trim failed: {r.stderr.strip()[:300]}")
    got = ffprobe(tmp)
    if got["duration"] is not None and got["duration"] > pause + 1e-3:  # never serve frames past the pause
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"trimmed segment {got['duration']:.4f}s exceeds pause {pause:.4f}s")
    os.replace(tmp, out)
    for old in segment_dir().glob(f"{a.video_asset_id}-*.mp4"):  # stale segments of this asset
        if old != out and ".tmp" not in old.name:
            old.unlink(missing_ok=True)
    return out


def media_for(session: Session, a: VideoAsset | None, segment: str = "pre_pitch") -> Path:
    """Path to serve for GET /api/videos/{id}/media. Raises LookupError / MediaForbidden / MediaNotAligned."""
    if a is None or not is_licensed_local(a):
        raise LookupError("no licensed local media for this asset")
    if not Path(a.permitted_media_reference).is_file():
        raise LookupError("licensed file missing")
    if segment == "full":
        if not pitch_revealed(session, a.pitch_uid):
            raise MediaForbidden("The full clip is available only after a prediction for this pitch has been "
                                 "revealed; before that only the pre-pitch segment is served.")
        return Path(a.permitted_media_reference)
    if segment != "pre_pitch":
        raise ValueError("segment must be pre_pitch or full")
    if not has_pre_pitch_segment(a):
        raise MediaNotAligned("This clip is not aligned (no reliable pre-pitch pause point), so no pre-pitch "
                              "segment exists. Use the data-only view, then watch the full clip after reveal, "
                              "or correct the alignment via PUT /api/videos/{id}/alignment.")
    return pre_pitch_segment(a)


def latest_run(session: Session, video_asset_id: int) -> VideoAlignmentRun | None:
    return session.scalars(select(VideoAlignmentRun).where(VideoAlignmentRun.video_asset_id == video_asset_id)
                           .order_by(VideoAlignmentRun.alignment_run_id.desc())).first()
