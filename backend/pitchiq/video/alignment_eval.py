"""Score the alignment detector against clips with known ground truth (synthetic, see tests/video_synth.py).

`evaluate_clips` runs `align_file` on each clip and compares with the ground truth; `store_validation` writes the
report to `video_alignment_validations` and mirrors it to MLflow (experiment `<prefix>-video-cv`).
Every number here comes from running the detector on the given clips; nothing is estimated or filled in.
"""
from __future__ import annotations

import logging
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path

import numpy as np

from ..logging_setup import log
from .alignment import DETECTOR_VERSION, AlignmentParams, align_file

logger = logging.getLogger(__name__)

GROSS_ERROR_MS = 200.0  # an auto-aligned onset further than this from the truth counts as a failure


def _gt_dict(gt) -> dict:
    return gt if isinstance(gt, dict) else asdict(gt)


def score_clip(path: str | Path, gt, params: AlignmentParams | None = None) -> dict:
    g = _gt_dict(gt)
    t0 = time.perf_counter()
    r = align_file(path, params)
    ms = (time.perf_counter() - t0) * 1000
    spec = g.get("spec", {})
    row = {"clip": Path(path).name, "kind": spec.get("kind", "pitch"), "fps": spec.get("fps"),
           "width": spec.get("width"), "height": spec.get("height"), "noise_sigma": spec.get("noise_sigma"),
           "crf": spec.get("crf"), "has_delivery": g["has_delivery"], "has_pitcher_view": g["has_pitcher_view"],
           "status": r.status, "confidence": r.alignment_confidence, "runtime_ms": round(ms, 1),
           "reasons": r.reasons, "error": r.error, "weakest": (min(r.confidence_components,
                                                                    key=r.confidence_components.get)
                                                                if r.confidence_components else None)}
    fps = spec.get("fps") or r.fps or 30.0
    gc, dc = g.get("scene_cuts") or [], r.scene_cuts or []
    tol = 1.5 / fps
    matched = sum(any(abs(c - d) <= tol for d in dc) for c in gc)
    row.update(cuts_true=len(gc), cuts_detected=len(dc), cuts_matched=matched)
    if g["has_delivery"] and r.onset_time is not None:
        row["onset_error_ms"] = round((r.onset_time - g["onset"]) * 1000, 1)
        row["release_error_ms"] = round((r.release_time - g["release"]) * 1000, 1)
        row["pause_before_true_onset"] = bool(r.pre_pitch_pause_time < g["onset"])
        row["pause_lead_ms"] = round((g["onset"] - r.pre_pitch_pause_time) * 1000, 1)
        row["onset_error_frames"] = round((r.onset_time - g["onset"]) * fps, 2)
    if g.get("scoreboard_box"):
        det = (r.scoreboard_region or {}).get("box")
        row["scoreboard_iou"] = round(_iou(det, g["scoreboard_box"]), 3) if det else 0.0
    elif r.scoreboard_region:
        row["scoreboard_false_positive"] = True
    return row


def _iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


def _pct(x, q):
    return round(float(np.percentile(x, q)), 1) if len(x) else None


def summarize(rows: list[dict]) -> dict:
    pos = [r for r in rows if r["has_delivery"]]
    neg = [r for r in rows if not r["has_delivery"]]
    aligned = [r for r in pos if r["status"] == "auto_aligned"]
    err = np.array([r["onset_error_ms"] for r in aligned], float)
    aerr = np.abs(err)
    rel = np.array([r["release_error_ms"] for r in aligned], float)
    gross = [r for r in aligned if abs(r["onset_error_ms"]) > GROSS_ERROR_MS]
    errf = np.abs(np.array([r["onset_error_frames"] for r in aligned], float))
    cut_true = sum(r["cuts_true"] for r in rows)
    sb = [r["scoreboard_iou"] for r in rows if "scoreboard_iou" in r]
    by_fps: dict = {}
    for r in aligned:
        k = f"{r['fps']:.2f}"
        by_fps.setdefault(k, []).append(abs(r["onset_error_ms"]))
    return {
        "data": "synthetic clips only (tests/video_synth.py); not real broadcast footage",
        "n_clips": len(rows), "n_positive": len(pos), "n_negative": len(neg),
        "positive_auto_aligned": len(aligned),
        "positive_unaligned": len(pos) - len(aligned),
        "unaligned_rate": round((len(pos) - len(aligned)) / len(pos), 4) if pos else None,
        "gross_error_threshold_ms": GROSS_ERROR_MS,
        "gross_errors": len(gross),
        "failure_rate": round((len(pos) - len(aligned) + len(gross)) / len(pos), 4) if pos else None,
        "onset_abs_error_ms": {"median": _pct(aerr, 50), "p90": _pct(aerr, 90), "p95": _pct(aerr, 95),
                               "max": _pct(aerr, 100), "mean": round(float(aerr.mean()), 1) if len(aerr) else None},
        "onset_signed_error_ms": {"median": _pct(err, 50), "p10": _pct(err, 10), "p90": _pct(err, 90)},
        "onset_abs_error_frames": {"median": _pct(errf, 50), "p90": _pct(errf, 90)},
        "onset_abs_error_ms_by_fps": {k: {"n": len(v), "median": _pct(v, 50), "p90": _pct(v, 90)}
                                      for k, v in sorted(by_fps.items())},
        "release_error_ms": {"median_abs": _pct(np.abs(rel), 50), "p90_abs": _pct(np.abs(rel), 90),
                             "median_signed": _pct(rel, 50)},
        "pause_before_true_onset_rate": round(float(np.mean([r["pause_before_true_onset"] for r in aligned])), 4)
        if aligned else None,
        "pause_after_true_onset_clips": [r["clip"] for r in aligned if not r["pause_before_true_onset"]],
        "negatives_falsely_aligned": [r["clip"] for r in neg if r["status"] == "auto_aligned"],
        "negative_false_alignment_rate": round(sum(r["status"] == "auto_aligned" for r in neg) / len(neg), 4)
        if neg else None,
        "scene_cut_recall": round(sum(r["cuts_matched"] for r in rows) / cut_true, 4) if cut_true else None,
        "scene_cut_false_positives": int(sum(max(r["cuts_detected"] - r["cuts_matched"], 0) for r in rows)),
        "scoreboard": {"n_with_scoreboard": len(sb), "detected_iou_ge_0_5": int(sum(x >= 0.5 for x in sb)),
                       "median_iou": _pct(sb, 50),
                       "false_positives_on_clips_without": int(sum(bool(r.get("scoreboard_false_positive"))
                                                                   for r in rows))},
        "unaligned_positive_reasons": {r["clip"]: r["reasons"] for r in pos if r["status"] != "auto_aligned"},
        "runtime_ms_median": _pct([r["runtime_ms"] for r in rows], 50),
    }


def _score_star(a):
    return score_clip(*a)


def evaluate_clips(items, params: AlignmentParams | None = None, workers: int = 1) -> dict:
    """items: [(path, ground_truth)] -> {"summary", "per_clip", "params", "detector_version"}."""
    params = params or AlignmentParams()
    args = [(str(p), _gt_dict(g), params) for p, g in items]
    if workers > 1:
        with ProcessPoolExecutor(workers) as ex:
            rows = list(ex.map(_score_star, args))
    else:
        rows = [score_clip(*a) for a in args]
    return {"detector_version": DETECTOR_VERSION, "params": params.as_dict(), "summary": summarize(rows),
            "per_clip": rows}


def store_validation(session, report: dict, role: str, generator: str, params: AlignmentParams | None = None):
    from ..models_video_cv import VideoAlignmentValidation
    row = VideoAlignmentValidation(detector_version=report["detector_version"], role=role, data_kind="synthetic",
                                   generator=generator, n_clips=report["summary"]["n_clips"],
                                   params=report["params"], summary=report["summary"],
                                   per_clip=report["per_clip"], seeds=report.get("seeds"))
    session.add(row)
    session.flush()
    row.mlflow_run_id = mirror_validation(row)
    session.flush()
    return row


def mirror_validation(row) -> str | None:
    """Best-effort MLflow mirror (Postgres row is the record)."""
    try:
        from ..ml.tracking import get_tracker
        tr = get_tracker()
        if not tr.enabled:
            return None
        c = tr.client
        name = f"{tr.prefix}-video-cv"
        exp = c.get_experiment_by_name(name)
        eid = exp.experiment_id if exp else c.create_experiment(
            name, artifact_location=(tr.artifact_root / name).resolve().as_uri() if tr._local_store() else None,
            tags={"pitchiq.source_of_truth": "postgres"})
        run = c.create_run(eid, run_name=f"alignment-{row.role}-{row.validation_id}",
                           tags={"pitchiq.kind": "video_alignment_validation", "pitchiq.role": row.role,
                                 "pitchiq.data_kind": "synthetic", "pitchiq.detector_version": row.detector_version,
                                 "pitchiq.generator": row.generator,
                                 "pitchiq.validation_id": str(row.validation_id),
                                 "pitchiq.note": "accuracy on synthetic clips only; not real broadcast footage"})
        rid = run.info.run_id
        s = row.summary
        flat = {"n_clips": s["n_clips"], "n_positive": s["n_positive"], "n_negative": s["n_negative"],
                "unaligned_rate": s["unaligned_rate"], "failure_rate": s["failure_rate"],
                "gross_errors": s["gross_errors"],
                "onset_abs_error_ms_median": s["onset_abs_error_ms"]["median"],
                "onset_abs_error_ms_p90": s["onset_abs_error_ms"]["p90"],
                "onset_signed_error_ms_median": s["onset_signed_error_ms"]["median"],
                "release_abs_error_ms_median": s["release_error_ms"]["median_abs"],
                "pause_before_true_onset_rate": s["pause_before_true_onset_rate"],
                "negative_false_alignment_rate": s["negative_false_alignment_rate"],
                "scene_cut_recall": s["scene_cut_recall"]}
        tr._log_batch(rid, {k: v for k, v in (row.params or {}).items()},
                      [(k, float(v), 0) for k, v in flat.items() if v is not None], int(time.time() * 1000))
        c.log_dict(rid, {"summary": s, "per_clip": row.per_clip}, "video_alignment_validation.json")
        c.set_terminated(rid, "FINISHED")
        return rid
    except Exception as e:  # noqa: BLE001 - mirror failures never fail the validation
        log(logger, "mlflow_mirror_failed", op="video_alignment_validation", error=f"{type(e).__name__}: {e}"[:500])
        return None
