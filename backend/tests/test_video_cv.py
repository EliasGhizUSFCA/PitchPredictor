"""Automatic video alignment (pitchiq.video.alignment) + server-side spoiler protection of licensed media.

Accuracy assertions run on SYNTHETIC clips from tests/video_synth.py (no licensed real footage exists here).
"""
import csv
import json
import subprocess

import numpy as np
import pytest
from sqlalchemy import select

from pitchiq.models import ChallengeAttempt, Pitch, PredictionChallenge, VideoAsset
from pitchiq.models_video_cv import VideoAlignmentRun
from pitchiq.video import alignment as A
from video_synth import SynthSpec, generate_clip

PID = 519242
TEST_SEEDS = list(range(30001, 30011))  # disjoint from the development (1..) and evaluation (10001..) seeds


def _probe(path):
    r = subprocess.run(["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0", "-show_entries",
                        "stream=nb_read_frames,avg_frame_rate,duration", "-of", "json", str(path)],
                       capture_output=True, text=True, check=True)
    st = json.loads(r.stdout)["streams"][0]
    a, b = st["avg_frame_rate"].split("/")
    return {"frames": int(st["nb_read_frames"]), "fps": float(a) / float(b), "duration": float(st["duration"])}


@pytest.fixture(scope="module")
def synth_clips(tmp_path_factory):
    d = tmp_path_factory.mktemp("synth")
    items = []
    for s in TEST_SEEDS:
        p = d / f"pitch_{s}.mp4"
        items.append((p, generate_clip(SynthSpec.random(s), p)))
    for i, kind in enumerate(["crowd_only", "closeup_only", "high_home", "no_delivery"]):
        p = d / f"{kind}.mp4"
        items.append((p, generate_clip(SynthSpec.random(31000 + i, kind=kind), p)))
    return [(p, gt, A.align_file(p)) for p, gt in items]


def _clean_clip(path, seed=777):
    spec = SynthSpec(seed=seed, fps=30.0, width=480, height=270, lead_kind="crowd", lead_s=0.8, pre_set_s=0.6,
                     come_set_s=0.4, set_s=1.4, delivery_s=1.0, follow_s=0.9, tail_kind="closeup", tail_s=0.6,
                     noise_sigma=1.5, jitter_px=0.3, crf=22)
    return generate_clip(spec, path)


def test_onset_accuracy_on_synthetic_clips(synth_clips):
    pos = [(p, gt, r) for p, gt, r in synth_clips if gt.has_delivery]
    aligned = [(p, gt, r) for p, gt, r in pos if r.status == "auto_aligned"]
    assert len(aligned) >= len(pos) - 1, [(p.name, r.reasons) for p, gt, r in pos if r.status != "auto_aligned"]
    errs = np.array([(r.onset_time - gt.onset) * 1000 for _, gt, r in aligned])
    assert np.median(np.abs(errs)) <= 100 and np.abs(errs).max() <= 200, errs
    for p, gt, r in aligned:
        # the pause point must precede the true delivery start, and release follows onset
        assert r.pre_pitch_pause_time < gt.onset, (p.name, r.pre_pitch_pause_time, gt.onset)
        assert r.onset_time < r.release_time <= r.pitch_end_time
        assert abs(r.release_time - gt.release) <= 0.35
        assert r.alignment_confidence >= A.AlignmentParams().min_confidence
        assert r.chosen_segment is not None and r.segments[r.chosen_segment]["view"] == "center_field"


def test_scene_cuts_and_views(synth_clips):
    for p, gt, r in synth_clips:
        fps = gt.spec["fps"]
        assert len(r.scene_cuts) == len(gt.scene_cuts), (p.name, r.scene_cuts, gt.scene_cuts)
        for c, d in zip(sorted(gt.scene_cuts), sorted(r.scene_cuts)):
            assert abs(c - d) <= 1.5 / fps
        if gt.has_pitcher_view:
            assert any(s["view"] == "center_field" for s in r.segments)
        else:
            assert all(s["view"] != "center_field" for s in r.segments), p.name


def test_no_pitcher_view_or_no_delivery_is_unaligned(synth_clips):
    neg = {p.stem: r for p, gt, r in synth_clips if not gt.has_delivery}
    assert set(neg) == {"crowd_only", "closeup_only", "high_home", "no_delivery"}
    for name, r in neg.items():
        assert r.status == "unaligned" and r.pre_pitch_pause_time is None, name
    assert any("no center-field" in x for x in neg["crowd_only"].reasons)


def test_undecodable_file_is_unaligned(tmp_path):
    f = tmp_path / "bad.mp4"
    f.write_bytes(b"\x00\x00\x00\x18ftypmp42 not a video")
    r = A.align_file(f)
    assert r.status == "unaligned" and r.error and "could not decode the clip" in r.reasons


def test_view_classifier_cues():
    H, W = 36, 64
    grass, dirt = np.zeros((H, W)), np.zeros((H, W))
    grass[:, :] = 1.0
    dirt[int(0.8 * H):, int(0.35 * W):int(0.65 * W)] = 1.0  # mound
    grass[dirt > 0] = 0
    dirt[int(0.15 * H):int(0.35 * H), int(0.35 * W):int(0.65 * W)] = 1.0  # home plate circle
    assert A.classify_view(grass, dirt)["view"] == "center_field"
    assert A.classify_view(np.zeros((H, W)), np.zeros((H, W)))["view"] == "other"  # crowd / close-up
    no_mound = grass.copy()
    no_mound[:] = 1.0
    assert A.classify_view(no_mound, np.zeros((H, W)))["view"] == "other"  # outfield-like


def _licensed_asset(session, tmp_path, pitch, clip_name, ref="TEST-LIC-CV"):
    from pitchiq.video.discovery import build_video_library
    from pitchiq.video.providers.licensed_local import LicensedLocalProvider
    with open(tmp_path / "manifest.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["file", "game_pk", "at_bat_number", "pitch_number", "pitcher_id", "batter_id",
                    "license_reference"])
        w.writerow([clip_name, pitch.game_pk, pitch.at_bat_number, pitch.pitch_number, PID, pitch.batter_id, ref])
    build_video_library(session, PID, providers=[LicensedLocalProvider(tmp_path)], force=True)
    session.commit()
    return session.scalars(select(VideoAsset).where(VideoAsset.source_provider == "licensed_local",
                                                    VideoAsset.pitch_uid == pitch.pitch_uid)).one()


def test_media_protection_flow(session, api, tmp_path):
    from pitchiq.challenge import create_challenge
    from pitchiq.video.align_service import align_videos
    ch = create_challenge(session, PID, 1, seed=4242)
    session.commit()
    uid = ch.pitch_uids[0]
    pitch = session.get(Pitch, uid)
    gt = _clean_clip(tmp_path / "clip.mp4")
    a = _licensed_asset(session, tmp_path, pitch, "clip.mp4")
    vid = a.video_asset_id
    try:
        assert a.synchronization_status == "unaligned"
        assert api.get(f"/api/videos/{vid}/media").status_code == 409  # not aligned yet: no segment

        stats = align_videos(session, PID, asset_ids=[vid])  # the job / CLI code path
        session.commit()
        assert stats["aligned"] == 1 and stats["auto_aligned"] == 1, stats
        session.refresh(a)
        assert a.synchronization_status == "auto_aligned" and a.pre_pitch_pause_time < gt.onset
        assert abs(a.release_frame_timestamp - gt.release) < 0.35
        run = session.scalars(select(VideoAlignmentRun).where(VideoAlignmentRun.video_asset_id == vid)).one()
        assert run.applied and run.result["motion"]["energy"] and run.detector_version == A.DETECTOR_VERSION
        assert align_videos(session, PID, asset_ids=[vid])["skipped_up_to_date"] == 1  # idempotent

        # default media = trimmed pre-pitch segment, never longer than the pause
        r = api.get(f"/api/videos/{vid}/media")
        assert r.status_code == 200 and r.headers["content-type"] == "video/mp4"
        assert "no-store" in r.headers["cache-control"]
        seg = tmp_path / "seg.mp4"
        seg.write_bytes(r.content)
        info = _probe(seg)
        assert info["duration"] <= a.pre_pitch_pause_time + 1e-6
        assert info["frames"] == int(np.floor(a.pre_pitch_pause_time * 30.0 + 1e-6))
        assert info["frames"] / info["fps"] <= a.pre_pitch_pause_time
        # full clip is forbidden before a reveal
        assert api.get(f"/api/videos/{vid}/media?segment=full").status_code == 403

        # challenge attempt: pre-reveal view exposes only the attempt-scoped pre-pitch URL (no asset id)
        nxt = api.post(f"/api/challenges/{ch.challenge_id}/next").json()
        assert nxt["mode"] == "in_app" and nxt["video"]["pre_pitch_segment_available"] is True
        url = nxt["video"]["pre_pitch_media_url"]
        assert url == f"/api/attempts/{nxt['attempt_id']}/pre-pitch-media"
        assert "video_reference" not in nxt and f"/videos/{vid}" not in json.dumps(nxt)
        r2 = api.get(url)
        assert r2.status_code == 200 and r2.content == r.content
        assert api.get(f"/api/videos/{vid}/media?segment=full").status_code == 403

        rev = api.post(f"/api/attempts/{nxt['attempt_id']}/reveal").json()
        ref = rev["video_reference"]
        assert ref["media_url"] == f"/api/videos/{vid}/media?segment=full"
        assert ref["synchronization_status"] == "auto_aligned"
        full = api.get(ref["media_url"])
        assert full.status_code == 200 and full.content == (tmp_path / "clip.mp4").read_bytes()

        # user correction -> new (shorter) segment; automatic re-alignment keeps the user's value
        assert api.put(f"/api/videos/{vid}/alignment", json={"pre_pitch_pause_time": 1.0}).status_code == 200
        r3 = api.get(f"/api/videos/{vid}/media")
        seg3 = tmp_path / "seg3.mp4"
        seg3.write_bytes(r3.content)
        assert r3.status_code == 200 and _probe(seg3)["duration"] <= 1.0 + 1e-6
        session.expire_all()  # the PUT committed in the API's own session
        assert align_videos(session, PID, asset_ids=[vid])["skipped_user_corrected"] == 1
        st = api.get(f"/api/videos/{vid}/alignment").json()
        assert st["synchronization_status"] == "user_corrected" and st["pre_pitch_pause_time"] == 1.0
        assert st["latest_run"]["status"] == "auto_aligned" and st["latest_run"]["motion"]["energy"]
    finally:
        session.rollback()
        session.query(ChallengeAttempt).filter_by(challenge_id=ch.challenge_id).delete()
        session.query(PredictionChallenge).filter_by(challenge_id=ch.challenge_id).delete()
        session.query(VideoAsset).filter_by(video_asset_id=vid).delete()
        session.commit()


def test_unaligned_clip_has_no_pre_pitch_segment(session, api, tmp_path):
    from pitchiq.models import PrePitchState
    uid = session.scalar(select(PrePitchState.pitch_uid).where(PrePitchState.split == "validation").limit(1))
    pitch = session.get(Pitch, uid)
    generate_clip(SynthSpec.random(31100, kind="crowd_only"), tmp_path / "crowd.mp4")
    a = _licensed_asset(session, tmp_path, pitch, "crowd.mp4")
    vid = a.video_asset_id
    try:
        out = api.post(f"/api/videos/{vid}/align").json()
        assert out["status"] == "unaligned" and out["synchronization_status"] == "unaligned"
        assert out["pre_pitch_pause_time"] is None and out["reasons"]
        assert api.get(f"/api/videos/{vid}/media").status_code == 409
        assert api.get(f"/api/videos/{vid}/media?segment=full").status_code == 403
        # after a dashboard reveal of this pitch the full clip opens (data-only -> after-reveal flow)
        pred = api.post(f"/api/pitchers/{PID}/pitches/{uid}/predict").json()
        rev = api.post(f"/api/predictions/{pred['prediction_id']}/reveal").json()
        assert rev["video_reference"]["media_url"].endswith("segment=full")
        assert rev["video_reference"]["pre_pitch_media_url"] is None
        assert api.get(f"/api/videos/{vid}/media?segment=full").status_code == 200
        assert api.get(f"/api/videos/{vid}/media").status_code == 409
    finally:
        session.rollback()
        session.query(VideoAsset).filter_by(video_asset_id=vid).delete()
        session.commit()


def test_alignment_rejects_external_assets(session, api):
    ext = session.scalars(select(VideoAsset).where(VideoAsset.playback_method == "external_link")).first()
    assert api.post(f"/api/videos/{ext.video_asset_id}/align").status_code == 422
    assert api.get(f"/api/videos/{ext.video_asset_id}/alignment").status_code == 422
    assert api.get(f"/api/videos/{ext.video_asset_id}/media").status_code == 404


def test_align_videos_job_registered():
    from pitchiq.jobs.tasks import DESCRIPTIONS, TASKS
    assert "align_videos" in TASKS and "align_videos" in DESCRIPTIONS


def test_validation_report_is_stored_and_labelled_synthetic(session, synth_clips, api):
    from pitchiq.video.alignment_eval import evaluate_clips, store_validation
    from video_synth import GENERATOR_VERSION
    items = [(p, gt) for p, gt, _ in synth_clips[:3]]
    rep = evaluate_clips(items, A.AlignmentParams(), workers=1)
    rep["seeds"] = {"pitch": [TEST_SEEDS[0], TEST_SEEDS[2]]}
    row = store_validation(session, rep, role="development", generator=f"tests/video_synth.py:{GENERATOR_VERSION}")
    session.commit()
    s = row.summary
    assert row.data_kind == "synthetic" and "synthetic" in s["data"] and s["n_clips"] == 3
    assert s["onset_abs_error_ms"]["median"] is not None and row.mlflow_run_id  # mirrored (test MLflow store)
    got = api.get("/api/videos/alignment/validations").json()
    assert got[0]["validation_id"] == row.validation_id and got[0]["data_kind"] == "synthetic"
    session.delete(row)
    session.commit()
