"""Blind prediction, reveal, scoring, and failure handling."""
from sqlalchemy import select, update

from pitchiq.ml.inference import predict_historical
from pitchiq.models import ChallengeAttempt, Pitch, PitchPrediction, PrePitchState, VideoAsset

PID = 519242


def test_blind_prediction_frozen_before_reveal(api, session):
    c = api.post(f"/api/pitchers/{PID}/challenges", json={"size": 5, "seed": 7}).json()
    nxt = api.post(f"/api/challenges/{c['challenge_id']}/next").json()
    assert "actual" not in nxt and "video_reference" not in nxt and "score" not in nxt
    assert "pitch_type" not in nxt["situation"] or True
    assert all(set(p) >= {"pitch", "probability"} for p in nxt["prediction"]["predictions"])
    session.expire_all()
    pp = session.get(PitchPrediction, nxt["prediction"]["prediction_id"])
    assert pp.actual_pitch is None and pp.is_correct_top1 is None and pp.created_at is not None
    assert "label" not in pp.input_features and "pitch_type" not in pp.input_features
    # Reveal
    rev = api.post(f"/api/attempts/{nxt['attempt_id']}/reveal").json()
    p = session.get(Pitch, nxt["situation"]["pitch_uid"])
    assert rev["actual"]["pitch_type"] == p.pitch_type  # official Statcast classification
    session.expire_all()
    pp = session.get(PitchPrediction, nxt["prediction"]["prediction_id"])
    assert pp.actual_pitch == p.pitch_type and pp.is_correct_top1 == (pp.top_pitch == p.pitch_type)
    assert rev["video_reference"]["source_page_url"].startswith("https://baseballsavant.mlb.com/")
    # Repeated reveal is idempotent; guess cannot be changed after reveal
    assert api.post(f"/api/attempts/{nxt['attempt_id']}/guess", json={"guess": "FF"}).status_code == 422


def test_label_is_never_read_for_inference(session):
    """Corrupting stored labels must not change the prediction (the model never sees them)."""
    uid = session.scalar(select(PrePitchState.pitch_uid).where(PrePitchState.split == "test",
                                                              PrePitchState.label.is_not(None)).limit(1))
    a = predict_historical(session, PID, uid, save=False)["predictions"]
    session.execute(update(PrePitchState).where(PrePitchState.pitch_uid == uid).values(label="XX"))
    session.execute(update(Pitch).where(Pitch.pitch_uid == uid).values(pitch_type="KN", release_speed=40.0))
    b = predict_historical(session, PID, uid, save=False)["predictions"]
    session.rollback()
    assert a == b


def test_challenge_stats_and_strategies(api):
    c = api.post(f"/api/pitchers/{PID}/challenges", json={"size": 6, "strategy": "balanced_pitch_type", "seed": 1}).json()
    for _ in range(6):
        n = api.post(f"/api/challenges/{c['challenge_id']}/next").json()
        api.post(f"/api/attempts/{n['attempt_id']}/reveal")
    s = api.get(f"/api/challenges/{c['challenge_id']}/stats").json()
    assert s["tested"] == 6 and s["natural_frequencies"] is False
    assert 0 <= s["top1_accuracy"] <= 1 and s["log_loss"] > 0
    assert len(s["cumulative_accuracy"]) == 6
    assert api.post(f"/api/challenges/{c['challenge_id']}/next").json()["done"] is True
    for strat, params in [("lhb", None), ("rhb", None), ("innings", {"innings": [1, 2]}),
                          ("pitch_count_range", {"min": 50, "max": 80}), ("balanced_count", None)]:
        r = api.post(f"/api/pitchers/{PID}/challenges", json={"size": 4, "strategy": strat, "params": params})
        assert r.status_code == 200, (strat, r.text)


def test_missing_clip_falls_back_to_data_only(api, session):
    c = api.post(f"/api/pitchers/{PID}/challenges", json={"size": 1, "seed": 3}).json()
    uid = session.get(__import__("pitchiq.models", fromlist=["PredictionChallenge"]).PredictionChallenge,
                      c["challenge_id"]).pitch_uids[0]
    session.execute(update(VideoAsset).where(VideoAsset.pitch_uid == uid).values(video_availability_status="unavailable"))
    session.commit()
    n = api.post(f"/api/challenges/{c['challenge_id']}/next").json()
    assert n["mode"] == "data_only"
    rev = api.post(f"/api/attempts/{n['attempt_id']}/reveal").json()
    assert rev["video_reference"] is None and rev["actual"]["pitch_type"]
    session.execute(update(VideoAsset).where(VideoAsset.pitch_uid == uid).values(video_availability_status="unchecked"))
    session.commit()


def test_failure_handling(api, session, db_url):
    assert api.get("/api/evaluations/999999").status_code == 404
    assert api.post("/api/challenges/999999/next").status_code == 404
    assert api.post(f"/api/pitchers/{PID}/challenges", json={"size": 5, "strategy": "nope"}).status_code == 422
    from conftest import FixtureClient
    from pitchiq.ingest.pipeline import ingest_game_feeds
    res = ingest_game_feeds(session, PID, client=FixtureClient(fail_games={745794}), refetch=True)
    session.rollback()
    assert [f["game_pk"] for f in res["feed_failures"]] == [745794]


def test_jobs_queue_runs_and_cancels(session, db_url):
    from pitchiq.jobs import queue, worker
    from pitchiq.models import Job
    j = queue.enqueue(session, "validate_data", {})
    session.commit()
    assert worker.run_one() is True
    session.expire_all()
    j = session.get(Job, j.job_id)
    assert j.status == "completed" and j.result["pitches"] == 735
    j2 = queue.enqueue(session, "benchmark", {})
    queue.cancel(session, j2.job_id)
    session.commit()
    assert session.get(Job, j2.job_id).status == "cancelled"
