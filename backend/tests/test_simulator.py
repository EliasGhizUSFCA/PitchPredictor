"""Simulator: input changes trigger genuine model inference; invalid states are rejected."""
import numpy as np
import pytest

from pitchiq.ml import registry
from pitchiq.ml.inference import SimState, predict_scenario

PID = 519242


def _probs(session, **kw):
    r = predict_scenario(session, PID, SimState(**kw), persist_trace=False)
    return {d["pitch"]: d["probability"] for d in r["predictions"]}, r


def test_changes_trigger_real_inference(session, monkeypatch):
    calls = []
    entry = registry.promoted(session, PID)
    model = registry.load_model(entry)
    orig = model.predict_proba

    def spy(X, calibrated=True):
        calls.append(X.copy())
        return orig(X, calibrated=calibrated)
    monkeypatch.setattr(model, "predict_proba", spy)
    base = dict(stand="R", inning=3, pitch_count_before=40, times_through_order=2, balls=1, strikes=1)
    p0, _ = _probs(session, **base)
    variants = [(dict(balls=0, strikes=2), "count_str", "0-2"), (dict(inning=7), "inning", 7),
                (dict(stand="L"), "stand", "L"), (dict(pitch_count_before=95), "pitch_count_before", 95)]
    outputs = []
    for v, col, val in variants:
        n_before = len(calls)
        p1, r = _probs(session, **{**base, **v})
        assert np.isclose(sum(p1.values()), 1.0)
        new = calls[n_before:]
        assert new, f"changing {v} must run inference"
        assert all(c[col].iloc[0] == val for c in new), f"{col} change must reach the model input"
        outputs.append(p1)
    # Not every feature need move every probability, but the model must respond to the scenario.
    assert any(o != p0 for o in outputs)


@pytest.mark.parametrize("bad", [
    dict(balls=4),
    dict(strikes=3),
    dict(balls=2, strikes=0, pitch_count_before=3, previous_pitches=[{"type": "SL", "result": "ball"}]),
    dict(inning=1, times_through_order=2),
    dict(pitch_count_before=1, previous_pitches=[{"type": "SL", "result": "ball", "same_pa": False},
                                                 {"type": "FF", "result": "ball", "same_pa": False}]),
])
def test_invalid_states_rejected(api, bad):
    r = api.post(f"/api/pitchers/{PID}/simulate", json=bad)
    assert r.status_code == 422


def test_sweep_and_compare_endpoints(api):
    r = api.post(f"/api/pitchers/{PID}/simulate/sweep", json={"state": {}, "field": "count"})
    assert r.status_code == 200 and len(r.json()["points"]) == 12
    r = api.post(f"/api/pitchers/{PID}/simulate/compare",
                 json={"a": {"balls": 0, "strikes": 0, "inning": 2, "pitch_count_before": 25, "times_through_order": 1},
                       "b": {"balls": 0, "strikes": 2, "inning": 7, "pitch_count_before": 95, "times_through_order": 3}})
    assert r.status_code == 200
    d = r.json()
    assert abs(sum(d["percentage_point_diff"].values())) < 1e-6
    assert "support" in d["a"] and "warnings" in d["b"]["support"]


def test_unknown_previous_pitch_is_marginalized(session):
    _, r = _probs(session, pitch_count_before=50, inning=4, times_through_order=2)
    assert r["assumptions"]["marginalized_over"] is not None
    assert np.isclose(sum(d["probability"] for d in r["predictions"]), 1.0)
