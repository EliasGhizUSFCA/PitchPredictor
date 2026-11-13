"""Model, accuracy-metric, comparison tests."""
import numpy as np
import pytest
from sqlalchemy import select

from pitchiq.features.store import load_feature_frame
from pitchiq.ml import registry
from pitchiq.ml.metrics import brier, confusion, log_loss, summarize, topk_acc
from pitchiq.ml.models import TemperatureCalibrator
from pitchiq.models import EvaluationRun, PitchPrediction

PID = 519242


def test_all_families_return_normalized_probabilities(session):
    df = load_feature_frame(session, PID, ["test"])
    fams = registry.latest_by_family(session, PID)
    assert set(fams) == {"majority", "frequency", "logreg", "lightgbm"}
    for e in fams.values():
        P = registry.load_model(e).predict_proba(df)
        assert P.shape == (len(df), len(e.classes))
        assert np.all(P >= 0) and np.allclose(P.sum(1), 1.0)


def test_temperature_calibration_preserves_normalization():
    rng = np.random.default_rng(0)
    P = rng.dirichlet(np.ones(4), size=300)
    y = rng.integers(0, 4, 300)
    Q = TemperatureCalibrator().fit(P, y).transform(P)
    assert np.allclose(Q.sum(1), 1.0) and np.all(Q >= 0)


def test_metrics_hand_computed():
    P = np.array([[0.56, 0.30, 0.10, 0.04], [0.7, 0.1, 0.1, 0.1], [0.25, 0.25, 0.25, 0.25]])
    y = np.array([1, 0, 3])
    assert topk_acc(P, y, 1) == pytest.approx(1 / 3)
    assert topk_acc(P, y, 2) == pytest.approx(2 / 3)
    assert log_loss(P, y) == pytest.approx(-(np.log(0.30) + np.log(0.7) + np.log(0.25)) / 3)
    b0 = 0.56**2 + 0.7**2 + 0.1**2 + 0.04**2
    b1 = 0.3**2 + 0.1**2 + 0.1**2 + 0.1**2
    b2 = 3 * 0.25**2 + 0.75**2
    assert brier(P, y) == pytest.approx((b0 + b1 + b2) / 3)
    cm = confusion(P, y, ["FF", "SL", "CH", "SI"])
    assert cm["matrix"][1][0] == 1 and cm["matrix"][0][0] == 1 and cm["matrix"][3][0] == 1
    s = summarize(P, y, ["FF", "SL", "CH", "SI"])
    assert s["top1_correct"] == 1 and s["mean_prob_actual"] == pytest.approx((0.3 + 0.7 + 0.25) / 3)


def test_models_compared_on_same_heldout_examples(session, pipeline):
    runs = session.scalars(select(EvaluationRun).where(EvaluationRun.evaluation_run_id.in_(pipeline["benchmarks"]))).all()
    assert len(runs) == 4
    assert len({r.n_samples for r in runs}) == 1
    assert len({(r.dataset_definition["pitch_uid_min"], r.dataset_definition["pitch_uid_max"]) for r in runs}) == 1
    uidsets = [set(session.scalars(select(PitchPrediction.pitch_uid).where(
        PitchPrediction.evaluation_run_id == r.evaluation_run_id))) for r in runs]
    assert all(u == uidsets[0] for u in uidsets)


def test_benchmark_metrics_match_stored_predictions(session, pipeline):
    run = session.get(EvaluationRun, pipeline["benchmarks"][0])
    preds = session.scalars(select(PitchPrediction).where(PitchPrediction.evaluation_run_id == run.evaluation_run_id)).all()
    acc = np.mean([p.is_correct_top1 for p in preds])
    ll = -np.mean([np.log(max(p.prob_actual, 1e-15)) for p in preds])
    assert acc == pytest.approx(run.metrics["summary"]["top1_accuracy"])
    assert ll == pytest.approx(run.metrics["summary"]["log_loss"], rel=1e-6)


def test_hyperparameters_never_see_test(session):
    for e in registry.latest_by_family(session, PID).values():
        assert set(e.dataset_fingerprint) == {"train", "validation"}
        assert e.calibration["fit_split"] == "validation"
