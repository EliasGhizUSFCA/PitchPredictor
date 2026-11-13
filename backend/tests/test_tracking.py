"""MLflow mirror: runs, params, metrics and artifacts exist; run ids round-trip to Postgres; no-op paths are safe.

The test session tracks into a throwaway SQLite store under the session tmp dir (see conftest.py).
"""
import sys
from pathlib import Path

import pytest
from sqlalchemy import select

from pitchiq.config import get_settings
from pitchiq.ml import registry, tracking
from pitchiq.models import EvaluationRun, ModelRegistry

PID = 519242


@pytest.fixture()
def tracker(pipeline):
    t = tracking.get_tracker()
    assert t.enabled, t.disabled_reason
    assert "pitchiq-test-" in t.uri  # never the real backend/mlruns store
    return t


def _artifacts(t, run_id, path=None):
    return {Path(a.path).name for a in t.client.list_artifacts(run_id, path)}


def test_every_trained_model_has_a_training_run(session, tracker):
    entries = session.scalars(select(ModelRegistry).where(ModelRegistry.pitcher_id == PID)).all()
    assert entries
    for e in entries:
        rid = e.config.get("mlflow_run_id")
        assert rid, e.model_version
        r = tracker.client.get_run(rid)
        tags, params, metrics = r.data.tags, r.data.params, r.data.metrics
        assert tags["pitchiq.run_kind"] == "training"
        assert tags["pitchiq.model_version"] == e.model_version
        assert tags["pitchiq.family"] == e.model_family
        assert tags["pitchiq.feature_version"] == e.feature_version
        assert tags["pitchiq.artifact_sha256"] == e.artifact_sha256
        assert "train=[2023, 2024]" in tags["pitchiq.split_policy"]
        assert params["family"] == e.model_family
        assert int(params["n_features"]) == len(e.feature_schema["used_features"])
        assert params["data.train.pitch_uid_hash"] == e.dataset_fingerprint["train"]["pitch_uid_hash"]
        for k, v in (e.config.get("params") or {}).items():
            assert params[f"hp.{k}"] == str(v)
        assert metrics["validation_log_loss"] == pytest.approx(e.validation_metrics["log_loss"])
        assert metrics["validation_top1_accuracy"] == pytest.approx(e.validation_metrics["top1_accuracy"])
        assert metrics["calibration_temperature"] == pytest.approx(e.calibration["temperature"])
        hist = tracker.client.get_metric_history(rid, "grid_val_log_loss")
        assert [h.value for h in sorted(hist, key=lambda h: h.step)] == pytest.approx(
            [g["val_log_loss"] for g in e.config["grid_results"]])
        assert {"grid_results.json", "features.json", "dataset_fingerprint.json", "config.json",
                "calibration.json"} <= _artifacts(tracker, rid)
        assert _artifacts(tracker, rid, "model") == {Path(e.artifact_path).name}
        assert tags["pitchiq.artifact_logged"] == "true"
        assert r.info.experiment_id == tracker.experiment_id(PID)


def test_benchmark_runs_mirrored_with_cis_and_artifacts(session, tracker, pipeline):
    for run_id in pipeline["benchmarks"]:
        er = session.get(EvaluationRun, run_id)
        rid = er.metrics.get("mlflow_run_id")
        assert rid
        r = tracker.client.get_run(rid)
        assert r.data.tags["pitchiq.evaluation_run_id"] == str(er.evaluation_run_id)
        assert r.data.tags["pitchiq.model_version"] == er.model_version
        assert r.data.tags["pitchiq.evaluation_kind"] == "holdout_benchmark"
        m = r.data.metrics
        assert m["test_log_loss"] == pytest.approx(er.metrics["summary"]["log_loss"])
        assert m["test_top1_accuracy"] == pytest.approx(er.metrics["summary"]["top1_accuracy"])
        lo, hi = er.metrics["bootstrap"]["log_loss_ci95"]
        assert (m["test_log_loss_boot_ci95_lo"], m["test_log_loss_boot_ci95_hi"]) == pytest.approx((lo, hi))
        assert m["test_ece"] == pytest.approx(er.metrics["calibration"]["ece_top_label"])
        assert {"confusion.json", "calibration.json", "bootstrap.json", "breakdowns.json"} <= _artifacts(tracker, rid)
        model = session.get(ModelRegistry, er.model_version)
        assert r.data.tags["pitchiq.model_run_id"] == model.config["mlflow_run_id"]


def test_walk_forward_and_ablation_mirrored(session, tracker):
    from pitchiq.ml.evaluate import ablation, walk_forward
    wf = walk_forward(session, PID, "frequency", block_games=50)
    ab = ablation(session, PID, "logreg")
    for er in (wf, ab):
        r = tracker.client.get_run(er.metrics["mlflow_run_id"])
        assert r.data.tags["pitchiq.evaluation_kind"] == er.kind
        assert r.data.tags["pitchiq.evaluation_run_id"] == str(er.evaluation_run_id)
    wm = tracker.client.get_run(wf.metrics["mlflow_run_id"]).data.metrics
    assert wm["test_log_loss"] == pytest.approx(wf.metrics["summary"]["log_loss"])
    am = tracker.client.get_run(ab.metrics["mlflow_run_id"]).data.metrics
    for res in ab.metrics["results"]:
        assert am[f"ablation/{res['config']}/delta_log_loss"] == pytest.approx(res["delta_log_loss"])
    assert "results.json" in _artifacts(tracker, ab.metrics["mlflow_run_id"])


def test_promoted_model_registered_and_alias_follows_promotion(session, tracker):
    name = tracker.experiment_name(PID)
    promoted = registry.promoted(session, PID)
    mv = tracker.client.get_model_version_by_alias(name, tracking.PROMOTED_ALIAS)
    assert mv.run_id == promoted.config["mlflow_run_id"]
    other = next(e for e in registry.latest_by_family(session, PID).values() if e.status == "candidate")
    try:
        registry.promote(session, other.model_version)
        mv2 = tracker.client.get_model_version_by_alias(name, tracking.PROMOTED_ALIAS)
        assert mv2.run_id == other.config["mlflow_run_id"]
        assert tracker.client.get_run(other.config["mlflow_run_id"]).data.tags["pitchiq.status"] == "promoted"
        assert tracker.client.get_run(promoted.config["mlflow_run_id"]).data.tags["pitchiq.status"] == "candidate"
    finally:  # restore the original promotion in MLflow too (the DB side is rolled back by the fixture)
        registry.promote(session, promoted.model_version)
    mv3 = tracker.client.get_model_version_by_alias(name, tracking.PROMOTED_ALIAS)
    assert mv3.run_id == promoted.config["mlflow_run_id"]


def test_backfill_into_fresh_store_is_idempotent_and_tags_orphans(session, pipeline, tmp_path):
    t = tracking.Tracker(uri=f"sqlite:///{tmp_path / 'mlflow.db'}", artifact_root=tmp_path / "art", enabled=True)
    # an archived row must be mirrored too
    arch = session.get(ModelRegistry, pipeline["models"][0])
    arch.status = "archived"
    session.flush()
    n_models = len(session.scalars(select(ModelRegistry)).all())
    n_evals = len(session.scalars(select(EvaluationRun)).all())
    rep = t.backfill(session, orphan_grace_s=0)
    assert rep["models"]["created"] == n_models and rep["models"]["failed"] == 0
    assert rep["models"]["by_status"].get("archived", 0) >= 1
    assert rep["evaluations"]["created"] == n_evals and rep["evaluations"]["failed"] == 0
    assert len(rep["registered_promoted"]) == 1
    arch_run = t.client.get_run(arch.config["mlflow_run_id"])
    assert arch_run.data.tags["pitchiq.status"] == "archived"
    assert arch_run.data.tags["pitchiq.backfilled"] == "true"
    assert arch_run.info.start_time == int(arch.created_at.timestamp() * 1000)

    rep2 = t.backfill(session, orphan_grace_s=0)
    assert rep2["models"]["created"] == 0 and rep2["models"]["already_mirrored"] == n_models
    assert rep2["evaluations"]["created"] == 0 and rep2["evaluations"]["already_mirrored"] == n_evals
    assert rep2["orphaned_runs_tagged"] == 0

    # A run whose Postgres row was never committed (rolled-back transaction) gets tagged as an orphan.
    eid = t.experiment_id(PID)
    ghost = t.client.create_run(eid, tags={"pitchiq.run_kind": "training", "pitchiq.model_version": "ghost",
                                           "pitchiq.database": str(session.get_bind().url.database)})
    t.client.set_terminated(ghost.info.run_id)
    rep3 = t.backfill(session, orphan_grace_s=0)
    assert rep3["orphaned_runs_tagged"] == 1
    assert t.client.get_run(ghost.info.run_id).data.tags["pitchiq.orphaned"] == "true"

    # A stored run id from a different store is re-created, not trusted.
    arch.config = {**arch.config, "mlflow_run_id": "0" * 32}
    t.client.delete_run(arch_run.info.run_id)
    rep4 = t.backfill(session, orphan_grace_s=0)
    assert rep4["models"]["created"] == 1
    assert arch.config["mlflow_run_id"] != "0" * 32


def test_noop_when_disabled(tmp_path, session):
    t = tracking.Tracker(uri=f"sqlite:///{tmp_path / 'off' / 'mlflow.db'}", enabled=False)
    assert not t.enabled and "PITCHIQ_MLFLOW_ENABLED" in t.disabled_reason
    e = registry.promoted(session, PID)
    assert t.log_model_entry(e) is None
    assert t.log_evaluation(session.scalars(select(EvaluationRun)).first(), e) is None
    assert t.backfill(session) == {"enabled": False, "disabled_reason": t.disabled_reason}
    assert not (tmp_path / "off").exists()


def test_noop_when_mlflow_not_importable(tmp_path, session, monkeypatch):
    monkeypatch.setitem(sys.modules, "mlflow.tracking", None)  # makes `from mlflow.tracking import ...` fail
    t = tracking.Tracker(uri=f"sqlite:///{tmp_path / 'x' / 'mlflow.db'}", enabled=True)
    assert not t.enabled and "not importable" in t.disabled_reason
    assert t.log_model_entry(registry.promoted(session, PID)) is None
    assert t.status()["enabled"] is False


def test_training_unaffected_when_tracking_disabled_or_failing(session, monkeypatch, tmp_path):
    from pitchiq.ml.train import train_family
    off = tracking.Tracker(enabled=False)
    monkeypatch.setattr(tracking, "get_tracker", lambda: off)
    e = train_family(session, PID, "majority", tag="notrack")
    assert "mlflow_run_id" not in e.config and e.validation_metrics["log_loss"] > 0

    broken = tracking.Tracker(uri=f"sqlite:///{tmp_path / 'b' / 'mlflow.db'}", enabled=True)

    def boom(*a, **k):
        raise RuntimeError("store unavailable")
    monkeypatch.setattr(broken.client, "create_run", boom)
    monkeypatch.setattr(tracking, "get_tracker", lambda: broken)
    e2 = train_family(session, PID, "majority", tag="broken")
    assert "mlflow_run_id" not in e2.config and e2.status == "candidate"


def test_api_exposes_run_ids(api, session, tracker):
    p = registry.promoted(session, PID)
    d = api.get(f"/api/models/{p.model_version}").json()
    assert d["mlflow_run_id"] == p.config["mlflow_run_id"]
    lst = api.get(f"/api/pitchers/{PID}/models").json()
    assert all(m["mlflow_run_id"] for m in lst)
    evs = api.get(f"/api/pitchers/{PID}/evaluations").json()
    assert evs and all(ev["mlflow_run_id"] for ev in evs)
    one = api.get(f"/api/evaluations/{evs[0]['evaluation_run_id']}").json()
    assert one["mlflow_run_id"] == evs[0]["mlflow_run_id"]
    lb = api.get(f"/api/pitchers/{PID}/leaderboard").json()
    assert all(r["mlflow_run_id"] and r["model_mlflow_run_id"] for r in lb["rows"])
    st = api.get("/api/tracking").json()
    assert st["enabled"] is True and st["source_of_truth"] == "postgres"
    assert st["tracking_uri"] == get_settings().mlflow_uri
