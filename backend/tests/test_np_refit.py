"""np-refit: evaluation roles, rolling-origin development evaluation, final refit, gates, rollback, retraining
policies, policy replay and the prospective ledger.

Everything that changes the registry runs inside one module session that is rolled back at the end, with the MLflow
mirror pointed at a throwaway store, so the rest of the suite sees the original promotion state.
"""
from datetime import date, datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import select, update

from pitchiq.config import get_settings
from pitchiq.features.store import get_classes
from pitchiq.ml import lifecycle, protocol, refit, registry, replay, tracking
from pitchiq.ml.metrics import log_loss
from pitchiq.models import EvaluationRun, ModelRegistry, PitchPrediction
from pitchiq.models_lifecycle import ModelLifecycle, ModelPromotionEvent

PID = 519242
SMALL_GRIDS = {"majority": [{}], "frequency": [{"alpha": 20.0}], "logreg": [{"C": 0.1}],
               "lightgbm": [{"num_leaves": 7, "min_child_samples": 20}]}


@pytest.fixture(scope="module")
def mp():
    m = pytest.MonkeyPatch()
    m.setattr(protocol, "LGBM_MAX_ROUNDS", 100)
    m.setattr(protocol, "LGBM_CHECKPOINT_STEP", 25)
    m.setattr(get_settings(), "dev_block_games", 1)
    # the fixture has ~740 pitches: relax the sample-size / noise thresholds (the gate logic itself is unchanged)
    m.setattr(get_settings(), "gate_min_training_samples", 100)
    m.setattr(get_settings(), "gate_max_dev_ece", 0.25)
    m.setattr(get_settings(), "gate_max_dev_log_loss_regression", 0.5)
    yield m
    m.undo()


@pytest.fixture(scope="module")
def tracker(pipeline, mp, tmp_path_factory):
    d = tmp_path_factory.mktemp("np_mlflow")
    t = tracking.Tracker(uri=f"sqlite:///{d / 'mlflow.db'}", artifact_root=d / "art", enabled=True)
    mp.setattr(tracking, "get_tracker", lambda: t)
    return t


@pytest.fixture(scope="module")
def world(pipeline, db_url, mp, tracker):
    """refit_all on the fixture DB inside a transaction that is rolled back afterwards."""
    from pitchiq.db import get_sessionmaker
    s = get_sessionmaker(db_url)()
    legacy = registry.promoted(s, PID)
    tracking.mirror_model_entry(legacy)  # so the throwaway store has the legacy production registered
    s.flush()
    out = refit.refit_all(s, PID, grids=SMALL_GRIDS)
    yield {"s": s, "legacy": legacy.model_version, "out": out}
    s.rollback()
    s.close()


# ------------------------------------------------------------------ roles + folds
def test_evaluation_roles_and_labels():
    assert protocol.evaluation_role(2023) == protocol.evaluation_role(2025) == "development"
    assert protocol.evaluation_role(2026) == "retrospective_viewed"
    assert protocol.evaluation_role(2027) == "prospective"
    assert protocol.split_role("train") == protocol.split_role("validation") == "development"
    assert protocol.split_role("test") == "retrospective_viewed"
    assert protocol.role_label("retrospective_viewed") == "retrospective (previously viewed)"
    d = protocol.protocol_dict()
    assert d["roles"]["development"] == [2023, 2024, 2025] and d["split_roles"]["test"] == "retrospective_viewed"


def test_rolling_folds_are_chronological_and_never_split_games(session, mp):
    df = protocol.load_dated_frame(session, PID)
    classes = get_classes(session, PID)
    folds = protocol.rolling_folds(df, classes, block_games=1)
    assert len(folds) == 4  # fixture: 2 games in each of 2024 and 2025
    for f in folds:
        tr, te = df.iloc[f.train_idx], df.iloc[f.eval_idx]
        assert tr.game_date.max() < te.game_date.min()
        assert not set(tr.game_pk) & set(te.game_pk)
        for g in te.game_pk.unique():  # whole games only
            assert (te.game_pk == g).sum() == (df.game_pk == g).sum()
        assert set(te.game_season) <= {2024, 2025} and set(tr.game_season) <= {2023, 2024, 2025}
    with pytest.raises(ValueError):
        protocol.rolling_folds(df, classes, eval_seasons=[2026])  # 2026 can never be a development fold


def test_lightgbm_round_checkpoints_equal_refits_with_that_many_rounds(session):
    df = protocol.load_dated_frame(session, PID)
    classes = get_classes(session, PID)
    f = [x for x in protocol.rolling_folds(df, classes, block_games=1) if not x.skipped][1]
    tr, te = df.iloc[f.train_idx], df.iloc[f.eval_idx]
    big = protocol.make_model("lightgbm", classes, {"num_leaves": 7, "min_child_samples": 20, "n_estimators": 60})
    big.fit(tr, tr["label"])
    small = protocol.make_model("lightgbm", classes, {"num_leaves": 7, "min_child_samples": 20, "n_estimators": 25})
    small.fit(tr, tr["label"])
    P = big.model.predict_proba(big._X(te), num_iteration=25)
    assert np.allclose(P, small.predict_proba(te, calibrated=False), atol=1e-9)


def test_chronological_calibration_uses_only_earlier_folds():
    rng = np.random.default_rng(1)
    Ps = [rng.dirichlet(np.ones(4), 50) for _ in range(3)]
    ys = [rng.integers(0, 4, 50) for _ in range(3)]
    P, temps = protocol.chronological_calibration(Ps, ys)
    assert temps[0] == 1.0 and np.allclose(P[:50], Ps[0])
    from pitchiq.ml.models import TemperatureCalibrator
    assert temps[1] == pytest.approx(TemperatureCalibrator().fit(Ps[0], ys[0]).T)
    assert np.allclose(P.sum(1), 1)


# ------------------------------------------------------------------ final fit
def test_refit_all_registers_final_fits_on_all_data_before_cutoff(world):
    s, out = world["s"], world["out"]
    df = protocol.load_dated_frame(s, PID)
    assert set(out["families"]) == {"majority", "frequency", "logreg", "lightgbm"}
    for fam, d in out["families"].items():
        e = s.get(ModelRegistry, d["model_version"])
        lc = s.get(ModelLifecycle, e.model_version)
        assert lc.training_scope == "deployment_refit" and e.config["training_scope"] == "deployment_refit"
        assert lc.n_train_samples == len(df) == e.dataset_fingerprint["train"]["n"]  # learned 2025 AND 2026
        assert lc.data_cutoff == df.game_date.max() and str(lc.data_cutoff) == e.config["data_cutoff"]
        assert e.config["train_seasons_present"] == [2023, 2024, 2025, 2026]
        assert e.calibration["fit_split"] == "development_rolling_oos"
        assert lc.selected_params == e.config["selected_params"]
        run = s.get(EvaluationRun, lc.dev_evaluation_run_id)
        assert run.kind == "rolling_dev_evaluation" and run.split == "development"
        assert run.metrics["evaluation_role"] == "development"
        assert all(f["end"] < "2026" for f in run.dataset_definition["folds"])
        assert run.metrics["dev_metrics"]["log_loss"] == pytest.approx(e.validation_metrics["log_loss"])
        assert "per_class" in run.metrics["confusion"]
        assert e.status in ("candidate", "promoted")
    lgb = s.get(ModelRegistry, out["families"]["lightgbm"]["model_version"])
    assert lgb.config["params"]["n_estimators"] in protocol.lgbm_checkpoints()


def test_temperature_fit_on_out_of_sample_dev_predictions_not_in_sample(world):
    s = world["s"]
    e = s.get(ModelRegistry, world["out"]["families"]["logreg"]["model_version"])
    df = protocol.load_dated_frame(s, PID)
    res = protocol.rolling_dev_evaluate(df, e.classes, "logreg", grid=[{"C": 0.1}])
    assert e.calibration["pooled_temperature"] == pytest.approx(res.temperature)
    assert e.calibration["val_log_loss_before"] == pytest.approx(log_loss(res.P_raw, res.y))
    assert e.calibration["chrono_calibrated_oos_log_loss"] == pytest.approx(log_loss(res.P_cal, res.y))
    # predetermined rule: the temperature is deployed only if chronological calibration helped on development folds
    applied = log_loss(res.P_cal, res.y) < log_loss(res.P_raw, res.y)
    assert e.calibration["applied"] is applied and res.calibration_applied is applied
    m = registry.load_model(e)
    if applied:
        assert m.calibrator.T == pytest.approx(res.temperature) and e.calibration["temperature"] == m.calibrator.T
    else:
        assert m.calibrator is None and e.calibration["temperature"] == 1.0
    assert e.validation_metrics["log_loss"] == pytest.approx(log_loss(res.P_deployed, res.y))


def test_calibration_rule_deploys_raw_outputs_when_chronological_calibration_hurts():
    rng = np.random.default_rng(3)
    y = rng.integers(0, 3, 200)
    P = rng.dirichlet(np.ones(3), 200)
    base = dict(family="logreg", params={}, grid_results=[], folds=[], y=y, eval_frame=pd.DataFrame(),
                fold_temperatures=[1.0], temperature=0.7, ece_cal=0.1, ece_raw=0.2, seconds=0.0)
    sm = {"brier": 0.6, "top1_accuracy": 0.4}
    hurt = protocol.DevResult(P_raw=P, P_cal=P, summary_raw=dict(log_loss=1.0, **sm), summary_cal=dict(log_loss=1.01, **sm),
                              **base)
    assert not hurt.calibration_applied and hurt.deployed_temperature == 1.0 and hurt.P_deployed is hurt.P_raw
    assert hurt.dev_metrics()["log_loss"] == 1.0 and hurt.dev_metrics()["ece"] == 0.2
    helped = protocol.DevResult(P_raw=P, P_cal=P, summary_raw=dict(log_loss=1.0, **sm), summary_cal=dict(log_loss=0.99, **sm),
                                **base)
    assert helped.calibration_applied and helped.deployed_temperature == 0.7
    assert helped.dev_metrics()["log_loss"] == 0.99 and helped.dev_metrics()["ece"] == 0.1


def test_lifecycle_models_are_mirrored_with_cutoff_tags(world, tracker):
    s = world["s"]
    for d in world["out"]["families"].values():
        e = s.get(ModelRegistry, d["model_version"])
        r = tracker.client.get_run(e.config["mlflow_run_id"])
        assert r.data.tags["pitchiq.training_scope"] == "deployment_refit"
        assert r.data.tags["pitchiq.data_cutoff"] == e.config["data_cutoff"]
        assert r.data.metrics["n_train"] == e.config["n_train"]
        assert r.data.metrics["validation_log_loss"] == pytest.approx(e.validation_metrics["log_loss"])
        run = s.get(EvaluationRun, s.get(ModelLifecycle, e.model_version).dev_evaluation_run_id)
        assert tracker.client.get_run(run.metrics["mlflow_run_id"]).data.tags["pitchiq.split"] == "development"


def test_refits_excluded_from_benchmarks_and_never_scored_in_sample(world):
    s = world["s"]
    fams = registry.latest_by_family(s, PID)
    assert not any(lifecycle.is_lifecycle_model(e) for e in fams.values())
    from pitchiq.ml.evaluate import holdout_benchmark
    with pytest.raises(ValueError, match="in-sample"):
        holdout_benchmark(s, PID, world["out"]["families"]["lightgbm"]["model_version"], "test")


# ------------------------------------------------------------------ gates / promotion / rollback
def test_gated_promotion_keeps_previous_production(world, tracker):
    s, out = world["s"], world["out"]
    promo = out["promotion"]
    assert promo is not None and promo["promoted"], out.get("gate_attempts")
    prod = registry.promoted(s, PID)
    assert prod.model_version == promo["model_version"] and lifecycle.is_lifecycle_model(prod)
    assert s.get(ModelRegistry, world["legacy"]).status == "previous_production"
    names = {g["gate"] for g in promo["gates"]["gates"]}
    assert names == {"artifact_checksum", "probabilities_normalized", "min_training_samples", "dev_ece",
                     "dev_log_loss_vs_production", "data_not_older_than_production", "development_only_provenance"}
    # the legacy production recipe was scored on the same rolling folds for the comparison gate
    assert s.scalars(select(EvaluationRun).where(EvaluationRun.model_version == world["legacy"],
                                                 EvaluationRun.kind == "rolling_dev_evaluation")).first()
    ev = s.scalars(select(ModelPromotionEvent).where(ModelPromotionEvent.model_version == prod.model_version,
                                                     ModelPromotionEvent.action == "promote")).one()
    assert ev.replaced_version == world["legacy"] and ev.gate_results["passed"]
    name = tracker.experiment_name(PID)
    assert tracker.client.get_model_version_by_alias(name, "production").run_id == prod.config["mlflow_run_id"]
    assert tracker.client.get_model_version_by_alias(name, "previous_production").run_id == \
        s.get(ModelRegistry, world["legacy"]).config["mlflow_run_id"]


def test_historical_defaults_use_a_model_trained_before_the_game(world):
    s = world["s"]
    from pitchiq.challenge import create_challenge
    from pitchiq.ml.inference import predict_historical
    from pitchiq.models import PrePitchState
    uid = s.scalars(select(PrePitchState.pitch_uid).where(PrePitchState.split == "test",
                                                          PrePitchState.label.is_not(None))).first()
    r = predict_historical(s, PID, uid, save=False, explain=False)
    assert r["model_version"] == world["legacy"] and r["training_data_includes_game"] is False
    ch = create_challenge(s, PID, 3, seed=1)
    assert ch.model_version == world["legacy"]


def test_failing_gate_leaves_candidate_and_records_results(world, mp):
    s = world["s"]
    cand = s.get(ModelRegistry, world["out"]["families"]["frequency"]["model_version"])
    if cand.status != "candidate":
        cand = s.get(ModelRegistry, world["out"]["families"]["logreg"]["model_version"])
    with pytest.MonkeyPatch.context() as m:
        m.setattr(get_settings(), "gate_max_dev_ece", 0.0)
        res = lifecycle.promote_with_gates(s, cand.model_version, source="test")
    assert res["promoted"] is False and cand.status == "candidate"
    assert {g["gate"] for g in res["gates"]["gates"] if not g["passed"]} >= {"dev_ece"}
    assert s.get(ModelLifecycle, cand.model_version).gate_results["passed"] is False
    assert s.scalars(select(ModelPromotionEvent).where(ModelPromotionEvent.model_version == cand.model_version,
                                                       ModelPromotionEvent.action == "gate_failed")).first()


def test_rollback_restores_previous_production_and_moves_alias(world, tracker):
    s = world["s"]
    prod = registry.promoted(s, PID)
    with pytest.raises(ValueError):
        lifecycle.rollback(s, world["legacy"])  # only production can be rolled back
    rep = lifecycle.rollback(s, prod.model_version)
    assert rep["production"] == world["legacy"]
    assert registry.promoted(s, PID).model_version == world["legacy"]
    assert prod.status == "candidate" and prod.config["rolled_back_to"] == world["legacy"]
    name = tracker.experiment_name(PID)
    assert tracker.client.get_model_version_by_alias(name, "production").run_id == \
        s.get(ModelRegistry, world["legacy"]).config["mlflow_run_id"]
    with pytest.raises(Exception):
        tracker.client.get_model_version_by_alias(name, "previous_production")
    with pytest.raises(LookupError):
        lifecycle.rollback(s, world["legacy"])  # nothing left to roll back to
    # re-promote (gates) so later tests in this module see a lifecycle production model again
    assert lifecycle.promote_with_gates(s, prod.model_version, source="test re-promote")["promoted"]
    assert s.get(ModelRegistry, world["legacy"]).status == "previous_production"


# ------------------------------------------------------------------ scheduled retraining
def test_policy_due_rules():
    now = datetime(2027, 5, 1, tzinfo=timezone.utc)
    g = lambda n: pd.DataFrame({"game_pk": range(n)})  # noqa: E731
    assert refit.policy_due("after_each_start", g(1), now, date(2027, 5, 2), False)[0]
    assert not refit.policy_due("after_each_start", g(0), now, date(2027, 5, 2), False)[0]
    assert not refit.policy_due("every_6_starts", g(5), now, date(2027, 5, 2), False)[0]
    assert refit.policy_due("every_6_starts", g(6), now, date(2027, 5, 2), False)[0]
    assert not refit.policy_due("weekly", g(2), now, date(2027, 5, 7), False)[0]
    assert refit.policy_due("weekly", g(1), now, date(2027, 5, 8), False)[0]
    # manual: the operator running the job is the trigger, but a refit on unchanged data is a no-op unless forced
    assert refit.policy_due("manual", g(1), now, date(2027, 5, 2), False)[0]
    assert not refit.policy_due("manual", g(0), now, date(2027, 5, 2), False)[0]
    assert refit.policy_due("manual", g(0), now, date(2027, 5, 2), True)[0]
    assert refit.policy_due("every_6_starts", g(0), now, date(2027, 5, 2), True)[0]
    with pytest.raises(ValueError):
        refit.policy_due("hourly", g(1), now, date(2027, 5, 2), False)


def test_retrain_job_refits_frozen_recipe_when_due(world, mp):
    s = world["s"]
    df = protocol.load_dated_frame(s, PID)
    first_2026 = df[df.game_season == 2026].game_date.min()
    prod = registry.promoted(s, PID)
    # a production model whose data ended before 2026 (as if deployed before the 2026 season)
    sel = refit.selection_from_entry(s, prod)
    early = refit.final_fit(s, PID, sel, first_2026, tag="early", df=df)
    assert lifecycle.data_cutoff(s, early) < first_2026
    with pytest.MonkeyPatch.context() as m:
        m.setattr(get_settings(), "gate_max_dev_ece", 1.0)
        m.setattr(get_settings(), "gate_min_training_samples", 1)
        lifecycle.promote_with_gates(s, early.model_version, source="test", force=True)
        rep = refit.retrain(s, PID, "every_6_starts", today=date(2027, 1, 1))
        assert rep["due"] is False and rep["new_completed_starts"] == 2
        rep = refit.retrain(s, PID, "after_each_start", today=date(2027, 1, 1))
    assert rep["retrained"] and rep["recipe_inherited_from"] == early.model_version
    new = s.get(ModelRegistry, rep["model_version"])
    lc = s.get(ModelLifecycle, new.model_version)
    assert lc.training_scope == "scheduled_retrain" and lc.retrain_policy == "after_each_start"
    assert lc.data_cutoff == df.game_date.max() and lc.selected_params == sel["params"]
    assert rep["promotion"]["promoted"] and new.status == "promoted"
    assert early.status == "previous_production"
    from pitchiq.jobs.tasks import DESCRIPTIONS, TASKS
    assert "retrain_next_pitch" in TASKS and "retrain_next_pitch" in DESCRIPTIONS


def test_after_ingest_runs_ledger_then_policy_check(world, mp):
    s = world["s"]
    prod = registry.promoted(s, PID)  # the scheduled retrain above: data through the last fixture game
    assert lifecycle.lifecycle_row(s, prod.model_version) is not None
    rep = refit.after_ingest(s, PID, "after_each_start")
    assert "prospective_ledger" in rep and rep["retrain"]["due"] is False and not rep["retrain"]["retrained"]
    assert rep["retrain"]["new_completed_starts"] == 0
    assert "disabled" in refit.after_ingest(s, PID, "manual")["retrain"]["skipped"]
    n_models = s.scalar(select(__import__("sqlalchemy").func.count()).select_from(ModelRegistry))
    legacy = s.get(ModelRegistry, world["legacy"])
    with pytest.MonkeyPatch.context() as m:  # season-split production: never an implicit re-selection
        m.setattr(registry, "promoted", lambda session, pid: legacy)
        assert "season-split" in refit.after_ingest(s, PID, "after_each_start")["retrain"]["skipped"]
    assert s.scalar(select(__import__("sqlalchemy").func.count()).select_from(ModelRegistry)) == n_models
    assert registry.promoted(s, PID).model_version == prod.model_version


# ------------------------------------------------------------------ replay
def test_policy_cutoffs():
    games = pd.DataFrame({"game_date": [date(2025, 4, 1) + timedelta(days=5 * i) for i in range(13)]})
    after = replay.policy_cutoffs("after_each_start", games)
    assert after == list(games.game_date)
    assert set(replay.policy_cutoffs("manual", games)) == {date(2025, 4, 1)}
    six = replay.policy_cutoffs("every_6_starts", games)
    assert six[5] == date(2025, 4, 1) and six[6] == games.game_date[6] and six[12] == games.game_date[12]
    wk = replay.policy_cutoffs("weekly", games)
    assert all(c <= d and c.weekday() in (0, date(2025, 4, 1).weekday()) for c, d in zip(wk, games.game_date))


def test_policy_replay_stores_runs_and_manual_matches_a_single_fit(world):
    s = world["s"]
    runs = replay.replay_policies(s, PID, families=("logreg",), seasons=(2025, 2026))
    assert [r.kind for r in runs] == ["retraining_policy_replay"] * 2
    r25, r26 = runs
    assert r25.metrics["evaluation_role"] == "development" and r25.split == "validation"
    assert r26.metrics["evaluation_label"] == "retrospective (previously viewed)" and r26.split == "test"
    pol = r26.metrics["policies"]
    assert set(pol) == set(refit.POLICIES)
    assert pol["manual"]["n_refits"] == 0 and pol["after_each_start"]["n_refits"] == 1
    df = protocol.load_dated_frame(s, PID)
    te = df[df.game_season == 2026]
    tr = df[df.game_date < te.game_date.min()]
    params = s.get(ModelLifecycle, r26.model_version).selected_params
    m = protocol.make_model("logreg", get_classes(s, PID), params).fit(tr, tr["label"])
    assert pol["manual"]["log_loss"] == pytest.approx(log_loss(m.predict_proba(te, calibrated=False),
                                                               m.y_index(te["label"])))


# ------------------------------------------------------------------ prospective ledger
def test_prospective_ledger_uses_only_models_trained_before_the_game(world):
    s = world["s"]
    df = protocol.load_dated_frame(s, PID)
    first_2026 = df[df.game_season == 2026].game_date.min()
    rep = __import__("pitchiq.ml.prospective", fromlist=["x"]).record_prospective(s, PID)
    assert rep["recorded_pitches"] == 0  # every model was created after every fixture game
    prod = registry.promoted(s, PID)
    sel = refit.selection_from_entry(s, prod)
    pre = refit.final_fit(s, PID, sel, first_2026, tag="pre2026", df=df)
    lifecycle.promote_with_gates(s, pre.model_version, source="test", force=True)
    before = datetime.combine(first_2026 - timedelta(days=3), datetime.min.time(), tzinfo=timezone.utc)
    s.execute(update(ModelRegistry).where(ModelRegistry.model_version == pre.model_version)
              .values(created_at=before - timedelta(days=1)))
    s.execute(update(ModelPromotionEvent).where(ModelPromotionEvent.model_version == pre.model_version)
              .values(created_at=before))
    s.expire_all()
    from pitchiq.ml.prospective import prospective_metrics, record_prospective
    rep = record_prospective(s, PID)
    te = df[df.game_season == 2026]
    assert rep["recorded_games"] == te.game_pk.nunique() and rep["recorded_pitches"] == len(te)
    rows = s.scalars(select(PitchPrediction).where(PitchPrediction.context == "prospective")).all()
    assert {r.model_version for r in rows} == {pre.model_version}
    assert {r.pitch_uid for r in rows} == set(te.pitch_uid)
    P = registry.load_model(s.get(ModelRegistry, pre.model_version)).predict_proba(te.drop(columns=["label"]))
    got = {r.pitch_uid: r.probabilities for r in rows}
    cls = pre.classes
    assert np.allclose([[got[u][c] for c in cls] for u in te.pitch_uid], P)
    assert record_prospective(s, PID)["recorded_pitches"] == 0  # idempotent
    met = prospective_metrics(s, PID)
    y = np.array([cls.index(a) for a in te.label])
    assert met["n"] == len(te) and met["summary"]["log_loss"] == pytest.approx(log_loss(P, y))


# ------------------------------------------------------------------ API (read-only on the committed state)
def test_lifecycle_api(api):
    r = api.get("/api/protocol").json()
    assert r["labels"]["retrospective_viewed"] == "retrospective (previously viewed)"
    ov = api.get(f"/api/pitchers/{PID}/lifecycle").json()
    assert ov["production"]["stage"] == "production" and ov["archive_policy"].startswith("keep")
    assert api.get(f"/api/pitchers/{PID}/prospective").json()["n"] == 0
    models = api.get(f"/api/pitchers/{PID}/models").json()
    assert all("stage" in m for m in models)
    cand = next(m for m in models if m["status"] == "candidate")
    assert api.post(f"/api/models/{cand['model_version']}/rollback").status_code == 422
    lb = api.get(f"/api/pitchers/{PID}/leaderboard").json()
    assert lb["evaluation_label"] == "retrospective (previously viewed)"
    ev = api.get(f"/api/pitchers/{PID}/evaluations").json()
    assert all("evaluation_role" in e for e in ev)
