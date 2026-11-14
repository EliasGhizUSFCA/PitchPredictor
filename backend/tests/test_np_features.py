"""np-features: v2 next-pitch features, MLB-wide batter priors, time-decay weights, simulator semantics, ablation.

Leakage tests (truncation invariance, future perturbation) cover every v2 feature, with the batter priors loaded from
fixtures cut from real Savant responses (tests/make_batter_lb_fixtures.py). Registry changes run in a transaction
that is rolled back, with a throwaway MLflow store.
"""
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import select

from pitchiq import batter_priors as bp
from pitchiq.config import get_settings
from pitchiq.features import v2 as fv2
from pitchiq.features.build import PRE_PITCH_COLUMNS, build_feature_rows
from pitchiq.features.schema import (FEATURE_VERSION, FORBIDDEN_SOURCE_COLUMNS, default_model_features,
                                     feature_list)
from pitchiq.features.store import get_classes, load_feature_frame, load_pitch_frame
from pitchiq.http import SourceUnavailable
from pitchiq.ml import decay, protocol, refit, tracking
from pitchiq.models import PrePitchState
from pitchiq.models_batter_priors import BatterPriorAsOf, BatterPriorMeta

from conftest import FixtureClient

PID = 519242
CLASSES = ["FF", "SL", "CH", "SI"]
LB = Path(__file__).parent / "fixtures" / "batter_lb"


class LBClient(FixtureClient):
    """FixtureClient + recorded batter leaderboards."""

    def get(self, url, params=None, *, namespace, suffix=".txt", use_cache=True, retries=3):
        if params and params.get("type") == "batter" and "pitch-arsenal-stats" in url:
            name = f"pitch_type_{params['year']}"
        elif params and params.get("type") == "batter" and "leaderboard/custom" in url:
            name = f"custom_{params['year']}"
        elif params and params.get("player_type") == "batter" and "statcast_search" in url:
            name = f"platoon_{params['pitcher_throws']}_{params['hfSea'].strip('|')}"
        else:
            return super().get(url, params, namespace=namespace, suffix=suffix, use_cache=use_cache, retries=retries)
        f = LB / f"{name}.csv"
        if not f.exists():
            raise SourceUnavailable(f"no fixture {name}")
        return f.read_text()


@pytest.fixture(scope="module")
def priors_db(pipeline, db_url):
    """Load the recorded leaderboards + as-of priors and rebuild v1/v2 states (committed: later tests read them)."""
    from pitchiq.db import session_scope
    from pitchiq.features.store import build_and_store
    with session_scope(db_url) as s:
        rep = bp.refresh(s, PID, client=LBClient())
        feats = build_and_store(s, PID)
    return {"refresh": rep, "features": feats}


@pytest.fixture(scope="module")
def frame(priors_db, db_url):
    from pitchiq.db import session_scope
    with session_scope(db_url) as s:
        df = load_pitch_frame(s, PID)
        pri = bp.load_priors(s, PID, CLASSES)
    return df, pri


def _feats(df, pri):
    return {r["pitch_uid"]: r["features"] for r in build_feature_rows(df, CLASSES, batter_priors=pri)}


def _same(a: dict, b: dict) -> bool:
    if a.keys() != b.keys():
        return False
    for k in a:
        x, y = a[k], b[k]
        if isinstance(x, float) and isinstance(y, float):
            if not (x == y or (np.isnan(x) and np.isnan(y))):
                return False
        elif x != y:
            return False
    return True


# ------------------------------------------------------------------ schema / versions
def test_v2_schema_is_superset_and_has_no_forbidden_inputs():
    v1 = {f.name for f in feature_list(CLASSES, "v1")}
    v2 = {f.name for f in feature_list(CLASSES, "v2")}
    assert v1 < v2 and FEATURE_VERSION == "v2"
    assert not v2 & FORBIDDEN_SOURCE_COLUMNS
    assert not set(PRE_PITCH_COLUMNS) & FORBIDDEN_SOURCE_COLUMNS and "catcher_id" in PRE_PITCH_COLUMNS
    groups = {f.group for f in feature_list(CLASSES, "v2")} - {f.group for f in feature_list(CLASSES, "v1")}
    assert groups == set(fv2.GROUPS)
    m = default_model_features(CLASSES, "v2")
    assert "prev1_speed" not in m and "season" not in m and "prev1_velocity" in m
    assert "prev1_speed" in default_model_features(CLASSES, "v1")


def test_v1_rows_unchanged_and_v2_rows_restrict_to_v1(priors_db, session):
    v1 = {r.pitch_uid: r.features for r in session.execute(select(PrePitchState.pitch_uid, PrePitchState.features)
                                                           .where(PrePitchState.feature_version == "v1")).all()}
    v2 = {r.pitch_uid: r.features for r in session.execute(select(PrePitchState.pitch_uid, PrePitchState.features)
                                                           .where(PrePitchState.feature_version == "v2")).all()}
    assert v1.keys() == v2.keys() and len(v1) > 700
    keys = [f.name for f in feature_list(CLASSES, "v1")]
    assert all(v1[u] == {k: v2[u][k] for k in keys} for u in v1)
    assert set(priors_db["features"]["versions"]) == {"v1", "v2"}


# ------------------------------------------------------------------ leakage
def test_v2_truncation_invariance(frame):
    df, pri = frame
    full = _feats(df, pri)
    rng = np.random.default_rng(1)
    for i in rng.choice(len(df), size=40, replace=False):
        trunc = df.iloc[: i + 1].copy()
        last = trunc.index[-1]
        trunc.loc[last, "pitch_type"] = "SI" if trunc.loc[last, "pitch_type"] != "SI" else "FF"
        trunc.loc[last, ["release_speed", "plate_x", "plate_z", "zone"]] = [55.0, 3.3, -1.0, 14]
        trunc.loc[last, "description"] = "swinging_strike"
        uid = df.iloc[i]["pitch_uid"]
        assert _same(_feats(trunc, pri)[uid], full[uid])


def test_v2_future_perturbation_does_not_change_past(frame):
    df, pri = frame
    full = _feats(df, pri)
    cut = len(df) // 2
    pert = df.copy()
    idx = pert.index[cut:]
    pert.loc[idx, "pitch_type"] = "CH"
    pert.loc[idx, ["release_speed", "plate_x", "plate_z", "zone"]] = [70.0, 0.0, 2.5, 5]
    pert.loc[idx, "description"] = "swinging_strike"
    pert.loc[idx, "catcher_id"] = 999999
    pf = _feats(pert, pri)
    for uid in df.iloc[:cut]["pitch_uid"]:
        assert _same(pf[uid], full[uid])


def test_batter_priors_use_only_earlier_seasons(priors_db, db_url):
    """Changing season-s leaderboards never changes the priors used in season s (or earlier)."""
    from pitchiq.db import get_sessionmaker
    from pitchiq.models_batter_priors import BatterDisciplineSeason, BatterPitchTypeSeason
    s = get_sessionmaker(db_url)()
    try:
        before = {(r.batter_id, r.as_of_season): r.features for r in s.scalars(select(BatterPriorAsOf))}
        for m in (BatterPitchTypeSeason, BatterDisciplineSeason):
            for r in s.scalars(select(m).where(m.season == 2025)):
                r.whiff_pct = 0.99
                r.woba = 0.9
        s.flush()
        bp.build_priors(s, [2023, 2024, 2025, 2026], ["L"], CLASSES)
        after = {(r.batter_id, r.as_of_season): r.features for r in s.scalars(select(BatterPriorAsOf))}
        same = [k for k in before if k[1] <= 2025]
        assert same and all(after[k] == before[k] for k in same)
        changed = [k for k in before if k[1] == 2026 and after[k] != before[k]]
        assert changed  # 2026 priors do use 2025
        meta = s.get(BatterPriorMeta, (2025, "L"))
        assert meta.source_seasons == [2022, 2023, 2024]
    finally:
        s.rollback()
        s.close()


# ------------------------------------------------------------------ explicit flags + missing data
def test_explicit_flags_and_previous_pitch_detail(frame):
    df, pri = frame
    rows = build_feature_rows(df, CLASSES, batter_priors=pri)
    d = pd.DataFrame([{"uid": r["pitch_uid"], "label": r["label"], **r["features"]} for r in rows])
    d = d.merge(df[["pitch_uid", "game_pk", "at_bat_number", "pitch_number", "release_speed", "plate_x"]],
                left_on="uid", right_on="pitch_uid")
    first = d.groupby("game_pk").head(1)
    assert (first.is_first_pitch_of_game == 1).all() and (first.has_previous_pitch == 0).all()
    assert (first.prev1_zone == "NONE").all() and first.prev1_velocity.isna().all()
    rest = d.drop(first.index)
    assert (rest.is_first_pitch_of_game == 0).all() and (rest.has_previous_pitch == 1).all()
    assert ((d.pitch_number == 1) == (d.is_first_pitch_of_plate_appearance == 1)).all()
    assert (rest.previous_pitch_velocity_missing == 0).all() and (rest.previous_pitch_location_missing == 0).all()
    # prev1_velocity / plate_x equal the previous thrown pitch's measurements
    for _, g in d.groupby("game_pk"):
        thrown = g[g.label.notna()]
        prev_speed = thrown.release_speed.shift(1)
        sub = g.loc[thrown.index[1:]]
        assert np.allclose(sub.prev1_velocity.astype(float), prev_speed.iloc[1:].astype(float))
        assert np.allclose(sub.prev1_plate_x.astype(float), thrown.plate_x.shift(1).iloc[1:].astype(float))


def test_missing_velocity_and_location_are_flagged_and_imputed_from_prior_games(frame):
    df, pri = frame
    games = df.game_pk.unique()
    g = df[df.game_pk == games[3]]
    thrown = g[g.pitch_type.notna()]
    target = thrown.index[5]
    nxt = thrown.index[6]
    pert = df.copy()
    pert.loc[target, ["release_speed", "plate_x", "zone"]] = [np.nan, np.nan, np.nan]
    f = _feats(pert, pri)[df.loc[nxt, "pitch_uid"]]
    assert f["previous_pitch_velocity_missing"] == 1 and f["previous_pitch_location_missing"] == 1
    assert f["prev1_zone"] == "UNK" and f["prev1_plate_x"] is None and f["prev1_in_zone"] is None
    prior = df[df.game_date < df.loc[target, "game_date"]]
    t = df.loc[target, "pitch_type"]
    expect = prior[prior.pitch_type == t].release_speed.mean()
    assert f["prev1_velocity"] == pytest.approx(expect)
    assert f["prev1_velo_vs_type_mean"] is None
    assert f["prev1_speed"] is None  # v1 semantics unchanged (missing)


def test_missing_batter_priors_and_catcher():
    from pitchiq.features.build import GameState, History, PAState, make_features
    ctx = {"stand": "R", "bat_order_slot": 1, "balls": 0, "strikes": 0, "inning": 1, "inning_topbot": "Top",
           "outs_when_up": 0, "on_1b": None, "on_2b": None, "on_3b": None, "fld_score": 0, "bat_score": 0,
           "at_bat_number": 1, "pitch_number": 1, "n_thruorder_pitcher": 1, "season": 2025, "days_rest": 5,
           "batter": 1, "game_date": date(2025, 4, 1), "catcher_id": None}
    f = make_features(ctx, History(), GameState(), PAState(), CLASSES)
    assert f["blb_missing"] == 1 and f["blb_whiff_CH"] is None and f["catcher_id"] == "NONE" and f["cat_n"] == 0
    assert f["rec_g1_n"] == 0 and f["rec_ewm_all_CH"] == pytest.approx(0.25)
    lk = bp.BatterPriors("L", rows={(7, 2025): {"blb_whiff_FF": 0.3}},
                         league={2025: {"blb_whiff_FF": 0.2, "blb_missing": 1, "blb_log_pa": 0.0}}, classes=CLASSES)
    assert lk.lookup(8, 2025)["blb_whiff_FF"] == 0.2 and lk.lookup(8, 2025)["blb_missing"] == 1
    assert lk.lookup(7, 2025)["blb_whiff_FF"] == 0.3
    assert lk.lookup(7, 2030)["blb_missing"] == 1 and lk.lookup(7, 2030)["blb_whiff_FF"] is None


def test_recency_windows_match_direct_counts(frame):
    df, pri = frame
    rows = {r["pitch_uid"]: r["features"] for r in build_feature_rows(df, CLASSES, batter_priors=pri)}
    games = df.groupby("game_pk").game_date.first().sort_values()
    gpk = games.index[5]
    first = df[df.game_pk == gpk].iloc[0]
    f = rows[first.pitch_uid]
    prev_games = list(games.index[:5])
    last = df[df.game_pk == prev_games[-1]]
    cnt = last.pitch_type.value_counts()
    n = int(cnt.sum())
    assert f["rec_g1_n"] == n
    prior_all = df[df.game_pk.isin(prev_games)].pitch_type.value_counts(normalize=True)
    exp = (cnt.get("CH", 0) + fv2.K_REC_ALL * prior_all.get("CH", 0)) / (n + fv2.K_REC_ALL)
    assert f["rec_g1_all_CH"] == pytest.approx(exp)
    w = sum(0.5 ** ((games.loc[gpk] - games.loc[g]).days / fv2.EWM_HALF_LIFE_DAYS) *
            (df[df.game_pk == g].pitch_type.notna().sum()) for g in prev_games)
    assert f["rec_ewm_n"] == pytest.approx(w)
    season_games = [g for g in prev_games if games.loc[g].year == games.loc[gpk].year]
    assert f["rec_season_n"] == df[df.game_pk.isin(season_games)].pitch_type.notna().sum()


# ------------------------------------------------------------------ empirical Bayes
def test_eb_strength_recovers_known_shrinkage():
    rng = np.random.default_rng(0)
    p = rng.beta(30, 70, size=3000)  # true-talent sd ~ 0.045 around 0.3
    n = rng.integers(20, 600, size=3000).astype(float)
    x = rng.binomial(n.astype(int), p) / n
    mu, k, diag = bp.eb_strength(x, n)
    assert mu == pytest.approx(0.3, abs=0.01)
    assert 60 < k < 160  # beta(30,70): sigma^2/tau^2 = a+b+1 = 101
    shrunk = bp._shrink(np.array([0.9, np.nan]), np.array([5.0, 0.0]), mu, k)
    assert shrunk[1] == pytest.approx(mu) and mu < shrunk[0] < 0.4


def test_priors_report_and_league_fallback(priors_db, session):
    rep = priors_db["refresh"]
    assert rep["leaderboards"]["sources"]["failures"] == []
    assert rep["priors"]["2025-L"]["source_seasons"] == [2022, 2023, 2024]
    pri = bp.load_priors(session, PID, CLASSES)
    f = pri.lookup(-1, 2025)
    meta = session.get(BatterPriorMeta, (2025, "L"))
    assert f["blb_missing"] == 1 and f["blb_whiff_SL"] == pytest.approx(meta.league_means["blb_whiff_SL"])
    known = session.scalars(select(BatterPriorAsOf).where(BatterPriorAsOf.as_of_season == 2025)).first()
    g = pri.lookup(known.batter_id, 2025)
    assert g["blb_missing"] == 0 and 0 < g["blb_whiff_FF"] < 1


# ------------------------------------------------------------------ decay
def test_decay_weights_halve_every_half_life_and_refuse_future_dates():
    ref = date(2025, 6, 1)
    dates = [ref - timedelta(days=d) for d in (0, 30, 60, 90)]
    w = decay.decay_weights(dates, ref, 30.0)
    assert w.mean() == pytest.approx(1.0) and np.allclose(w[1:] / w[:-1], 0.5)
    assert decay.decay_weights(dates, ref, None) is None
    with pytest.raises(ValueError):
        decay.decay_weights([ref + timedelta(days=1)], ref, 30.0)


def test_decay_changes_fits_through_sample_weights(session, priors_db):
    df = protocol.load_dated_frame(session, PID)
    ref = max(df.game_date) + timedelta(days=1)
    plain = decay.fit(protocol.make_model("majority", CLASSES, {}), df, ref)
    dec = decay.fit(protocol.make_model("majority", CLASSES, {decay.DECAY_KEY: 30.0}), df, ref)
    assert dec.decay_half_life_days == 30.0 and plain.decay_half_life_days is None
    recent = df[df.game_date == max(df.game_date)].label.value_counts(normalize=True)
    top = recent.index[0]
    j = CLASSES.index(top)
    assert abs(dec.prior[j] - recent[top]) < abs(plain.prior[j] - recent[top])
    for fam in ("frequency", "logreg", "lightgbm"):
        p = {"lightgbm": {"num_leaves": 7, "min_child_samples": 20, "n_estimators": 30}}.get(fam, {})
        a = decay.fit(protocol.make_model(fam, CLASSES, p), df, ref).predict_proba(df.head(50))
        b = decay.fit(protocol.make_model(fam, CLASSES, {**p, decay.DECAY_KEY: 30.0}), df, ref).predict_proba(
            df.head(50))
        assert np.allclose(a.sum(1), 1) and not np.allclose(a, b)


# ------------------------------------------------------------------ refit with decay + ablation (rolled back)
@pytest.fixture(scope="module")
def mp():
    m = pytest.MonkeyPatch()
    m.setattr(protocol, "LGBM_MAX_ROUNDS", 50)
    m.setattr(protocol, "LGBM_CHECKPOINT_STEP", 25)
    m.setattr(get_settings(), "dev_block_games", 1)
    m.setattr(get_settings(), "gate_min_training_samples", 100)
    m.setattr(get_settings(), "gate_max_dev_ece", 0.25)
    m.setattr(get_settings(), "gate_max_dev_log_loss_regression", 0.5)
    yield m
    m.undo()


@pytest.fixture(scope="module")
def world(priors_db, db_url, mp, tmp_path_factory):
    from pitchiq.db import get_sessionmaker
    d = tmp_path_factory.mktemp("npf_mlflow")
    t = tracking.Tracker(uri=f"sqlite:///{d / 'mlflow.db'}", artifact_root=d / "art", enabled=True)
    mp.setattr(tracking, "get_tracker", lambda: t)
    s = get_sessionmaker(db_url)()
    out = refit.refit_all(s, PID, families=("logreg", "lightgbm"), promote=False,
                          grids={"logreg": [{"C": 0.1}], "lightgbm": [{"num_leaves": 7, "min_child_samples": 20}]},
                          decay_half_lives=(None, 60.0))
    yield {"s": s, "out": out, "tracker": t}
    s.rollback()
    s.close()


def test_refit_selects_half_life_on_dev_folds_and_stores_it(world):
    s = world["s"]
    from pitchiq.ml import registry
    from pitchiq.models_lifecycle import ModelLifecycle
    for fam in ("logreg", "lightgbm"):
        info = world["out"]["families"][fam]
        assert decay.DECAY_KEY in info["selected_params"]
        e = registry.get_entry(s, PID, info["model_version"])
        assert e.feature_version == "v2" and "-v2-" in e.model_version
        assert "prev1_velocity" in e.feature_schema["used_features"]
        lc = s.get(ModelLifecycle, e.model_version)
        assert lc.selected_params[decay.DECAY_KEY] == info["selected_params"][decay.DECAY_KEY]
        m = registry.load_model(e)
        assert m.decay_half_life_days == info["selected_params"][decay.DECAY_KEY]
        stages = {r.get("stage") for r in e.config["grid_results"]}
        assert stages == {"hyperparameters (no decay)", "decay half-life (hyperparameters fixed)"}
        hl = [r["params"][decay.DECAY_KEY] for r in e.config["grid_results"] if r["stage"].startswith("decay")]
        assert hl == [None, 60.0]


def test_frozen_v1_recipe_is_retrained_on_v1_features(world):
    """A v1 lifecycle recipe keeps its v1 feature list (selection_from_entry -> final_fit)."""
    s = world["s"]
    from pitchiq.ml import registry
    e = registry.get_entry(s, PID, world["out"]["families"]["logreg"]["model_version"])
    sel = refit.selection_from_entry(s, e)
    v1feats = default_model_features(CLASSES, "v1")
    sel.update(features=v1feats, feature_version="v1")
    e1 = refit.final_fit(s, PID, sel, tag="t-v1")
    assert e1.feature_version == "v1" and e1.feature_schema["used_features"] == v1feats
    assert "-v1-" in e1.model_version


def test_feature_ablation_reports_per_class_scores(world):
    from pitchiq.ml import feature_ablation as fa
    s = world["s"]
    out = fa.run_ablation(s, PID, family="logreg",
                          configs=["v1_features", "v2_without_batter_league", "v1_plus_catcher"])
    m = out["metrics"]
    names = [r["config"] for r in m["results"]]
    assert names[0] in ("recipe", "v2_all") and m["reference"] == names[0]
    assert {"v1_features", "v2_without_batter_league", "v1_plus_catcher"} <= set(names)
    ref = m["results"][0]
    assert ref["delta_log_loss"] == 0
    for r in m["results"]:
        for c in ("CH", "SI"):
            pc = r["per_class"][c]
            assert pc["log_loss"] > 0 and 0 <= pc["brier_ovr"] <= 1 and "delta_brier_ovr" in pc
    from pitchiq.models import EvaluationRun
    run = s.get(EvaluationRun, out["evaluation_run_id"])
    assert run.kind == fa.KIND and run.split == "development" and run.metrics.get("mlflow_run_id")
    rid = run.metrics["mlflow_run_id"]
    got = world["tracker"].client.get_run(rid).data.metrics
    assert "ablation/v2_all/CH_log_loss" in got


def test_per_class_scores_definition():
    from pitchiq.ml.feature_ablation import per_class_scores
    P = np.array([[0.5, 0.5], [0.2, 0.8], [0.9, 0.1]])
    y = np.array([0, 1, 1])
    d = per_class_scores(P, y, ["A", "B"])
    assert d["B"]["log_loss"] == pytest.approx(-(np.log(0.8) + np.log(0.1)) / 2)
    assert d["A"]["brier_ovr"] == pytest.approx(((0.5 - 1) ** 2 + 0.2 ** 2 + 0.9 ** 2) / 3)
    assert d["B"]["recall"] == pytest.approx(0.5)


# ------------------------------------------------------------------ simulator semantics
def test_simulator_v2_features_match_history_for_the_same_state(world, frame):
    """A scenario that reproduces a historical pre-pitch state yields the same v2 features as history."""
    from pitchiq.ml import inference, registry
    s = world["s"]
    e = registry.get_entry(s, PID, world["out"]["families"]["lightgbm"]["model_version"])
    df, _ = frame
    stored = load_feature_frame(s, PID, None, "v2", labeled_only=False).set_index("pitch_uid")
    games = df.groupby("game_pk").game_date.first().sort_values()
    g = df[df.game_pk == games.index[4]].reset_index(drop=True)
    inference._history_cached.cache_clear()
    keys = [f.name for f in feature_list(CLASSES, "v2") if f.group in
            {"explicit_flags", "prev_pitch_detail", "catcher", "recency_overall", "recency_count",
             "recency_transition", "batter_league", "batter_platoon"}]
    keys = [k for k in keys if not k.startswith("prev2_")]
    for i in (0, 1):
        row = g.iloc[i]
        prev = []
        if i == 1:
            p = g.iloc[0]
            prev = [inference.PrevPitch(type=p.pitch_type, result="ball" if p.description == "ball" else
                                        __import__("pitchiq.pitch_types", fromlist=["x"]).result_category(p.description),
                                        speed=float(p.release_speed), same_pa=bool(p.at_bat_number == row.at_bat_number),
                                        plate_x=float(p.plate_x), plate_z=float(p.plate_z), zone=int(p.zone))]
        st = inference.SimState(stand=row.stand, bat_order_slot=int(row.bat_order_slot), batter_id=int(row.batter),
                                balls=int(row.balls), strikes=int(row.strikes), inning=int(row.inning),
                                outs=int(row.outs_when_up), pitch_count_before=i, previous_pitches=prev,
                                as_of_date=row.game_date, catcher_id=int(row.catcher_id),
                                is_home=row.inning_topbot == "Top")
        mix, _ = inference.sim_feature_mixture(s, e, st)
        sim = mix[0][1]
        hist = stored.loc[row.pitch_uid]
        for k in keys:
            a, b = sim[k], hist[k]
            if isinstance(a, float) or isinstance(b, float):
                assert (a is None and pd.isna(b)) or a == pytest.approx(b), k
            else:
                assert a == b, k
    # an unspecified previous velocity / location is flagged exactly like an unmeasured one in history
    st = inference.SimState(pitch_count_before=1, previous_pitches=[inference.PrevPitch(type="FF", same_pa=False)],
                            as_of_date=games.iloc[4])
    mix, notes = inference.sim_feature_mixture(s, e, st)
    f = mix[0][1]
    assert f["previous_pitch_velocity_missing"] == 1 and f["previous_pitch_location_missing"] == 1
    prior = df[(df.game_date < games.iloc[4]) & (df.pitch_type == "FF")].release_speed.mean()
    assert f["prev1_velocity"] == pytest.approx(prior) and f["prev1_zone"] == "UNK"
    assert any("velocity" in n for n in notes["imputed"])


def test_simulator_predicts_with_v2_model_and_marginalizes(world):
    from pitchiq.ml import inference, registry
    s = world["s"]
    e = registry.get_entry(s, PID, world["out"]["families"]["lightgbm"]["model_version"])
    st = inference.SimState(balls=1, strikes=1, pitch_count_before=20, inning=3)
    r = inference.predict_scenario(s, PID, st, model_version=e.model_version, persist_trace=False)
    assert abs(sum(d["probability"] for d in r["predictions"]) - 1) < 1e-6
    combos = r["assumptions"]["marginalized_over"]["combinations"]
    assert combos and "detail" in combos[0]


def test_api_batter_priors_and_schema(priors_db, api, session):
    known = session.scalars(select(BatterPriorAsOf).where(BatterPriorAsOf.as_of_season == 2025)).first()
    r = api.get(f"/api/batters/{known.batter_id}/priors", params={"season": 2025})
    assert r.status_code == 200
    body = r.json()
    assert body["as_of_season"] == 2025 and body["source_seasons"] == [2022, 2023, 2024]
    assert body["features"]["blb_missing"] == 0
    r = api.get("/api/features/schema", params={"version": "v2"})
    assert r.status_code == 200 and r.json()["feature_version"] == "v2"
    assert "explicit_flags" in {f["group"] for f in r.json()["features"]}


def test_jobs_and_cli_registered():
    from pitchiq.jobs.tasks import DESCRIPTIONS, TASKS
    assert {"batter_priors", "feature_ablation"} <= set(TASKS) and "batter_priors" in DESCRIPTIONS
    from pitchiq import cli
    with pytest.raises(SystemExit):
        cli.main(["feature-ablation", "--help"])
