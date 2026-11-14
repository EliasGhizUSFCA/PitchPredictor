"""Aging forecast engine (System B): methods, leakage guards, metrics, selection, persistence, API, MLflow mirror.

Data: real rows cut from the aging dataset (tests/make_aging_model_fixtures.py -> tests/fixtures/aging_model):
Chris Sale + 119 other pitchers, four-seam / slider pitch types and pitcher-season stats, 2015-2026. Pure-function
tests build the panel in memory; the end-to-end test inserts the rows into the rolled-back test session.
"""
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import func, select, text

from pitchiq.aging_model import backtest as bt
from pitchiq.aging_model import methods as mt
from pitchiq.aging_model.panel import Target, build_panel, target_frame, targets

FIX = Path(__file__).parent / "fixtures" / "aging_model"
PID = 519242
FAST = ("no_change", "marcel", "age_curve", "age_curve_ipw", "lgbm_quantile", "eb_kalman")


def _frames():
    pss = pd.read_csv(FIX / "pss.csv")
    pss["quality_flags"] = pss["quality_flags"].fillna("").map(lambda s: [x for x in str(s).split("|") if x])
    psps = pd.read_csv(FIX / "psps.csv")
    bio = pd.read_csv(FIX / "bio.csv", parse_dates=["birth_date"])
    return pss, psps, bio


@pytest.fixture(scope="module")
def panel():
    pss, psps, bio = _frames()
    return build_panel(pss, psps, bio)


def _t(key):
    return next(t for t in targets(["four_seam", "slider"]) if t.key == key)


def test_panel_family_aggregation_and_normalisation(panel):
    pss, psps, bio = _frames()
    # Sale 2026 four-seam velocity is the stored leaderboard value (single code FF in the family)
    v = panel[(panel.pitcher_id == PID) & (panel.season == 2026) & (panel.unit == "four_seam") &
              (panel.metric == "avg_speed_mph")]
    raw = psps[(psps.pitcher_id == PID) & (psps.season == 2026) & (psps.pitch_type == "FF")]
    assert float(v["value"].iloc[0]) == pytest.approx(float(raw["avg_speed_mph"].iloc[0]))
    # slider family = pitch-weighted mean over SL/ST/SV rows
    multi = psps[psps.pitch_family == "slider"].groupby(["pitcher_id", "season"]).filter(lambda g: len(g) > 1)
    if len(multi):
        g = multi.groupby(["pitcher_id", "season"]).get_group(next(iter(multi.groupby(["pitcher_id", "season"]).groups)))
        exp = np.average(g["avg_speed_mph"], weights=g["pitches"])
        row = panel[(panel.pitcher_id == g.pitcher_id.iloc[0]) & (panel.season == g.season.iloc[0]) &
                    (panel.unit == "slider") & (panel.metric == "avg_speed_mph")]
        assert float(row["value"].iloc[0]) == pytest.approx(exp)
    # release side: arm side positive for both hands
    rx = panel[(panel.metric == "release_pos_x_armside_ft") & panel.qualified]
    assert (rx["value"] > 0).mean() > 0.9
    # ages are exact June 30 ages; 2020 thresholds scaled
    s20 = panel[(panel.season == 2020) & (panel.unit == "ALL")]
    assert ((s20["workload"] >= 185) == s20["qualified"]).all()


def test_rcs_is_linear_beyond_outer_knots():
    x = np.arange(37.0, 45.0)
    b = mt.rcs(x)
    assert np.allclose(np.diff(b, 2, axis=0), 0, atol=1e-9)
    x2 = np.arange(18.0, 23.0)
    assert np.allclose(np.diff(mt.rcs(x2), 2, axis=0), 0, atol=1e-9)


def test_pinball_and_score():
    df = pd.DataFrame({"actual": [1.0, 2.0], "q10": [0.0, 2.5], "q50": [1.0, 3.0], "q90": [2.0, 3.5]})
    s = bt.score(df)
    assert s["mae"] == pytest.approx(0.5)
    assert s["coverage80"] == pytest.approx(0.5)
    # row 1: 0.1 * (1 - 0); row 2: actual below q10 -> 0.9 * 0.5
    assert s["pinball_q10"] == pytest.approx((0.1 + 0.9 * 0.5) / 2)
    assert s["pinball_q50"] == pytest.approx((0 + 0.5 * 1.0) / 2)


@pytest.mark.parametrize("method", list(mt.METHODS))
def test_methods_use_only_data_up_to_the_origin(panel, method):
    """Forecasts from origin 2021 are identical when every later season is changed."""
    t = _t("four_seam|avg_speed_mph")
    d = target_frame(panel, t)
    c = 2021
    a = mt.run_method(method, mt.OriginData(t, c, d[d.season <= c]))
    d2 = d.copy()
    d2.loc[d2.season > c, "value"] += 7.0
    b = mt.run_method(method, mt.OriginData(t, c, d2[d2.season <= c]))
    assert len(a) and len(a) == len(b)
    assert np.allclose(a["point"].values, b["point"].values)


def test_backtest_no_lookahead_and_2026_not_used_for_selection(panel):
    t = _t("four_seam|avg_speed_mph")
    d = target_frame(panel, t)
    cfg = bt.BacktestConfig(origins=tuple(range(2017, 2026)), methods=FAST, final_cutoff=2026)
    base = bt.backtest_target(d, t, cfg, PID)
    # 1) perturb 2026 values: development metrics, selection, Marcel k of the final forecast unchanged
    d26 = d.copy()
    d26.loc[d26.season == 2026, "value"] += 3.0
    pert = bt.backtest_target(d26, t, cfg, PID)
    m0, m1 = base["metrics"], pert["metrics"]
    dev0 = m0[(m0.breakdown_type == "subset") & (m0.breakdown_key == "selection")].sort_values(["method", "h"])
    dev1 = m1[(m1.breakdown_type == "subset") & (m1.breakdown_key == "selection")].sort_values(["method", "h"])
    assert np.allclose(dev0["pinball_mean"].values, dev1["pinball_mean"].values)
    assert {h: v["method"] for h, v in base["selection"].items()} == \
        {h: v["method"] for h, v in pert["selection"].items()}
    r0 = m0[(m0.breakdown_key == "retro_2026")].sort_values(["method", "h"])
    r1 = m1[(m1.breakdown_key == "retro_2026")].sort_values(["method", "h"])
    assert not np.allclose(r0["mae"].values, r1["mae"].values)
    assert base["fitted_state"]["marcel"]["k"] == pert["fitted_state"]["marcel"]["k"]
    assert base["fitted_state"]["marcel"]["tuned_through_origin"] == 2024
    # final-forecast empirical intervals come from development targets only
    f0, f1 = base["final"], pert["final"]
    emp = ~f0["method"].isin(list(mt.NATIVE_DISTRIBUTION))
    assert np.allclose((f0["q90"] - f0["point"])[emp].values, (f1["q90"] - f1["point"])[emp].values)
    # conformalized (CQR) widening of the native-quantile methods also uses development outcomes only
    assert "cqr_qhat" in f0.columns
    assert np.allclose(f0["cqr_qhat"].dropna().values, f1["cqr_qhat"].dropna().values)
    assert set(f0.loc[f0["cqr_qhat"].notna(), "method"]) == {"lgbm_quantile", "eb_kalman"}
    # 2) perturb seasons >= 2023: the pitcher's backtest forecasts from origins <= 2022 (points and intervals) agree
    d23 = d.copy()
    d23.loc[d23.season >= 2023, "value"] -= 4.0
    p23 = bt.backtest_target(d23, t, cfg, PID)
    a = base["target_pitcher_rows"]
    b = p23["target_pitcher_rows"]
    a, b = a[a.origin_season <= 2022].sort_values(["method", "origin_season", "h"]), \
        b[b.origin_season <= 2022].sort_values(["method", "origin_season", "h"])
    assert len(a) and np.allclose(a["point"].values, b["point"].values)
    assert np.allclose(a["q10"].values, b["q10"].values) and np.allclose(a["q90"].values, b["q90"].values)


def test_selection_subset_excludes_2020_cross_era_and_young():
    df = pd.DataFrame({"origin_season": [2018, 2019, 2021, 2022, 2023, 2024],
                       "target_season": [2020, 2021, 2022, 2025, 2026, 2025],
                       "origin_age": [35, 35, 35, 25, 35, 33.0]})
    sub = bt.subsets(df, bt.BacktestConfig())
    assert sub["selection"].tolist() == [False, False, True, False, False, True]
    assert sub["retro_2026"].tolist() == [False, False, False, False, True, False]


def test_eb_shrinks_short_histories_to_population(panel):
    t = _t("four_seam|avg_speed_mph")
    d = target_frame(panel, t)
    curve = mt.fit_age_curve(d, 2025, ipw=True)
    p = mt.fit_eb(d, 2025, curve)
    assert p.season_var >= 0 and p.drift_var > 0 and p.prior_var > 0
    league = pd.Series({2025: 94.0})
    # one 120-pitch season far from the population: posterior moves toward the league mean
    x1, P1 = mt.kalman_level([2025], [99.0], [120], [27.0], league, curve, p)
    assert 94.0 < x1 < 99.0
    # long, large-sample history at the same value: posterior close to the data
    x5, P5 = mt.kalman_level([2021, 2022, 2023, 2024, 2025], [99.0] * 5, [2000] * 5, [23, 24, 25, 26, 27.0],
                             pd.Series({s: 94.0 for s in range(2021, 2026)}), mt.AgeCurve(None, 20, 45, 0, 0.0, 0.0,
                                                                                          False), p)
    assert abs(x5 - 99.0) < abs(x1 - 99.0) and P5 < P1


def test_survival_weights_and_ipw_curve(panel):
    t = _t("ALL|k_pct")
    d = target_frame(panel, t)
    w = mt.survival_weights(d, 2025)
    assert len(w) and w.mean() == pytest.approx(1.0) and (w > 0).all()
    pairs = set(zip(d.pitcher_id, d.season))
    assert all((p, s + 1) in pairs for p, s in w.index)  # weights only for survivors
    c0, c1 = mt.fit_age_curve(d, 2025), mt.fit_age_curve(d, 2025, ipw=True)
    assert c0.n_pairs == c1.n_pairs > 100


def test_sample_draws_reproducible():
    from pitchiq.aging_model.pipeline import sample_draws
    a = sample_draws({"type": "normal", "mean": 95.0, "sd": 1.0}, 2000, 3)
    assert sample_draws({"type": "normal", "mean": 95.0, "sd": 1.0}, 2000, 3) == a
    assert np.mean(a) == pytest.approx(95.0, abs=0.1) and np.std(a) == pytest.approx(1.0, abs=0.1)
    q = {"0.1": 93.0, "0.5": 95.0, "0.9": 96.0}
    b = np.array(sample_draws({"type": "quantile_interp", "quantiles": q, "tail_sd": 1.17}, 4000, 5))
    assert np.mean(b <= 95.0) == pytest.approx(0.5, abs=0.04)
    assert np.mean(b <= 93.0) == pytest.approx(0.1, abs=0.03)


def test_location_backtest_own_history():
    rows = []
    for s, sx in zip(range(2015, 2027), [0.80, 0.78, 0.75, 0.77, 0.76, None, 0.79, None, 0.75, 0.73, 0.76, 0.73]):
        if sx is not None:
            rows.append({"pitcher_id": PID, "season": s, "unit": "slider", "metric": "plate_x_sd_ft", "value": sx,
                         "n": 800, "age": s - 1989.25, "qualified": True})
    res = bt.location_backtest(pd.DataFrame(rows))
    assert set(res["selection"]) == {"plate_x_sd_ft"}
    fin = res["rows"][res["rows"]["final"]]
    assert set(fin["target_season"]) == {2027, 2028, 2029}
    assert (fin["q90"] > fin["q10"]).all()
    dev = res["metrics"][res["metrics"]["subset"] == "dev"]
    assert len(dev) and (dev["n"] > 0).all()


# --------------------------------------------------------------------------- end to end on the test database
def _insert_fixture_rows(session):
    pss, psps, bio = _frames()
    from pitchiq.aging.families import age_completed_years, measurement_era, season_midpoint
    session.execute(text("DELETE FROM pitcher_bio WHERE pitcher_id IN :ids").bindparams(
        __import__("sqlalchemy").bindparam("ids", expanding=True)), {"ids": bio.pitcher_id.tolist()})
    from pitchiq.models_aging import PitcherBio, PitcherSeasonPitchStats, PitcherSeasonStats
    for r in bio.to_dict("records"):
        session.add(PitcherBio(pitcher_id=int(r["pitcher_id"]), full_name=r["full_name"],
                               birth_date=r["birth_date"].date(), throws=r["throws"]))
    session.flush()
    births = {int(r["pitcher_id"]): r["birth_date"].date() for r in bio.to_dict("records")}
    for r in pss.to_dict("records"):
        kw = {k: (None if isinstance(v, float) and np.isnan(v) else v) for k, v in r.items()}
        kw["age_june30"] = age_completed_years(births[int(kw["pitcher_id"])], season_midpoint(int(kw["season"])))
        session.add(PitcherSeasonStats(**kw, measurement_era=measurement_era(int(kw["season"])),
                                       low_sample=(kw["pitch_count"] or 0) < 500))
    for r in psps.to_dict("records"):
        kw = {k: (None if isinstance(v, float) and np.isnan(v) else v) for k, v in r.items()}
        session.add(PitcherSeasonPitchStats(**kw, pitch_group="fastball" if kw["pitch_family"] == "four_seam"
                                            else "breaking", measurement_era=measurement_era(int(kw["season"])),
                                            low_sample=(kw["pitches"] or 0) < 100))
    session.flush()


@pytest.fixture()
def engine_run(session):
    from pitchiq.aging_model.pipeline import run_aging_model
    _insert_fixture_rows(session)
    res = run_aging_model(session, PID, families=["four_seam"], origins=list(range(2017, 2026)), workers=1,
                          methods=list(FAST), analyses=True, mirror=True, n_draws=50)
    return {"res": res, "session": session}


def test_engine_end_to_end(engine_run):
    from pitchiq.models_aging_forecast import (AgingAnalysis, AgingBacktest, AgingBacktestMetric,
                                               AgingBacktestPrediction, AgingBacktestRun, AgingModelEntry,
                                               PitchCharacteristicForecast)
    s, res = engine_run["session"], engine_run["res"]
    key = res["backtest_key"]
    run = s.get(AgingBacktest, key)
    assert run.status == "succeeded" and run.historical_cutoff == 2026
    fc = s.scalars(select(PitchCharacteristicForecast).where(PitchCharacteristicForecast.backtest_key == key)).all()
    assert {f.target_season for f in fc} >= {2027, 2028, 2029}
    sel = [f for f in fc if f.selected and f.unit == "four_seam" and f.metric == "avg_speed_mph"]
    assert sorted(f.target_season for f in sel) == [2027, 2028, 2029]
    for f in sel:
        assert f.historical_cutoff == 2026 and f.horizon == f.target_season - 2026
        exp = (date(f.target_season, 6, 30) - date(1989, 3, 30)).days / 365.2425
        assert f.projected_age == pytest.approx(exp)
        assert f.lo80 <= f.point <= f.hi80 and len(f.sample_draws) == 50
        assert f.assumptions["tuning_and_interval_calibration_through_target_season"] == 2025
        assert f.assumptions["no_forced_decline"] is True
        assert f.last_observed_season == 2026
    # selection never names a method that was not backtested; entries + MLflow ids
    entries = s.scalars(select(AgingModelEntry).where(AgingModelEntry.backtest_key == key)).all()
    fams = {e.model_family for e in entries}
    assert {"aging_selected", "aging_location_own_history"} <= fams
    assert all(e.config.get("mlflow_run_id") for e in entries)
    assert [e.status for e in entries if e.model_family == "aging_selected"] == ["promoted"]
    runs = s.scalars(select(AgingBacktestRun).where(AgingBacktestRun.backtest_key == key)).all()
    assert {r.split for r in runs} >= {"development", "retrospective_2026_previously_viewed"}
    assert all(r.metrics.get("mlflow_run_id") for r in runs)
    assert s.scalar(select(func.count()).select_from(AgingBacktestMetric)
                    .where(AgingBacktestMetric.backtest_key == key)) > 50
    preds = s.scalars(select(AgingBacktestPrediction).where(AgingBacktestPrediction.backtest_key == key)).all()
    assert preds and all(p.pitcher_id == PID for p in preds)
    assert {p.evaluation_label for p in preds if p.target_season == 2026} <= {"retrospective_2026_previously_viewed"}
    an = s.scalars(select(AgingAnalysis).where(AgingAnalysis.backtest_key == key)).all()
    assert an and "ipw" in an[0].results["curves"] and "survival" in an[0].results
    # next-pitch registry untouched
    assert s.execute(text("SELECT count(*) FROM model_registry WHERE model_family LIKE 'aging%'")).scalar() == 0


def test_engine_mlflow_experiment(engine_run):
    from pitchiq.aging_model.pipeline import aging_tracker
    tr = aging_tracker()
    exp = tr.client.get_experiment_by_name(tr.experiment_name(PID))
    assert exp is not None and exp.name.endswith(f"-aging-{PID}")
    runs = tr.client.search_runs([exp.experiment_id], "tags.`pitchiq.system` = 'aging_forecast'")
    kinds = {r.data.tags.get("pitchiq.run_kind") for r in runs}
    assert kinds == {"training", "evaluation"}
    assert all(r.data.tags.get("pitchiq.split_policy", "").startswith("rolling origin") for r in runs)


def test_engine_api(engine_run):
    from fastapi.testclient import TestClient

    from pitchiq.api.main import app
    from pitchiq.db import get_db
    s = engine_run["session"]

    def _db():
        yield s
    app.dependency_overrides[get_db] = _db
    try:
        c = TestClient(app)
        h = c.get(f"/api/pitchers/{PID}/aging/history")
        assert h.status_code == 200
        keys = {x["key"] for x in h.json()["series"]}
        assert "four_seam|avg_speed_mph" in keys and "ALL|k_pct" in keys
        f = c.get(f"/api/pitchers/{PID}/aging/forecast")
        assert f.status_code == 200
        rows = f.json()["forecasts"]
        assert rows and all(r["selected"] for r in rows) and all("sample_draws" not in r for r in rows)
        assert c.get(f"/api/pitchers/{PID}/aging/forecast?all_methods=true&include_draws=true&unit=four_seam"
                     ).json()["forecasts"][0]["sample_draws"]
        b = c.get(f"/api/pitchers/{PID}/aging/backtest?unit=four_seam&metric=avg_speed_mph&include_predictions=true"
                  "&include_analyses=true")
        assert b.status_code == 200
        j = b.json()
        assert j["metrics"] and j["target_pitcher_predictions"] and j["analyses"] and j["selection"]
        assert c.get("/api/pitchers/1/aging/forecast").status_code == 404
    finally:
        app.dependency_overrides.pop(get_db, None)


def test_job_kind_registered():
    from pitchiq.jobs.tasks import DESCRIPTIONS, TASKS
    assert "aging_model" in TASKS and "aging_model" in DESCRIPTIONS
