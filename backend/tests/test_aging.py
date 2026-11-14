"""Aging dataset ETL: offline on fixtures cut from real Savant / MLB Stats API responses (make_aging_fixtures.py).

Everything runs inside the rolled-back `session` fixture, so the shared test database is left as other tests expect.
"""
import io
import json
import tempfile
from datetime import date
from pathlib import Path

import pandas as pd
import pytest
from sqlalchemy import func, select, text

from pitchiq.aging import families as fam
from pitchiq.aging import sources as src
from pitchiq.http import PoliteClient, SourceUnavailable

AFIX = Path(__file__).parent / "fixtures" / "aging"
SEASONS = [2019, 2020, 2022, 2023]
PID = 519242
TMP = Path(tempfile.mkdtemp(prefix="pitchiq-aging-test-"))
SABATHIA, HERNANDEZ, HOUCK, FLAHERTY = 282332, 433587, 656557, 656427


class AgingFixtureClient(PoliteClient):
    """Replays recorded real responses by exact request key; Statcast pitch-level requests are served from one real
    Sale 2019 game. Anything else raises SourceUnavailable (like an outage)."""

    def __init__(self):
        super().__init__(cache_dir=TMP / "raw_aging", min_interval_s=0)
        self.index = json.loads((AFIX / "index.json").read_text())
        self.sale = pd.read_csv(AFIX / "sale_statcast_2019_subset.csv", low_memory=False)
        self.calls: list[str] = []
        people = [n for k, n in self.index.items() if k.startswith("https://statsapi.mlb.com/api/v1/people?")]
        self.people = [p for n in people for p in json.loads((AFIX / n).read_text())["people"]]

    def get(self, url, params=None, *, namespace, suffix=".txt", use_cache=True, retries=3):
        self.calls.append(url)
        if "statcast_search" in url and params and "game_date_gt" in params:
            d = self.sale
            m = (d.game_date >= params["game_date_gt"]) & (d.game_date <= params["game_date_lt"])
            m &= d.game_year == int(params["hfSea"].strip("|"))
            if not m.any():
                return ""
            buf = io.StringIO()
            d[m].to_csv(buf, index=False)
            return buf.getvalue()
        if url.endswith("/v1/people") and params and "personIds" in params:
            ids = {int(x) for x in params["personIds"].split(",")}
            return json.dumps({"people": [p for p in self.people if p["id"] in ids]})
        key = url + "?" + json.dumps(params or {}, sort_keys=True)
        if key not in self.index:
            raise SourceUnavailable(f"no aging fixture for {url}")
        return (AFIX / self.index[key]).read_text()


def _fixture_csv(url_part: str, **params) -> pd.DataFrame:
    idx = json.loads((AFIX / "index.json").read_text())
    for key, name in idx.items():
        u, _, p = key.partition("?")
        p = json.loads(p)
        if url_part in u and all(str(p.get(k)) == str(v) for k, v in params.items()):
            return pd.read_csv(AFIX / name)
    raise KeyError(url_part)


@pytest.fixture()
def aged(session):
    """Run the full aging ETL (incl. Sale's 2019 career pitches) inside the rolled-back test session."""
    from pitchiq.aging.etl import run_aging_etl
    from pitchiq.features.store import load_pitch_frame
    from pitchiq.models import Pitcher
    before_frame = load_pitch_frame(session, PID)
    before_seasons = list(session.get(Pitcher, PID).seasons)
    client = AgingFixtureClient()
    res = run_aging_etl(session, SEASONS, client=client, career_pitcher_id=PID, career_seasons=[2019])
    return {"res": res, "session": session, "before_frame": before_frame, "before_seasons": before_seasons,
            "client": client}


# --------------------------------------------------------------------------- pure helpers
def test_pitch_families_and_reclassification_mapping():
    assert {fam.pitch_family(c) for c in ("SL", "ST", "SV")} == {"slider"}
    assert fam.pitch_family("KC") == fam.pitch_family("CU") == "curveball"
    assert fam.leaderboard_code("KC") == "CU" and fam.leaderboard_code("SL") == "SL"
    assert fam.pitch_group("SI") == fam.pitch_group("FF") == "fastball"
    assert fam.pitch_group("ST") == "breaking" and fam.pitch_group("FS") == "offspeed"
    assert fam.pitch_family(None) == "unknown" and fam.pitch_family("ZZ") == "other"


def test_measurement_era_and_ages():
    assert fam.measurement_era(2019) == "trackman" and fam.measurement_era(2020) == "hawkeye"
    assert fam.era_change(2019, 2020) and not fam.era_change(2022, 2023)
    birth = date(1989, 3, 30)  # Chris Sale, as returned by the MLB Stats API fixture
    assert fam.age_completed_years(birth, date(2019, 6, 30)) == 30
    assert fam.age_completed_years(birth, date(2019, 3, 29)) == 29
    assert fam.age_completed_years(birth, date(2019, 3, 30)) == 30
    assert fam.age_years(birth, date(2019, 6, 30)) == pytest.approx((date(2019, 6, 30) - birth).days / 365.2425)
    assert fam.age_years(None, date(2019, 6, 30)) is None


def test_search_long_folds_knuckle_curve_into_cu():
    def f(code, pitches, velo, whiffs, swings):
        df = pd.DataFrame({"player_id": [1], "pitches": [pitches], "total_pitches": [1000], "velocity": [velo],
                           "spin_rate": [2500], "release_extension": [6.0], "whiffs": [whiffs], "swings": [swings]})
        return src.Fetched("search", 2024, f"u-{code}", pd.Timestamp("2026-01-01", tz="UTC"), df, code)
    out = src.search_long([f("CU", 100, 80.0, 10, 40), f("KC", 300, 84.0, 30, 60), f("FF", 600, 95.0, 20, 200)])
    cu = out[out.pitch_type == "CU"].iloc[0]
    assert cu.gs_pitches == 400 and cu.gs_whiffs == 40 and cu.gs_swings == 100
    assert cu.gs_velocity == pytest.approx((100 * 80 + 300 * 84) / 400)
    assert sorted(cu.search_codes) == ["CU", "KC"]
    assert out[out.pitch_type == "FF"].iloc[0].gs_pitches == 600


# --------------------------------------------------------------------------- ETL on real-response fixtures
def test_etl_stores_pitcher_season_rows_matching_sources(aged):
    from pitchiq.models_aging import AgingEtlRun, PitcherBio, PitcherSeasonPitchStats, PitcherSeasonStats
    s = aged["session"]
    custom = {y: _fixture_csv("leaderboard/custom", year=y) for y in SEASONS}
    n_pss = s.scalar(select(func.count()).select_from(PitcherSeasonStats))
    assert n_pss >= sum(len(c) for c in custom.values())
    sale = s.get(PitcherSeasonStats, (PID, 2019))
    c19 = custom[2019].set_index("player_id").loc[PID]
    assert sale.pitch_count == int(c19["pitch_count"]) and sale.k_pct == pytest.approx(float(c19["k_percent"]))
    assert sale.oz_swing_pct == pytest.approx(float(c19["oz_swing_percent"]))
    assert sale.age_june30 == 30 == sale.savant_player_age
    assert sale.measurement_era == "trackman" and sale.field_sources["pitch_count"] == "custom"
    assert sale.source_urls["custom"].startswith("https://baseballsavant.mlb.com/leaderboard/custom?")
    assert sale.extension_ft is not None and sale.field_sources["extension_ft"] == "search"
    whole, _, frac = str(c19["p_formatted_ip"]).partition(".")
    assert sale.outs == int(whole) * 3 + int(frac or 0) and sale.ip == pytest.approx(sale.outs / 3)
    bio = s.get(PitcherBio, PID)
    assert bio.birth_date == date(1989, 3, 30) and bio.throws == "L"
    # pitch-type rows equal the movement leaderboard values for the same pitcher / type
    mv = _fixture_csv("pitch-movement", year=2019, pitch_type="SL").set_index("pitcher_id").loc[PID]
    sl = s.get(PitcherSeasonPitchStats, (PID, 2019, "SL"))
    assert sl.pitches == int(mv["pitches_thrown"]) and sl.avg_speed_mph == pytest.approx(float(mv["avg_speed"]))
    assert sl.ivb_in == pytest.approx(float(mv["pitcher_break_z_induced"])) and sl.pitch_family == "slider"
    assert sl.field_sources["avg_speed_mph"] == "movement" and "movement" in sl.values_by_source
    # 2019: pitch-arsenal-stats exists from 2017, so whiff% comes from it
    assert sl.field_sources["whiff_pct"] == "arsenal_stats"
    run = s.scalars(select(AgingEtlRun).order_by(AgingEtlRun.etl_run_id.desc())).first()
    assert run.status == "succeeded" and run.units["avg_speed_mph"].startswith("mph")
    assert run.coverage["totals"]["seasons"] == [2019, 2023]
    assert run.validation["savant_age_vs_computed_june30_age"]["agreement_rate"] == 1.0


def test_low_sample_and_flags(aged):
    from pitchiq.models_aging import PitcherSeasonPitchStats, PitcherSeasonStats
    s = aged["session"]
    low = s.scalars(select(PitcherSeasonPitchStats).where(PitcherSeasonPitchStats.pitches < 100)).all()
    assert low and all(r.low_sample and "low_sample" in r.quality_flags for r in low)
    ok = s.scalars(select(PitcherSeasonPitchStats).where(PitcherSeasonPitchStats.pitches >= 100)).all()
    assert ok and not any(r.low_sample for r in ok)
    s20 = s.scalars(select(PitcherSeasonStats).where(PitcherSeasonStats.season == 2020)).all()
    assert s20 and all("short_season" in r.quality_flags and r.measurement_era == "hawkeye" for r in s20)


def test_knuckle_curve_pitcher_has_cu_row_only(aged):
    from pitchiq.models_aging import PitcherSeasonPitchStats
    s = aged["session"]
    codes = set(s.scalars(select(PitcherSeasonPitchStats.pitch_type).where(
        PitcherSeasonPitchStats.pitcher_id == FLAHERTY)))
    assert "KC" not in codes  # leaderboards fold KC into CU; the search rows follow the same convention
    cu = s.scalars(select(PitcherSeasonPitchStats).where(PitcherSeasonPitchStats.pitcher_id == FLAHERTY,
                                                         PitcherSeasonPitchStats.pitch_type == "CU")).all()
    assert cu and all(r.pitch_family == "curveball" for r in cu)


def test_observations_survivors_dropouts_and_flags(aged):
    from pitchiq.models_aging import AgingObservation as O
    s = aged["session"]

    def obs(pid, unit, metric, season):
        return s.scalars(select(O).where(O.pitcher_id == pid, O.pitch_family == unit, O.metric == metric,
                                         O.season_from == season)).first()
    # Sabathia and Felix Hernandez threw their last MLB pitches in 2019: recorded as dropouts, no delta
    for pid in (SABATHIA, HERNANDEZ):
        o = obs(pid, "ALL", "pitch_count", 2019)
        assert o is not None and o.survived is False and o.value_to is None and o.delta is None
        assert o.returned_later is False and o.next_qualifying_season is None
    # 2019 -> 2020 pairs carry the TrackMan -> Hawk-Eye and short-season flags
    pairs = s.scalars(select(O).where(O.season_from == 2019)).all()
    assert pairs and all(o.era_change and o.short_season and o.era_from == "trackman" for o in pairs)
    # 2021 was not fetched: no 2020 -> 2021 rows (a missing season must not look like a dropout)
    assert s.scalar(select(func.count()).select_from(O).where(O.season_from == 2020)) == 0
    # delta = value_to - value_from, ages from the birth date
    o = obs(PID, "four_seam", "avg_speed_mph", 2022)
    if o is not None and o.value_to is not None:
        assert o.delta == pytest.approx(o.value_to - o.value_from)
    surv = s.scalars(select(O).where(O.survived, O.both_qualified, O.metric == "fastball_avg_speed_mph")).all()
    assert surv and all(o.delta == pytest.approx(o.value_to - o.value_from) for o in surv)
    assert all(o.age_to - o.age_from == pytest.approx(1.0, abs=0.01) for o in surv if o.age_from is not None)
    # Houck: 2022 slider family is SL only; in 2023 Statcast added ST (sweeper) -> reclass flag
    h = obs(HOUCK, "slider", "avg_speed_mph", 2022)
    assert h is not None and h.codes_from == ["SL"] and "ST" in (h.codes_to or [])
    assert h.reclass_flag is True


def test_cohorts_count_every_at_risk_pitcher_season(aged):
    from pitchiq.models_aging import AgingCohort, AgingObservation as O
    s = aged["session"]
    at_risk = s.scalars(select(O).where(O.pitch_family == "ALL", O.metric == "pitch_count")).all()
    pooled = s.scalars(select(AgingCohort).where(AgingCohort.cohort_type == "age",
                                                 AgingCohort.scope == "pooled")).all()
    # pooled survival counts exclude short-season pairs (2019->2020), which are counted separately
    full = [o for o in at_risk if o.age_from_int is not None and not o.short_season]
    short = [o for o in at_risk if o.age_from_int is not None and o.short_season]
    assert full and short
    assert sum(c.n_pitchers for c in pooled) == len(full)
    assert sum(c.n_dropped for c in pooled) == sum(1 for o in full if not o.survived)
    assert sum(c.metric_summaries.get("short_season_pairs", {}).get("n", 0) for c in pooled) == len(short)
    assert sum(c.n_era_change for c in pooled) == 0
    for c in pooled:
        assert c.n_survived + c.n_dropped == c.n_pitchers
        if c.n_pitchers:
            assert c.survival_rate == pytest.approx(c.n_survived / c.n_pitchers)
    # per-season scope 2019 keeps its (short-season, era-change) pairs
    s19 = s.scalars(select(AgingCohort).where(AgingCohort.cohort_type == "age", AgingCohort.scope == "2019")).all()
    assert sum(c.n_pitchers for c in s19) == sum(1 for o in short if o.season_from == 2019)


def test_short_season_survival_uses_scaled_threshold(aged):
    from pitchiq.models_aging import AgingObservation as O, PitcherSeasonStats as PSS
    s = aged["session"]
    assert fam.scaled_threshold(500, 2020) == 185 and fam.scaled_threshold(500, 2019) == 500
    pc20 = {r.pitcher_id: r.pitch_count for r in s.scalars(select(PSS).where(PSS.season == 2020)).all()}
    pairs = s.scalars(select(O).where(O.season_from == 2019, O.pitch_family == "ALL", O.metric == "pitch_count")).all()
    assert pairs
    for o in pairs:
        assert o.survived == (pc20.get(o.pitcher_id) is not None and pc20[o.pitcher_id] >= 185)
    # with a 2000-pitch threshold (741 in 2020): Scherzer (2,770 -> 1,213 in 2020) survives although 1,213 < 2,000;
    # Flaherty (3,179 -> 719) does not
    from pitchiq.aging.etl import rebuild_observations
    rebuild_observations(s, 100, 2000, None)
    s.expire_all()
    got = {o.pitcher_id: o.survived for o in s.scalars(select(O).where(
        O.season_from == 2019, O.pitch_family == "ALL", O.metric == "pitch_count")).all()}
    assert got == {453286: True, 656427: False, PID: False}


def test_reclass_flag_only_for_code_changes_inside_a_family(aged):
    from pitchiq.models_aging import AgingObservation as O
    s = aged["session"]
    rows = s.scalars(select(O).where(O.metric.in_(["avg_speed_mph", "pitch_count"]))).all()
    fam_flag = {(o.pitcher_id, o.season_from, o.pitch_family): o.reclass_flag for o in rows
                if not o.pitch_family.startswith("group:") and o.pitch_family != "ALL"}
    groups = [o for o in rows if o.pitch_family.startswith("group:")]
    assert groups
    for o in groups:
        g = o.pitch_family.split(":", 1)[1]
        expect = any(v for (pid, sf, f), v in fam_flag.items()
                     if pid == o.pitcher_id and sf == o.season_from and fam.FAMILY_GROUP.get(f) == g)
        assert o.reclass_flag == expect
    # arsenal-only changes in group:fastball (e.g. adding a cutter) are not flagged
    assert any(o.pitch_family == "group:fastball" and o.codes_from and o.codes_to
               and set(o.codes_from) != set(o.codes_to) and not o.reclass_flag for o in groups)
    assert not any(o.reclass_flag for o in rows if o.pitch_family == "ALL")


def test_as_of_cutoff_removes_look_ahead(aged):
    from pitchiq.aging.asof import apply_cutoff, cohorts_as_of, load_observations_as_of
    s = aged["session"]
    full = pd.read_sql(text("SELECT * FROM aging_observations WHERE pitch_family='ALL' AND metric='pitch_count'"),
                       s.connection())
    # the stored columns look ahead: 2019 non-survivors who qualified again in 2022/2023
    ahead = full[(full.season_to <= 2020) & full.returned_later.astype(bool)]
    assert len(ahead) and (ahead.next_qualifying_season > 2020).all()
    v = load_observations_as_of(s, 2020, "ALL", "pitch_count")
    assert len(v) and (v.season_to <= 2020).all()
    assert v.next_qualifying_season.dropna().le(2020).all() and not v.returned_later.any()
    assert v.loc[v.pitcher_id.isin(ahead.pitcher_id), "outcome_censored"].all()
    assert apply_cutoff(full, 2023).returned_later.sum() == full.returned_later.sum()
    rows = cohorts_as_of(s, 2020)
    assert rows and all(r["n_returned_later"] == 0 for r in rows)
    assert all(r["metric_summaries"]["returned_later_lookahead_through"] <= 2020 for r in rows)
    pooled = [r for r in rows if r["cohort_type"] == "age" and r["scope"] == "pooled"]
    assert sum(r["n_pitchers"] for r in pooled) == 0  # only short-season pairs exist by 2020 in the fixtures


def test_rebuild_is_idempotent(aged):
    from pitchiq.aging.etl import rebuild_observations
    s = aged["session"]
    a = rebuild_observations(s, 100, 500, None)
    b = rebuild_observations(s, 100, 500, None)
    assert a == b and a["observations"] == aged["res"]["observations"]


def test_career_pitches_are_history_only_and_do_not_reach_modeling(aged):
    from pitchiq.analytics import filter_options
    from pitchiq.features.store import load_pitch_frame
    from pitchiq.models import Pitcher
    from pitchiq.video.discovery import coverage
    s = aged["session"]
    n19 = s.execute(text("SELECT count(*) FROM pitches WHERE pitcher_id=:p AND season=2019"), {"p": PID}).scalar()
    assert n19 == len(pd.read_csv(AFIX / "sale_statcast_2019_subset.csv")) > 0
    assert aged["res"]["career"]["rows_by_season"] == {"2019": n19}
    # exact age per pitch from the pitch_ages view (birth date 1989-03-30)
    ages = s.execute(text("""SELECT p.game_date, a.age_exact_years, a.age_completed_years FROM pitches p
                             JOIN pitch_ages a USING (pitch_uid) WHERE p.pitcher_id=:p AND p.season=2019"""),
                     {"p": PID}).all()
    assert len(ages) == n19
    for gd, exact, whole in ages:
        assert float(exact) == pytest.approx((gd - date(1989, 3, 30)).days / 365.2425)
        assert whole == fam.age_completed_years(date(1989, 3, 30), gd)
    # modeling paths are unchanged by the history-only season
    after = load_pitch_frame(s, PID)
    assert 2019 not in set(after["season"])
    pd.testing.assert_frame_equal(after.reset_index(drop=True), aged["before_frame"].reset_index(drop=True))
    p = s.get(Pitcher, PID)
    assert list(p.seasons) == aged["before_seasons"] and p.extra["history_only_seasons"] == [2019]
    assert 2019 not in filter_options(s, PID)["seasons"]
    assert coverage(s, PID)["total_pitches"] == s.execute(text(
        "SELECT count(*) FROM pitches WHERE pitcher_id=:p AND pitch_type IS NOT NULL AND season >= 2023"),
        {"p": PID}).scalar()
    # no pre-pitch states were created for the history-only rows
    assert s.execute(text("""SELECT count(*) FROM pre_pitch_states s JOIN pitches p USING (pitch_uid)
                             WHERE p.season = 2019""")).scalar() == 0
    v = aged["res"]["validation"]["career_pitch_level_vs_savant"]["2019"]
    assert v["pitch_rows"] == n19 and v["savant_pitch_count"] is not None


def test_bio_source_url_is_exact_for_full_bulk_request(session):
    """A 100-id bulk people URL exceeds 512 chars; the stored provenance must be the exact URL (not truncated)."""
    from pitchiq.aging.etl import ensure_bios
    from pitchiq.models_aging import PitcherBio
    ids = [PID] + list(range(600001, 600100))  # 99 ids absent from the fixtures: requested, not returned
    log_ = src.SourceLog()
    res = ensure_bios(session, AgingFixtureClient(), set(ids), log_, refresh=True)
    assert res["requested"] == 100 and res["returned"] == 1 and len(res["not_returned"]) == 99
    bio = session.get(PitcherBio, PID)
    expected = src.full_url(src.PEOPLE_URL, {"personIds": ",".join(map(str, sorted(ids)))})
    assert len(expected) > 512 and bio.source_url == expected
    assert bio.birth_date == date(1989, 3, 30)


def test_career_ingest_refuses_modeling_seasons(session):
    from pitchiq.aging.etl import ingest_career_pitches
    out = ingest_career_pitches(session, PID, [2023, 2024], client=AgingFixtureClient())
    assert out == {"seasons_requested": [], "pitches": 0}


def test_feature_rebuild_identical_with_history_only_rows(aged):
    """Recomputing features after the 2019 rows were added reproduces the stored pre-pitch states exactly."""
    from pitchiq.features.build import build_feature_rows
    from pitchiq.features.store import get_classes, load_pitch_frame
    s = aged["session"]
    rows = build_feature_rows(load_pitch_frame(s, PID), get_classes(s, PID))
    stored = {r.pitch_uid: (r.features, r.label) for r in s.execute(text(
        "SELECT pitch_uid, features, label FROM pre_pitch_states WHERE pitcher_id=:p AND feature_version='v1'"),
        {"p": PID})}
    assert len(rows) == len(stored)
    from pitchiq.features.store import _jsonable
    for r in rows:
        f, lab = stored[int(r["pitch_uid"])]
        assert lab == r["label"]
        assert _jsonable(r["features"]) == f


def test_job_kind_and_api(api):
    from pitchiq.jobs.tasks import DESCRIPTIONS, TASKS
    assert "aging_etl" in TASKS and "aging_etl" in DESCRIPTIONS
    r = api.get("/api/aging/coverage")
    assert r.status_code == 200 and "etl_run" in r.json()
    assert api.get("/api/aging/cohorts").status_code == 200
    r = api.get(f"/api/aging/pitchers/{PID}/career-pitches")
    assert r.status_code == 200
    assert all(x["role"] == "modeling" for x in r.json()["seasons"])  # committed test DB has no history-only rows
    assert api.get("/api/aging/pitchers/1").status_code == 404
    assert api.get("/api/aging/cohorts?as_of=2022").status_code == 200
    r = api.get("/api/aging/observations?as_of=2022&metric=pitch_count")
    assert r.status_code == 200 and isinstance(r.json(), list)
