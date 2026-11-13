"""Feature + data-leakage tests: every input reflects information known before the target pitch."""
import numpy as np
import pandas as pd
import pytest

from pitchiq.features.build import PRE_PITCH_COLUMNS, build_feature_rows
from pitchiq.features.schema import FORBIDDEN_SOURCE_COLUMNS, MODEL_EXCLUDED_DEFAULT, feature_list
from pitchiq.features.store import load_feature_frame, load_pitch_frame

PID = 519242
CLASSES = ["FF", "SL", "CH", "SI"]


@pytest.fixture(scope="module")
def frame(pipeline, db_url):
    from pitchiq.db import session_scope
    with session_scope(db_url) as s:
        return load_pitch_frame(s, PID)


def _feats(df):
    return {r["pitch_uid"]: r["features"] for r in build_feature_rows(df, CLASSES)}


def test_truncation_invariance(frame):
    """Features of pitch i are identical when computed from only rows[:i+1] with row i's outcome scrambled."""
    full = _feats(frame)
    rng = np.random.default_rng(0)
    for i in rng.choice(len(frame), size=40, replace=False):
        trunc = frame.iloc[: i + 1].copy()
        trunc.loc[trunc.index[-1], "pitch_type"] = "SI" if trunc.iloc[-1]["pitch_type"] != "SI" else "FF"
        trunc.loc[trunc.index[-1], "release_speed"] = 55.0
        trunc.loc[trunc.index[-1], "description"] = "hit_into_play"
        uid = frame.iloc[i]["pitch_uid"]
        assert _feats(trunc)[uid] == full[uid]


def test_future_perturbation_does_not_change_past(frame):
    full = _feats(frame)
    cut = len(frame) // 2
    pert = frame.copy()
    pert.loc[pert.index[cut:], "pitch_type"] = "CH"
    pert.loc[pert.index[cut:], "release_speed"] = 70.0
    pf = _feats(pert)
    for uid in frame.iloc[:cut]["pitch_uid"]:
        assert pf[uid] == full[uid]


def test_model_inputs_exclude_forbidden_columns():
    names = {f.name for f in feature_list(CLASSES)}
    assert not names & FORBIDDEN_SOURCE_COLUMNS
    assert not set(PRE_PITCH_COLUMNS) & FORBIDDEN_SOURCE_COLUMNS
    assert "season" in MODEL_EXCLUDED_DEFAULT


def test_pitch_count_excludes_automatic_calls_and_differs_from_pa_number(frame):
    rows = build_feature_rows(frame, CLASSES)
    df = pd.DataFrame([{"uid": r["pitch_uid"], "label": r["label"], **r["features"]} for r in rows])
    df = df.merge(frame[["pitch_uid", "game_pk"]], left_on="uid", right_on="pitch_uid")
    for _, g in df.groupby("game_pk"):
        thrown = g["label"].notna().cumsum().shift(fill_value=0).to_numpy()
        assert (g["pitch_count_before"].to_numpy() == thrown).all()
    assert (df["pitch_count_before"] != df["pa_pitch_number"] - 1).any()


def test_first_pitch_of_game_has_no_history(frame):
    rows = build_feature_rows(frame, CLASSES)
    first = rows[0]["features"]
    assert first["pitch_count_before"] == 0 and first["prev1_type"] == "NONE" and first["prev1_speed"] is None


def test_splits_are_chronological(session):
    df = load_feature_frame(session, PID, None)
    seasons = df.groupby("split")["season"].agg(["min", "max"])
    assert seasons.loc["train", "max"] < seasons.loc["validation", "min"] <= seasons.loc["validation", "max"] < seasons.loc["test", "min"]


def test_no_game_spans_splits(session):
    from sqlalchemy import text
    n = session.execute(text("""SELECT count(*) FROM (SELECT p.game_pk FROM pre_pitch_states s JOIN pitches p USING (pitch_uid)
                               GROUP BY p.game_pk HAVING count(DISTINCT s.split) > 1) x""")).scalar()
    assert n == 0
