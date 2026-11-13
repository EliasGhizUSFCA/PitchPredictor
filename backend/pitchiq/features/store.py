"""Load pitches from PostgreSQL, compute pre-pitch states, persist them with split assignment."""
from __future__ import annotations

from collections import Counter

import pandas as pd
from sqlalchemy import delete, select, text
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from ..config import get_settings
from ..models import Pitcher, PrePitchState
from .build import build_feature_rows
from .schema import FEATURE_VERSION, schema_dict

PITCH_SQL = text("""
SELECT p.pitch_uid, p.game_pk, p.game_date, p.season, p.batter_id AS batter, p.stand, p.balls, p.strikes,
       p.inning, p.inning_topbot, p.outs_when_up, p.on_1b, p.on_2b, p.on_3b, p.bat_score, p.fld_score,
       p.at_bat_number, p.pitch_number, p.n_thruorder_pitcher, pa.batting_order_slot AS bat_order_slot,
       (p.raw->>'pitcher_days_since_prev_game')::float AS days_rest,
       p.pitch_type, p.release_speed, p.description, p.home_team, p.away_team, p.catcher_id
FROM pitches p JOIN plate_appearances pa USING (game_pk, at_bat_number)
WHERE p.pitcher_id = :pid
ORDER BY p.game_date, p.game_pk, p.at_bat_number, p.pitch_number
""")


def load_pitch_frame(session: Session, pitcher_id: int) -> pd.DataFrame:
    df = pd.read_sql(PITCH_SQL, session.connection(), params={"pid": pitcher_id})
    df["game_date"] = pd.to_datetime(df["game_date"]).dt.date
    return df


def split_for_season(season: int) -> str:
    s = get_settings()
    if season in s.train_seasons:
        return "train"
    if season in s.validation_seasons:
        return "validation"
    if season in s.test_seasons:
        return "test"
    return "unassigned"


def determine_classes(df: pd.DataFrame) -> list[str]:
    """Model classes = official Statcast codes with >= min_class_share of TRAINING pitches."""
    s = get_settings()
    train = df[df["season"].isin(s.train_seasons) & df["pitch_type"].notna()]
    counts = Counter(train["pitch_type"])
    n = sum(counts.values())
    classes = sorted([c for c, k in counts.items() if k / n >= s.min_class_share], key=lambda c: -counts[c])
    if any(k / n < s.min_class_share for k in counts.values()) or set(df["pitch_type"].dropna()) - set(classes):
        classes.append("OTHER")
    return classes


def build_and_store(session: Session, pitcher_id: int) -> dict:
    df = load_pitch_frame(session, pitcher_id)
    classes = determine_classes(df)
    rows = build_feature_rows(df, classes)
    season_of = dict(zip(df["pitch_uid"], df["season"]))
    session.execute(delete(PrePitchState).where(PrePitchState.pitcher_id == pitcher_id,
                                                PrePitchState.feature_version == FEATURE_VERSION))
    payload = [dict(pitch_uid=int(r["pitch_uid"]), feature_version=FEATURE_VERSION, pitcher_id=pitcher_id,
                    season=int(season_of[r["pitch_uid"]]), split=split_for_season(int(season_of[r["pitch_uid"]])),
                    features=_jsonable(r["features"]), label=r["label"]) for r in rows]
    for i in range(0, len(payload), 1000):
        session.execute(insert(PrePitchState).values(payload[i:i + 1000]))
    p = session.get(Pitcher, pitcher_id)
    extra = dict(p.extra or {})
    fs = dict(extra.get("feature_sets", {}))
    fs[FEATURE_VERSION] = schema_dict(classes)
    extra["feature_sets"] = fs
    p.extra = extra
    session.flush()
    splits = Counter((x["split"], x["label"] is not None) for x in payload)
    return {"feature_version": FEATURE_VERSION, "classes": classes, "rows": len(payload),
            "labeled_by_split": {k[0]: v for k, v in splits.items() if k[1]}}


def _jsonable(d: dict) -> dict:
    out = {}
    for k, v in d.items():
        if v is None or (isinstance(v, float) and v != v):
            out[k] = None
        elif hasattr(v, "item"):
            out[k] = v.item()
        else:
            out[k] = v
    return out


def get_classes(session: Session, pitcher_id: int, feature_version: str = FEATURE_VERSION) -> list[str]:
    p = session.get(Pitcher, pitcher_id)
    return p.extra["feature_sets"][feature_version]["classes"]


def load_feature_frame(session: Session, pitcher_id: int, splits: list[str] | None = None,
                       feature_version: str = FEATURE_VERSION, labeled_only: bool = True) -> pd.DataFrame:
    q = select(PrePitchState.pitch_uid, PrePitchState.split, PrePitchState.label, PrePitchState.features,
               PrePitchState.season).where(PrePitchState.pitcher_id == pitcher_id,
                                           PrePitchState.feature_version == feature_version)
    if splits:
        q = q.where(PrePitchState.split.in_(splits))
    if labeled_only:
        q = q.where(PrePitchState.label.is_not(None))
    rows = session.execute(q.order_by(PrePitchState.pitch_uid)).all()
    feats = pd.DataFrame([r.features for r in rows])
    meta = pd.DataFrame([{"pitch_uid": r.pitch_uid, "split": r.split, "label": r.label} for r in rows])
    return pd.concat([meta, feats], axis=1)
