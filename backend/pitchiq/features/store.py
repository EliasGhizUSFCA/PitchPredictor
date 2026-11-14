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
from .schema import FEATURE_VERSION, FEATURE_VERSIONS, feature_list, schema_dict

PITCH_SQL = text("""
SELECT p.pitch_uid, p.game_pk, p.game_date, p.season, p.batter_id AS batter, p.stand, p.balls, p.strikes,
       p.inning, p.inning_topbot, p.outs_when_up, p.on_1b, p.on_2b, p.on_3b, p.bat_score, p.fld_score,
       p.at_bat_number, p.pitch_number, p.n_thruorder_pitcher, pa.batting_order_slot AS bat_order_slot,
       (p.raw->>'pitcher_days_since_prev_game')::float AS days_rest,
       p.pitch_type, p.release_speed, p.description, p.home_team, p.away_team, p.catcher_id,
       p.plate_x, p.plate_z, p.zone  -- post-pitch: read only after the pitch's features are frozen
FROM pitches p JOIN plate_appearances pa USING (game_pk, at_bat_number)
WHERE p.pitcher_id = :pid AND p.season = ANY(:seasons)  -- modeling seasons only (history-only seasons excluded)
ORDER BY p.game_date, p.game_pk, p.at_bat_number, p.pitch_number
""")


def load_pitch_frame(session: Session, pitcher_id: int) -> pd.DataFrame:
    from ..seasons import modeling_seasons
    df = pd.read_sql(PITCH_SQL, session.connection(), params={"pid": pitcher_id, "seasons": modeling_seasons()})
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


def build_and_store(session: Session, pitcher_id: int, versions: tuple[str, ...] = FEATURE_VERSIONS) -> dict:
    """Rebuild the pre-pitch states of every feature version in `versions` in one pass.

    The builder produces the v2 dictionary; v1 rows are its v1 keys (identical values), so models trained on any
    version keep scoring newly ingested games."""
    from ..batter_priors import load_priors
    df = load_pitch_frame(session, pitcher_id)
    classes = determine_classes(df)
    priors = load_priors(session, pitcher_id, classes)
    rows = build_feature_rows(df, classes, batter_priors=priors)
    season_of = dict(zip(df["pitch_uid"], df["season"]))
    p = session.get(Pitcher, pitcher_id)
    extra = dict(p.extra or {})
    fs = dict(extra.get("feature_sets", {}))
    out = {"feature_version": FEATURE_VERSION, "classes": classes, "versions": {},
           "batter_priors_loaded": priors is not None}
    for version in versions:
        keys = [f.name for f in feature_list(classes, version)]
        session.execute(delete(PrePitchState).where(PrePitchState.pitcher_id == pitcher_id,
                                                    PrePitchState.feature_version == version))
        payload = [dict(pitch_uid=int(r["pitch_uid"]), feature_version=version, pitcher_id=pitcher_id,
                        season=int(season_of[r["pitch_uid"]]), split=split_for_season(int(season_of[r["pitch_uid"]])),
                        features=_jsonable({k: r["features"][k] for k in keys}), label=r["label"]) for r in rows]
        for i in range(0, len(payload), 1000):
            session.execute(insert(PrePitchState).values(payload[i:i + 1000]))
        fs[version] = schema_dict(classes, version)
        splits = Counter((x["split"], x["label"] is not None) for x in payload)
        out["versions"][version] = {"rows": len(payload), "n_features": len(keys),
                                    "labeled_by_split": {k[0]: v for k, v in splits.items() if k[1]}}
    extra["feature_sets"] = fs
    p.extra = extra
    session.flush()
    cur = out["versions"].get(FEATURE_VERSION) or next(iter(out["versions"].values()))
    out.update(rows=cur["rows"], labeled_by_split=cur["labeled_by_split"])
    return out


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
    fs = p.extra["feature_sets"]
    if feature_version not in fs:  # classes do not depend on the version (determine_classes); older DBs lack v2
        return next(iter(fs.values()))["classes"]
    return fs[feature_version]["classes"]


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
