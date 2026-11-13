"""Prospective ledger: the honest future-evaluation mechanism.

When completed starts are ingested (feature rebuild), every labeled pitch of a game is predicted by the model that was
*production on that game's date* - provided that model was created before the game date and its data cutoff is
before the game date. Predictions are stored in `pitch_predictions` with context "prospective". Games for which no
such model existed are skipped (and counted), never back-filled with a later model.

Predictions use only the stored pre-pitch feature state; the label is attached afterwards for scoring.
"""
from __future__ import annotations

import logging
from datetime import datetime, time, timezone

import numpy as np
import pandas as pd
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from ..features.schema import feature_list
from ..logging_setup import log
from ..models import ModelRegistry, PitchPrediction
from ..models_lifecycle import ModelPromotionEvent
from . import lifecycle, registry
from .metrics import reliability, summarize
from .protocol import game_table, load_dated_frame

logger = logging.getLogger(__name__)
CONTEXT = "prospective"


def production_timeline(session: Session, pitcher_id: int) -> list[tuple[datetime, str]]:
    """(time, version) at which each model became production, oldest first.

    Promotion events are authoritative; models promoted before events were recorded use config["promoted_at"]."""
    tl = [(ev.created_at, ev.model_version) for ev in session.scalars(
        select(ModelPromotionEvent).where(ModelPromotionEvent.pitcher_id == pitcher_id,
                                          ModelPromotionEvent.to_status == "promoted"))]
    with_events = {v for _, v in tl}
    for e in session.scalars(select(ModelRegistry).where(ModelRegistry.pitcher_id == pitcher_id)):
        at = (e.config or {}).get("promoted_at")
        if at and e.model_version not in with_events:
            tl.append((datetime.fromisoformat(at), e.model_version))
    return sorted(tl, key=lambda t: t[0])


def production_at(timeline: list[tuple[datetime, str]], t: datetime) -> str | None:
    cur = None
    for at, v in timeline:
        if at < t:
            cur = v
        else:
            break
    return cur


def record_prospective(session: Session, pitcher_id: int) -> dict:
    df = load_dated_frame(session, pitcher_id)
    if df.empty:
        return {"games": 0, "recorded_games": 0, "recorded_pitches": 0}
    timeline = production_timeline(session, pitcher_id)
    have = set(session.execute(select(PitchPrediction.pitch_uid, PitchPrediction.model_version)
                               .where(PitchPrediction.context == CONTEXT)).all())
    rep = {"games": 0, "recorded_games": 0, "recorded_pitches": 0, "already_recorded_games": 0,
           "skipped": {"no_production_model_before_game": 0, "model_not_trained_before_game": 0}}
    feat_cols = None
    for g in game_table(df).itertuples():
        rep["games"] += 1
        start = datetime.combine(g.game_date, time(0, 0), tzinfo=timezone.utc)
        v = production_at(timeline, start)
        if v is None:
            rep["skipped"]["no_production_model_before_game"] += 1
            continue
        e = session.get(ModelRegistry, v)
        cut = lifecycle.data_cutoff(session, e)
        if not (e.created_at < start and cut is not None and cut < g.game_date):
            rep["skipped"]["model_not_trained_before_game"] += 1
            continue
        te = df[df.game_pk == g.game_pk]
        if all((int(u), v) in have for u in te.pitch_uid):
            rep["already_recorded_games"] += 1
            continue
        te = te[[(int(u), v) not in have for u in te.pitch_uid]]
        model = registry.load_model(e)
        X = te.drop(columns=["label"])  # the label is never part of the model input
        P = model.predict_proba(X)
        feat_cols = feat_cols or [f.name for f in feature_list(e.classes)]
        rows = []
        for i in range(len(te)):
            probs = {c: float(P[i, j]) for j, c in enumerate(e.classes)}
            top = e.classes[int(P[i].argmax())]
            actual = te.label.iloc[i]
            rows.append(dict(pitch_uid=int(te.pitch_uid.iloc[i]), model_version=v, context=CONTEXT,
                             input_features={k: _py(te[k].iloc[i]) for k in feat_cols if k in te},
                             probabilities=probs, top_pitch=top, actual_pitch=actual, is_correct_top1=top == actual,
                             prob_actual=probs.get(actual), revealed_at=datetime.now(timezone.utc)))
        for i in range(0, len(rows), 1000):
            session.execute(insert(PitchPrediction).values(rows[i:i + 1000]))
        rep["recorded_games"] += 1
        rep["recorded_pitches"] += len(rows)
    session.flush()
    log(logger, "prospective_ledger", pitcher_id=pitcher_id, **{k: v for k, v in rep.items() if k != "skipped"},
        **rep["skipped"])
    return rep


def prospective_metrics(session: Session, pitcher_id: int) -> dict:
    rows = session.execute(select(PitchPrediction.model_version, PitchPrediction.pitch_uid,
                                  PitchPrediction.probabilities, PitchPrediction.actual_pitch,
                                  PitchPrediction.created_at)
                           .join(ModelRegistry, ModelRegistry.model_version == PitchPrediction.model_version)
                           .where(PitchPrediction.context == CONTEXT, ModelRegistry.pitcher_id == pitcher_id)).all()
    base = {"context": CONTEXT, "evaluation_role": "prospective",
            "definition": ("predictions by the production model of each game's date, created and trained before "
                           "that game; computed when the game is ingested")}
    if not rows:
        return {**base, "n": 0, "summary": None, "by_model": [],
                "note": "No prospective predictions yet: no ingested game is newer than a production model's "
                        "creation date and data cutoff."}
    out = []
    df = pd.DataFrame(rows, columns=["model_version", "pitch_uid", "probabilities", "actual", "created_at"])
    for v, part in [("all", df)] + list(df.groupby("model_version")):
        classes = list(part.probabilities.iloc[0].keys())
        P = np.array([[p[c] for c in classes] for p in part.probabilities])
        y = np.array([classes.index(a) for a in part.actual])
        out.append({"model_version": v, **summarize(P, y, classes),
                    "ece": reliability(P, y, classes)["ece_top_label"]})
    return {**base, "n": int(len(df)), "summary": out[0], "by_model": out[1:]}


def _py(v):
    if v is None:
        return None
    if hasattr(v, "item"):
        v = v.item()
    if isinstance(v, float) and np.isnan(v):
        return None
    return v
