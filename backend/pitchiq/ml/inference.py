"""Inference for historical pitches (blind: label never loaded) and for simulator scenarios."""
from __future__ import annotations

import time
from collections import Counter
from datetime import date
from functools import lru_cache

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, field_validator, model_validator
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..features.build import GameState, PAState, history_as_of, make_features
from ..features.schema import FEATURE_VERSION, NONE_TOKEN, feature_list
from ..features.store import load_feature_frame, load_pitch_frame
from ..models import ModelRegistry, PitchPrediction, PrePitchState
from ..pitch_types import pitch_name
from ..trace import Trace
from . import registry

FORBIDDEN_IN_INPUT = {"label", "pitch_type", "actual_pitch"}


# ------------------------------------------------------------------ explanation helpers
def summarize_attribution(expl: dict | None, top: str, classes: list[str], k: int = 12) -> dict | None:
    if not expl:
        return None
    groups = {f.name: f.group for f in feature_list(classes)}
    out = {"space": expl["space"], "top_class": top, "per_class": {}}
    for c, d in expl["per_class"].items():
        contrib = d["contributions"]
        ranked = sorted(contrib.items(), key=lambda kv: -abs(kv[1]))
        by_group: dict[str, float] = {}
        for f, v in contrib.items():
            by_group[groups.get(f, "other")] = by_group.get(groups.get(f, "other"), 0.0) + v
        out["per_class"][c] = {"base": d["base"], "top_features": [{"feature": f, "value": v} for f, v in ranked[:k]],
                               "by_group": dict(sorted(by_group.items(), key=lambda kv: -abs(kv[1])))}
    return out


def _probs(P_row: np.ndarray, classes: list[str]) -> list[dict]:
    items = [{"pitch": c, "pitch_name": pitch_name(c), "probability": float(P_row[j])} for j, c in enumerate(classes)]
    return sorted(items, key=lambda d: -d["probability"])


def _clean(v):
    if v is None:
        return None
    if isinstance(v, float) and np.isnan(v):
        return None
    if hasattr(v, "item"):
        v = v.item()
        return None if isinstance(v, float) and np.isnan(v) else v
    return v


# ------------------------------------------------------------------ historical (blind) prediction
def predict_historical(session: Session, pitcher_id: int, pitch_uid: int, model_version: str | None = None,
                       context: str = "dashboard", save: bool = True, explain: bool = True) -> dict:
    from .lifecycle import out_of_sample_entry, training_includes
    # Default: a model whose training data ended before this pitch's game (production unless it was refit on it).
    entry = registry.get_entry(session, pitcher_id, model_version) if model_version else \
        out_of_sample_entry(session, pitcher_id, pitch_uid=pitch_uid)
    tr = Trace("historical_prediction", entry.model_version)
    with tr.stage("historical_game_state_retrieved", {"pitch_uid": pitch_uid}) as st:
        # Load ONLY the stored pre-pitch feature dict. The label column is deliberately not selected.
        feats = session.scalar(select(PrePitchState.features).where(
            PrePitchState.pitch_uid == pitch_uid, PrePitchState.feature_version == entry.feature_version))
        if feats is None:
            raise LookupError(f"No pre-pitch state for pitch {pitch_uid}")
        st["outputs"] = {"n_fields": len(feats)}
    with tr.stage("pitch_sequence_loaded") as st:
        st["outputs"] = {k: feats.get(k) for k in ("prev1_type", "prev2_type", "prev3_type", "prev1_result")}
    result = _run_model(session, entry, pd.DataFrame([feats]), tr, explain=explain)
    pred_id = None
    if save:
        with tr.stage("prediction_saved") as st:
            pred = PitchPrediction(pitch_uid=pitch_uid, model_version=entry.model_version, context=context,
                                   input_features={k: _clean(v) for k, v in feats.items()},
                                   probabilities={d["pitch"]: d["probability"] for d in result["predictions"]},
                                   top_pitch=result["predictions"][0]["pitch"], attribution=result["attribution"],
                                   inference_ms=result["inference_ms"], trace_id=tr.trace_id)
            session.add(pred)
            session.flush()
            pred_id = pred.prediction_id
            st["outputs"] = {"prediction_id": pred_id, "actual_pitch_known_to_model": False}
    tr.persist(session)
    from ..models import Pitch
    return {**result, "prediction_id": pred_id, "pitch_uid": pitch_uid, "input_features": feats,
            "trace": tr.to_dict(), "model_version": entry.model_version,
            "training_data_includes_game": training_includes(
                session, entry, session.scalar(select(Pitch.game_date).where(Pitch.pitch_uid == pitch_uid)))}


def _run_model(session: Session, entry: ModelRegistry, X: pd.DataFrame, tr: Trace, explain: bool = True) -> dict:
    classes = entry.classes
    with tr.stage("features_constructed") as st:
        st["outputs"] = {"n_features": len(entry.feature_schema.get("used_features", []))}
    with tr.stage("feature_validation_completed") as st:
        bad = FORBIDDEN_IN_INPUT & set(X.columns)
        if bad:
            raise ValueError(f"Leakage guard: forbidden fields in model input: {sorted(bad)}")
        used = entry.feature_schema.get("used_features", [])
        missing = [f for f in used if f not in X.columns]
        if missing:
            raise ValueError(f"Missing features: {missing}")
        nulls = [f for f in used if pd.isna(X[f].iloc[0]) if not isinstance(X[f].iloc[0], str)]
        if nulls:
            st["warnings"].append(f"Missing values (handled by model): {nulls}")
        st["outputs"] = {"checked": len(used), "forbidden_fields_present": False}
    with tr.stage("model_artifact_loaded", {"artifact": entry.artifact_path}) as st:
        model = registry.load_model(entry)
        st["outputs"] = {"family": entry.model_family, "sha256": entry.artifact_sha256[:12]}
    with tr.stage("inference_executed") as st:
        t = time.perf_counter()
        P_raw = model.predict_proba(X, calibrated=False)
        inf_ms = (time.perf_counter() - t) * 1000
        st["outputs"] = {"raw": {c: float(P_raw[0, j]) for j, c in enumerate(classes)}}
    with tr.stage("probabilities_calibrated") as st:
        P = model.predict_proba(X, calibrated=True)
        if not (np.all(P >= 0) and abs(P.sum() - len(P)) < 1e-6):
            raise ValueError("Probabilities not normalized")
        st["outputs"] = {"method": (entry.calibration or {}).get("method"),
                         "temperature": (entry.calibration or {}).get("temperature"),
                         "calibrated": {c: float(P[0, j]) for j, c in enumerate(classes)}}
    preds = _probs(P[0], classes)
    attribution = None
    if explain:
        with tr.stage("feature_attribution_calculated") as st:
            expl = model.explain(X)
            if expl is None:
                st["warnings"].append(f"{entry.model_family} has no per-feature attribution")
            else:
                attribution = summarize_attribution(expl[0], preds[0]["pitch"], classes)
                st["outputs"] = {"method": expl[0]["space"]}
    return {"model_version": entry.model_version, "model_family": entry.model_family,
            "calibration_status": f"temperature_scaled(T={entry.calibration['temperature']:.3f}) on validation"
            if entry.calibration else "uncalibrated",
            "predictions": preds, "attribution": attribution, "inference_ms": inf_ms}


# ------------------------------------------------------------------ simulator
class PrevPitch(BaseModel):
    type: str
    result: str = Field("ball", description="ball|called_strike|swinging_strike|foul|in_play|hbp")
    speed: float | None = None
    same_pa: bool = True


class SimState(BaseModel):
    stand: str = "R"
    bat_order_slot: int = Field(1, ge=1, le=9)
    batter_id: int | None = None
    times_through_order: int = Field(1, ge=1, le=5)
    balls: int = Field(0, ge=0, le=3)
    strikes: int = Field(0, ge=0, le=2)
    inning: int = Field(1, ge=1, le=12)
    outs: int = Field(0, ge=0, le=2)
    on_1b: bool = False
    on_2b: bool = False
    on_3b: bool = False
    score_diff: int = Field(0, ge=-15, le=15)
    is_home: bool = True
    pitch_count_before: int = Field(0, ge=0, le=140)
    previous_pitches: list[PrevPitch] = Field(default_factory=list, description="oldest first")
    as_of_date: date | None = None
    days_rest: float | None = 5.0

    @field_validator("stand")
    @classmethod
    def _stand(cls, v):
        if v not in ("L", "R"):
            raise ValueError("stand must be L or R")
        return v

    @model_validator(mode="after")
    def _consistent(self):
        pa = [p for p in self.previous_pitches if p.same_pa]
        if any(not p.same_pa for p in self.previous_pitches[self.previous_pitches.index(pa[0]):]) if pa else False:
            raise ValueError("Same-PA pitches must be the most recent ones in the sequence")
        b = s = 0
        for p in pa:
            if p.result in ("in_play", "hbp"):
                raise ValueError("A plate appearance cannot continue after a ball in play / HBP")
            if p.result == "ball":
                b += 1
            elif p.result in ("called_strike", "swinging_strike"):
                s += 1
            elif p.result == "foul":
                s = min(s + 1, 2)
            if b > 3 or s > 2:
                raise ValueError("Sequence would have ended the plate appearance")
        if pa and (b, s) != (self.balls, self.strikes):
            raise ValueError(f"Previous same-PA pitches imply a {b}-{s} count, but count is {self.balls}-{self.strikes}")
        if not pa and (self.balls or self.strikes):
            pass  # count given without explicit sequence: allowed, PA sequence features treated as unknown
        if len(self.previous_pitches) > self.pitch_count_before:
            raise ValueError("More previous pitches than the cumulative pitch count")
        if self.inning == 1 and self.times_through_order > 1:
            raise ValueError("Cannot be past first time through the order in the 1st inning")
        if self.times_through_order > 1 and self.pitch_count_before < 9:
            raise ValueError("Times through order > 1 requires at least 9 prior pitches")
        return self


@lru_cache(maxsize=8)
def _history_cached(pitcher_id: int, as_of: date, classes: tuple, _key: str):
    """(History counters, mean velocity per class) from games strictly before `as_of`."""
    from ..db import session_scope
    from ..features.build import map_label
    with session_scope() as s:
        df = load_pitch_frame(s, pitcher_id)
    before = df[df["game_date"] < as_of].copy()
    before["cls"] = [map_label(t, list(classes)) for t in before["pitch_type"]]
    velo = before.groupby("cls")["release_speed"].mean().dropna().to_dict()
    return history_as_of(df, list(classes), as_of), velo


@lru_cache(maxsize=4)
def _train_frame_cached(pitcher_id: int, feature_version: str, _key: str) -> pd.DataFrame:
    from ..db import session_scope
    with session_scope() as s:
        return load_feature_frame(s, pitcher_id, ["train"], feature_version)


def default_as_of(session: Session, pitcher_id: int) -> date:
    from ..config import get_settings
    from ..models import Pitch
    first_val = min(get_settings().validation_seasons)
    d = session.scalar(select(Pitch.game_date).where(Pitch.pitcher_id == pitcher_id, Pitch.season >= first_val)
                       .order_by(Pitch.game_date).limit(1))
    return d or date.today()


def _unknown_prev_distribution(pitcher_id: int, feature_version: str, key: str, count_str: str,
                               max_combos: int = 12) -> list[tuple[float, str, str, int]]:
    """Empirical joint of (prev1_type, prev1_result, prev1_same_pa) in training rows with this count and
    at least one prior pitch. Used to marginalize the model over an unknown previous pitch."""
    tr = _train_frame_cached(pitcher_id, feature_version, key)
    sub = tr[(tr["count_str"] == count_str) & (tr["pitch_count_before"] > 0)]
    if len(sub) < 20:
        sub = tr[tr["pitch_count_before"] > 0]
    vc = sub.groupby(["prev1_type", "prev1_result", "prev1_same_pa"]).size().sort_values(ascending=False)
    vc = vc.iloc[:max_combos]
    w = vc / vc.sum()
    return [(float(w.iloc[i]), *vc.index[i]) for i in range(len(vc))]


def sim_feature_mixture(session: Session, entry: ModelRegistry, state: SimState) -> tuple[list[tuple[float, dict]], dict]:
    """Returns [(weight, feature_dict)] (one element unless the previous pitch is unknown) + notes."""
    classes = entry.classes
    pitcher_id = entry.pitcher_id
    as_of = state.as_of_date or default_as_of(session, pitcher_id)
    hist, velo = _history_cached(pitcher_id, as_of, tuple(classes), "v")
    notes = {"as_of_date": str(as_of), "imputed": [], "marginalized_over": None}

    def build(prev: list[PrevPitch]) -> dict:
        game, pa = GameState(), PAState()
        for p in prev:
            t = p.type if p.type in classes else "OTHER"
            spd = p.speed if p.speed is not None else velo.get(t)
            game.types.append(t)
            game.results.append(p.result)
            game.speeds.append(spd)
            game.ab_numbers.append(1 if p.same_pa else 0)
            if t == "FF" and spd:
                game.ff_speeds.append(spd)
            game.usage[t] += 1
            if p.same_pa:
                pa.usage[t] += 1
        if not game.ff_speeds and state.pitch_count_before > 0 and velo.get("FF"):
            game.ff_speeds.append(velo["FF"])
        n_pa = sum(1 for p in prev if p.same_pa)
        ctx = {
            "stand": state.stand, "bat_order_slot": state.bat_order_slot, "balls": state.balls,
            "strikes": state.strikes, "inning": state.inning, "inning_topbot": "Top" if state.is_home else "Bot",
            "outs_when_up": state.outs, "on_1b": state.on_1b, "on_2b": state.on_2b, "on_3b": state.on_3b,
            "fld_score": state.score_diff, "bat_score": 0, "at_bat_number": 1,
            "pitch_number": max(n_pa + 1, state.balls + state.strikes + 1),
            "n_thruorder_pitcher": state.times_through_order, "season": as_of.year, "days_rest": state.days_rest,
            "batter": state.batter_id, "pitch_count_before": state.pitch_count_before,
        }
        return make_features(ctx, hist, game, pa, classes)

    if any(p.speed is None for p in state.previous_pitches):
        notes["imputed"].append("previous pitch velocity = pitcher's historical mean for that pitch type")
    if state.pitch_count_before > 0 and not state.previous_pitches:
        mix = []
        combos = _unknown_prev_distribution(pitcher_id, entry.feature_version, entry.model_version,
                                            f"{state.balls}-{state.strikes}")
        for w, t, r, same in combos:
            mix.append((w, build([PrevPitch(type=t, result=r, same_pa=bool(same))])))
        tot = sum(w for w, _ in mix)
        mix = [(w / tot, f) for w, f in mix]
        notes["marginalized_over"] = {
            "reason": "pitch count > 0 but previous pitch not specified",
            "method": "weighted average of model outputs over the training distribution of "
                      "(previous type, previous result, same PA) in this count",
            "combinations": [{"prev1_type": t, "prev1_result": r, "prev1_same_pa": bool(same), "weight": w / tot}
                             for w, t, r, same in combos]}
        return mix, notes
    return [(1.0, build(state.previous_pitches))], notes


def ood_report(session: Session, entry: ModelRegistry, feats: dict) -> dict:
    tr = _train_frame_cached(entry.pitcher_id, entry.feature_version, entry.model_version)
    warnings = []
    same_cell = int(((tr["stand"] == feats["stand"]) & (tr["count_str"] == feats["count_str"])).sum())
    near_pc = int(((tr["pitch_count_before"] - feats["pitch_count_before"]).abs() <= 5).sum())
    inn = int((tr["inning"] == feats["inning"]).sum())
    exact = int(((tr["stand"] == feats["stand"]) & (tr["count_str"] == feats["count_str"]) &
                 (tr["inning"] == feats["inning"]) &
                 ((tr["pitch_count_before"] - feats["pitch_count_before"]).abs() <= 10)).sum())
    if same_cell < 30:
        warnings.append(f"Only {same_cell} training pitches with {feats['stand']}HB in a {feats['count_str']} count")
    if near_pc < 30:
        warnings.append(f"Only {near_pc} training pitches within ±5 of pitch count {feats['pitch_count_before']}")
    if inn < 30:
        warnings.append(f"Only {inn} training pitches in inning {feats['inning']}")
    if feats["pitch_count_before"] > tr["pitch_count_before"].max():
        warnings.append("Pitch count exceeds anything in the training data")
    if (feats.get("times_through_order") or 0) > tr["times_through_order"].max():
        warnings.append("Times through order exceeds training data")
    return {"training_rows": len(tr), "same_hand_and_count": same_cell, "pitch_count_within_5": near_pc,
            "same_inning": inn, "similar_situations": exact, "out_of_distribution": bool(warnings),
            "warnings": warnings}


def _mixture_predict(model, mix):
    X = pd.DataFrame([f for _, f in mix])
    w = np.array([w for w, _ in mix])[:, None]
    return (model.predict_proba(X) * w).sum(axis=0, keepdims=True), X


def predict_scenario(session: Session, pitcher_id: int, state: SimState, model_version: str | None = None,
                     explain: bool = True, persist_trace: bool = True) -> dict:
    entry = registry.get_entry(session, pitcher_id, model_version)
    tr = Trace("simulator_prediction", entry.model_version)
    with tr.stage("historical_game_state_retrieved", {"as_of_date": str(state.as_of_date)}) as st:
        mix, notes = sim_feature_mixture(session, entry, state)
        st["outputs"] = {"source": "user-constructed scenario; pitcher history strictly before as-of date",
                         **notes}
    with tr.stage("pitch_sequence_loaded", {"previous_pitches": len(state.previous_pitches)}) as st:
        st["outputs"] = {"mixture_components": len(mix)}
    if len(mix) == 1:
        res = _run_model(session, entry, pd.DataFrame([mix[0][1]]), tr, explain=explain)
    else:
        res = _run_model(session, entry, pd.DataFrame([mix[0][1]]), tr, explain=explain)
        with tr.stage("unknown_previous_pitch_marginalized", {"components": len(mix)}) as st:
            model = registry.load_model(entry)
            P, X = _mixture_predict(model, mix)
            res["predictions"] = _probs(P[0], entry.classes)
            if explain and res["attribution"] is not None:
                ex = model.explain(X)
                w = [w for w, _ in mix]
                agg = {"space": ex[0]["space"] + " (mixture-weighted)", "per_class": {}}
                for c in entry.classes:
                    contrib: dict[str, float] = {}
                    base = 0.0
                    for wi, e in zip(w, ex):
                        base += wi * e["per_class"][c]["base"]
                        for f, v in e["per_class"][c]["contributions"].items():
                            contrib[f] = contrib.get(f, 0.0) + wi * v
                    agg["per_class"][c] = {"base": base, "contributions": contrib}
                res["attribution"] = summarize_attribution(agg, res["predictions"][0]["pitch"], entry.classes)
            st["outputs"] = {"marginal": {d["pitch"]: d["probability"] for d in res["predictions"]}}
    feats = mix[0][1]
    ood = ood_report(session, entry, feats)
    if persist_trace:
        tr.persist(session)
    return {**res, "input_features": {k: _clean(v) for k, v in feats.items()}, "support": ood, "trace": tr.to_dict(),
            "assumptions": notes, "note": "Simulator output shows model sensitivity, not causal effects."}


def sweep(session: Session, pitcher_id: int, state: SimState, field: str, values: list,
          model_version: str | None = None) -> dict:
    """Vary one simulator field holding everything else constant; batched inference."""
    entry = registry.get_entry(session, pitcher_id, model_version)
    model = registry.load_model(entry)
    points = []
    t = time.perf_counter()
    for v in values:
        d = state.model_dump()
        if field == "count":
            d["balls"], d["strikes"] = v
            d["previous_pitches"] = [p for p in d["previous_pitches"] if not p["same_pa"]]
        else:
            d[field] = v
        try:
            st = SimState(**d)
        except Exception:
            continue
        mix, _ = sim_feature_mixture(session, entry, st)
        P, _ = _mixture_predict(model, mix)
        points.append({"value": v if field != "count" else f"{v[0]}-{v[1]}",
                       "probabilities": {c: float(P[0, j]) for j, c in enumerate(entry.classes)}})
    ms = (time.perf_counter() - t) * 1000
    return {"model_version": entry.model_version, "field": field, "classes": entry.classes,
            "points": points, "inference_ms": ms}


def models_for_comparison(session: Session, pitcher_id: int) -> list[ModelRegistry]:
    return list(registry.latest_by_family(session, pitcher_id).values())
