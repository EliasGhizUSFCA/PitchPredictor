"""PitchIQ Prediction Challenge: blind sampling, frozen predictions, reveal, live scoring."""
from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from .ml import registry
from .ml.inference import predict_historical
from .ml.metrics import brier, confusion, log_loss, reliability, summarize, wilson
from .models import (Batter, ChallengeAttempt, Pitch, PitchPrediction, PredictionChallenge, PrePitchState,
                     VideoAsset)
from .pitch_types import pitch_name, result_category

STRATEGIES = {"natural", "balanced_pitch_type", "balanced_count", "lhb", "rhb", "innings", "pitch_count_range"}


def _eligible(session: Session, pitcher_id: int, split: str, feature_version: str) -> pd.DataFrame:
    rows = session.execute(
        select(Pitch.pitch_uid, Pitch.stand, Pitch.balls, Pitch.strikes, Pitch.inning, Pitch.game_pitch_seq,
               PrePitchState.label, PrePitchState.features["pitch_count_before"].astext.label("pc"))
        .join(PrePitchState, PrePitchState.pitch_uid == Pitch.pitch_uid)
        .where(Pitch.pitcher_id == pitcher_id, PrePitchState.split == split,
               PrePitchState.feature_version == feature_version, PrePitchState.label.is_not(None))
    ).all()
    return pd.DataFrame(rows, columns=["pitch_uid", "stand", "balls", "strikes", "inning", "seq", "label", "pc"])


def create_challenge(session: Session, pitcher_id: int, size: int, strategy: str = "natural",
                     params: dict | None = None, seed: int | None = None, model_version: str | None = None,
                     split: str = "test") -> PredictionChallenge:
    if strategy not in STRATEGIES:
        raise ValueError(f"Unknown strategy {strategy}")
    if not 1 <= size <= 1000:
        raise ValueError("size must be 1..1000")
    params = params or {}
    if model_version is None:  # default: a model whose training data ended before the challenge split
        from .ml.lifecycle import out_of_sample_entry
        entry = out_of_sample_entry(session, pitcher_id, split=split)
    else:
        entry = registry.get_entry(session, pitcher_id, model_version)
    seed = int(seed if seed is not None else np.random.SeedSequence().entropy % (2**31))
    rng = np.random.default_rng(seed)
    df = _eligible(session, pitcher_id, split, entry.feature_version)
    if strategy == "lhb":
        df = df[df.stand == "L"]
    elif strategy == "rhb":
        df = df[df.stand == "R"]
    elif strategy == "innings":
        df = df[df.inning.isin(params.get("innings", [1]))]
    elif strategy == "pitch_count_range":
        lo, hi = params.get("min", 0), params.get("max", 200)
        df = df[(df.pc.astype(float) >= lo) & (df.pc.astype(float) <= hi)]
    if df.empty:
        raise ValueError("No eligible pitches for this strategy")
    if strategy in ("balanced_pitch_type", "balanced_count"):
        # Labels are used ONLY to choose which pitches enter the session; the model never sees them.
        key = df.label if strategy == "balanced_pitch_type" else df.balls.astype(str) + "-" + df.strikes.astype(str)
        groups = [g for _, g in df.groupby(key)]
        picks = []
        while len(picks) < min(size, len(df)):
            for g in groups:
                remaining = g[~g.pitch_uid.isin(picks)]
                if len(remaining) and len(picks) < size:
                    picks.append(int(remaining.pitch_uid.iloc[rng.integers(len(remaining))]))
        uids = [int(x) for x in rng.permutation(picks)]
    else:
        uids = [int(x) for x in rng.choice(df.pitch_uid.to_numpy(), size=min(size, len(df)), replace=False)]
    ch = PredictionChallenge(pitcher_id=pitcher_id, model_version=entry.model_version, size=len(uids),
                             sampling_strategy=strategy, strategy_params=params, seed=seed, split=split,
                             pitch_uids=uids)
    session.add(ch)
    session.flush()
    return ch


def pre_pitch_situation(session: Session, pitch_uid: int) -> dict:
    """Display-only pre-pitch context. Contains NOTHING about the target pitch's type or outcome."""
    p = session.get(Pitch, pitch_uid)
    b = session.get(Batter, p.batter_id)
    feats = session.scalar(select(PrePitchState.features).where(PrePitchState.pitch_uid == pitch_uid))
    prior = session.execute(select(Pitch.pitch_type, Pitch.description, Pitch.at_bat_number, Pitch.release_speed,
                                   Pitch.balls, Pitch.strikes)
                            .where(Pitch.game_pk == p.game_pk, Pitch.pitcher_id == p.pitcher_id,
                                   Pitch.game_pitch_seq < p.game_pitch_seq)
                            .order_by(Pitch.game_pitch_seq.desc()).limit(8)).all()
    pitching_home = p.inning_topbot == "Top"
    return {
        "pitch_uid": p.pitch_uid, "game_pk": p.game_pk, "game_date": str(p.game_date), "season": p.season,
        "pitcher_team": p.home_team if pitching_home else p.away_team,
        "opponent": p.away_team if pitching_home else p.home_team, "is_home": pitching_home,
        "batter": {"id": p.batter_id, "name": b.full_name if b else None, "stand": p.stand},
        "batting_order_slot": feats.get("bat_order_slot") if feats else None,
        "times_through_order": p.n_thruorder_pitcher,
        "inning": p.inning, "half": p.inning_topbot, "outs": p.outs_when_up,
        "count": {"balls": p.balls, "strikes": p.strikes},
        "pitch_count_before": feats.get("pitch_count_before") if feats else None,
        "pa_pitch_number": p.pitch_number, "at_bat_number": p.at_bat_number,
        "runners": {"on_1b": p.on_1b is not None, "on_2b": p.on_2b is not None, "on_3b": p.on_3b is not None},
        "score": {"pitcher_team": p.fld_score, "batting_team": p.bat_score},
        "previous_pitches": [{"pitch_type": r.pitch_type, "pitch_name": pitch_name(r.pitch_type),
                              "result": result_category(r.description), "speed": r.release_speed,
                              "same_pa": r.at_bat_number == p.at_bat_number,
                              "count": f"{r.balls}-{r.strikes}"} for r in prior if r.pitch_type],
    }


def best_video(session: Session, pitch_uid: int) -> VideoAsset | None:
    order = {"licensed_local": 0, "official_embed": 1, "external_link": 2}
    assets = session.scalars(select(VideoAsset).where(VideoAsset.pitch_uid == pitch_uid,
                                                      VideoAsset.match_confidence == "exact_verified",
                                                      VideoAsset.video_availability_status.notin_(["unavailable"])))
    assets = sorted(assets, key=lambda a: order.get(a.playback_method, 9))
    return assets[0] if assets else None


def video_mode(a: VideoAsset | None) -> str:
    if a is None:
        return "data_only"
    if a.playback_method in ("licensed_local", "official_embed"):
        return "in_app"
    return "external"


def next_attempt(session: Session, challenge_id: int) -> dict:
    ch = session.get(PredictionChallenge, challenge_id)
    if ch is None:
        raise LookupError("challenge not found")
    open_attempt = session.scalars(select(ChallengeAttempt).where(ChallengeAttempt.challenge_id == challenge_id,
                                                                  ChallengeAttempt.revealed.is_(False))).first()
    if open_attempt:
        return attempt_view(session, open_attempt)
    pos = session.query(ChallengeAttempt).filter_by(challenge_id=challenge_id).count()
    if pos >= len(ch.pitch_uids):
        ch.status = "completed"
        return {"done": True, "challenge_id": challenge_id}
    uid = ch.pitch_uids[pos]
    # Step 4/5: predict from pre-pitch data and freeze BEFORE anything about the outcome is read.
    pred = predict_historical(session, ch.pitcher_id, uid, ch.model_version, context="challenge", save=True)
    video = best_video(session, uid)
    att = ChallengeAttempt(challenge_id=challenge_id, position=pos, pitch_uid=uid,
                           prediction_id=pred["prediction_id"], video_asset_id=video.video_asset_id if video else None,
                           mode=video_mode(video))
    session.add(att)
    session.flush()
    return attempt_view(session, att, pred)


def attempt_view(session: Session, att: ChallengeAttempt, pred: dict | None = None) -> dict:
    pp = session.get(PitchPrediction, att.prediction_id)
    ch = session.get(PredictionChallenge, att.challenge_id)
    view = {
        "attempt_id": att.attempt_id, "challenge_id": att.challenge_id, "position": att.position,
        "total": ch.size, "mode": att.mode, "revealed": att.revealed, "user_guess": att.user_guess,
        "situation": pre_pitch_situation(session, att.pitch_uid),
        "prediction": {
            "prediction_id": pp.prediction_id, "model_version": pp.model_version, "frozen_at": str(pp.created_at),
            "predictions": sorted([{"pitch": k, "pitch_name": pitch_name(k), "probability": v}
                                   for k, v in pp.probabilities.items()], key=lambda d: -d["probability"]),
            "attribution": pp.attribution, "inference_ms": pp.inference_ms, "trace_id": pp.trace_id,
        },
        "trace": pred["trace"] if pred else None,
        # The video reference is withheld until reveal so the clip cannot be opened before the prediction.
        # In-app (licensed, aligned) clips may expose ONLY the attempt-scoped pre-pitch segment URL before reveal.
        "video": {"available": att.video_asset_id is not None, "mode": att.mode, **_pre_pitch_video(session, att)},
    }
    if att.revealed:
        view.update(reveal_payload(session, att))
    return view


def _pre_pitch_video(session: Session, att: ChallengeAttempt) -> dict:
    from .video.align_service import has_pre_pitch_segment
    v = session.get(VideoAsset, att.video_asset_id) if att.video_asset_id and att.mode == "in_app" else None
    ok = has_pre_pitch_segment(v)
    return {"pre_pitch_segment_available": ok,
            "pre_pitch_media_url": f"/api/attempts/{att.attempt_id}/pre-pitch-media" if ok else None}


def set_guess(session: Session, attempt_id: int, guess: str) -> None:
    att = session.get(ChallengeAttempt, attempt_id)
    if att.revealed:
        raise ValueError("Cannot change the guess after reveal")
    att.user_guess = guess


def reveal_payload(session: Session, att: ChallengeAttempt) -> dict:
    from .video.align_service import has_pre_pitch_segment
    p = session.get(Pitch, att.pitch_uid)
    pp = session.get(PitchPrediction, att.prediction_id)
    label = session.scalar(select(PrePitchState.label).where(PrePitchState.pitch_uid == att.pitch_uid))
    v = session.get(VideoAsset, att.video_asset_id) if att.video_asset_id else None
    return {
        "actual": {"pitch_type": p.pitch_type, "pitch_name": pitch_name(p.pitch_type), "model_class": label,
                   "release_speed": p.release_speed, "spin_rate": p.release_spin_rate,
                   "result": p.description, "event": p.events, "description": p.des,
                   "plate_x": p.plate_x, "plate_z": p.plate_z, "source": "Statcast (Baseball Savant)"},
        "score": {"top1_correct": pp.is_correct_top1, "prob_actual": pp.prob_actual,
                  "rank_actual": sorted(pp.probabilities, key=lambda k: -pp.probabilities[k]).index(label) + 1
                  if label in pp.probabilities else None,
                  "user_guess_correct": (att.user_guess == label) if att.user_guess else None},
        "video_reference": None if v is None else {
            "video_asset_id": v.video_asset_id, "source_provider": v.source_provider,
            "playback_method": v.playback_method, "source_page_url": v.source_page_url,
            "authorized_embed_reference": v.authorized_embed_reference,
            "media_url": f"/api/videos/{v.video_asset_id}/media?segment=full"
            if v.playback_method == "licensed_local" else None,
            "pre_pitch_media_url": f"/api/videos/{v.video_asset_id}/media" if has_pre_pitch_segment(v) else None,
            "synchronization_status": v.synchronization_status, "alignment_confidence": v.alignment_confidence,
            "pitch_end_time": v.pitch_end_time,
            "pre_pitch_pause_time": v.pre_pitch_pause_time, "release_frame_timestamp": v.release_frame_timestamp,
            "match_confidence": v.match_confidence, "rights": v.rights_or_permission_status,
            "availability": v.video_availability_status},
    }


def reveal(session: Session, attempt_id: int) -> dict:
    att = session.get(ChallengeAttempt, attempt_id)
    if att is None:
        raise LookupError("attempt not found")
    pp = session.get(PitchPrediction, att.prediction_id)
    if pp is None or pp.created_at is None:
        raise RuntimeError("Prediction must be frozen before reveal")
    if not att.revealed:
        label = session.scalar(select(PrePitchState.label).where(PrePitchState.pitch_uid == att.pitch_uid))
        pp.actual_pitch = label
        pp.is_correct_top1 = pp.top_pitch == label
        pp.prob_actual = float(pp.probabilities.get(label, 0.0))
        pp.revealed_at = datetime.now(timezone.utc)
        att.revealed = True
        att.revealed_at = pp.revealed_at
        session.flush()
    return attempt_view(session, att)


def challenge_stats(session: Session, challenge_id: int) -> dict:
    ch = session.get(PredictionChallenge, challenge_id)
    rows = session.execute(select(ChallengeAttempt, PitchPrediction)
                           .join(PitchPrediction, PitchPrediction.prediction_id == ChallengeAttempt.prediction_id)
                           .where(ChallengeAttempt.challenge_id == challenge_id, ChallengeAttempt.revealed.is_(True))
                           .order_by(ChallengeAttempt.position)).all()
    entry = registry.get_entry(session, ch.pitcher_id, ch.model_version)
    classes = entry.classes
    base = {"challenge_id": challenge_id, "model_version": ch.model_version, "sampling_strategy": ch.sampling_strategy,
            "seed": ch.seed, "size": ch.size, "tested": len(rows), "classes": classes,
            "natural_frequencies": ch.sampling_strategy not in ("balanced_pitch_type", "balanced_count"),
            "note": "Interactive subset; official performance comes from the full holdout benchmark."}
    if not rows:
        return base
    P = np.array([[pp.probabilities.get(c, 0.0) for c in classes] for _, pp in rows])
    y = np.array([classes.index(pp.actual_pitch) for _, pp in rows])
    s = summarize(P, y, classes)
    correct = (P.argmax(1) == y).astype(int)
    cum = np.cumsum(correct) / np.arange(1, len(correct) + 1)
    per_type = {}
    for j, c in enumerate(classes):
        m = y == j
        if m.sum():
            k = int(correct[m].sum())
            per_type[c] = {"n": int(m.sum()), "correct": k, "accuracy": k / int(m.sum()),
                           "ci95": list(wilson(k, int(m.sum())))}
    # Baselines on the SAME pitches (frequency + majority), computed from stored pre-pitch states.
    baselines = {}
    uids = [att.pitch_uid for att, _ in rows]
    feats = pd.DataFrame([f for (f,) in session.execute(select(PrePitchState.features).where(
        PrePitchState.pitch_uid.in_(uids)).order_by(PrePitchState.pitch_uid))])
    order = sorted(uids)
    idx = [order.index(u) for u in uids]
    feats = feats.iloc[idx].reset_index(drop=True)
    for fam, e in registry.latest_by_family(session, ch.pitcher_id).items():
        if fam in ("majority", "frequency") and e.model_version != ch.model_version:
            bm = registry.load_model(e)
            Pb = bm.predict_proba(feats)
            baselines[fam] = {"model_version": e.model_version, **summarize(Pb, y, classes)}
    guesses = [(att.user_guess, pp.actual_pitch) for att, pp in rows if att.user_guess]
    return {**base, **s, "per_pitch_type": per_type,
            "cumulative_accuracy": [float(x) for x in cum],
            "confusion": confusion(P, y, classes), "calibration": reliability(P, y, classes, n_bins=5),
            "baselines_same_pitches": baselines,
            "user": {"guesses": len(guesses), "correct": sum(g == a for g, a in guesses),
                     "accuracy": (sum(g == a for g, a in guesses) / len(guesses)) if guesses else None},
            "attempts": [{"position": att.position, "pitch_uid": att.pitch_uid, "top_pitch": pp.top_pitch,
                          "actual": pp.actual_pitch, "correct": pp.is_correct_top1, "prob_actual": pp.prob_actual,
                          "user_guess": att.user_guess, "mode": att.mode} for att, pp in rows]}
