"""Model lifecycle: data cutoffs, predetermined promotion gates, gated promotion, rollback, MLflow aliases.

Stages: candidate -> production (stored status "promoted") -> previous_production (one rollback slot) -> candidate.
Models are never deleted (archive policy: keep).

Gates are fixed in `Settings` (gate_*) and only read development evidence:

* ``artifact_checksum`` - the artifact loads and its SHA-256 matches the registry.
* ``probabilities_normalized`` - on every labeled development pitch: finite, in [0, 1], rows sum to 1 (tolerance).
* ``min_training_samples`` - the fit used at least `gate_min_training_samples` labeled pitches.
* ``dev_ece`` - top-label ECE of the deployed output on the recipe's rolling development predictions (raw if the
  calibration rule left calibration off, else chronologically calibrated).
* ``dev_log_loss_vs_production`` - candidate dev rolling log loss <= production's (same protocol, same folds) +
  `gate_max_dev_log_loss_regression`.
* ``data_not_older_than_production`` - the candidate's data cutoff is not earlier than production's.
* ``development_only_provenance`` - selection and calibration evidence lies inside the development seasons.
"""
from __future__ import annotations

import logging
from datetime import date, datetime, timezone

import numpy as np
from sqlalchemy import func, select, update
from sqlalchemy.orm import Session, object_session

from ..config import get_settings
from ..logging_setup import log
from ..models import EvaluationRun, ModelRegistry, Pitch, PrePitchState
from ..models_lifecycle import STAGE_BY_STATUS, ModelLifecycle, ModelPromotionEvent
from . import registry
from .protocol import gate_thresholds

logger = logging.getLogger(__name__)

LIFECYCLE_SCOPES = ("deployment_refit", "scheduled_retrain")
PRODUCTION_ALIAS = "production"
PREVIOUS_ALIAS = "previous_production"


def _now() -> datetime:
    return datetime.now(timezone.utc)


# ------------------------------------------------------------------ provenance
def lifecycle_row(session: Session, version: str) -> ModelLifecycle | None:
    return session.get(ModelLifecycle, version)


def is_lifecycle_model(entry: ModelRegistry) -> bool:
    return (entry.config or {}).get("training_scope") in LIFECYCLE_SCOPES


def data_cutoff(session: Session, entry: ModelRegistry) -> date | None:
    """Last game date whose pitches influenced the model (fit, early stopping or calibration).

    Lifecycle models store it. Legacy season-split models used the splits named in their dataset fingerprint
    (train + validation), so the cutoff is the last game date in those splits."""
    lc = lifecycle_row(session, entry.model_version)
    if lc is not None:
        return lc.data_cutoff
    splits = [k for k in (entry.dataset_fingerprint or {}) if k in ("train", "validation", "test", "unassigned")]
    if not splits:
        return None
    return session.scalar(select(func.max(Pitch.game_date)).join(PrePitchState, PrePitchState.pitch_uid ==
                                                                  Pitch.pitch_uid)
                          .where(PrePitchState.pitcher_id == entry.pitcher_id, PrePitchState.split.in_(splits),
                                 PrePitchState.feature_version == entry.feature_version))


def _first_date(session: Session, pitcher_id: int, pitch_uid: int | None = None, split: str | None = None):
    if pitch_uid is not None:
        return session.scalar(select(Pitch.game_date).where(Pitch.pitch_uid == pitch_uid))
    return session.scalar(select(func.min(Pitch.game_date)).join(PrePitchState, PrePitchState.pitch_uid ==
                                                                  Pitch.pitch_uid)
                          .where(PrePitchState.pitcher_id == pitcher_id, PrePitchState.split == split))


def out_of_sample_entry(session: Session, pitcher_id: int, *, before: date | None = None,
                        pitch_uid: int | None = None, split: str | None = None) -> ModelRegistry:
    """The model to use for a *historical* pitch/split when no version is requested.

    Production is used if its data cutoff precedes `before`; otherwise the newest non-archived model whose data
    ended before it (previous production first, then production's family). If no such model exists, production is
    returned and `training_includes(...)` reports the overlap (blind replay on early seasons keeps working)."""
    prod = registry.get_entry(session, pitcher_id, None)
    before = before or _first_date(session, pitcher_id, pitch_uid, split)
    if before is None:
        return prod
    cut = data_cutoff(session, prod)
    if cut is not None and cut < before:
        return prod
    rows = session.scalars(select(ModelRegistry).where(ModelRegistry.pitcher_id == pitcher_id,
                                                       ModelRegistry.status != "archived")
                           .order_by(ModelRegistry.created_at.desc())).all()
    ok = [r for r in rows if not (r.config or {}).get("ablation")
          and (c := data_cutoff(session, r)) is not None and c < before]
    if not ok:
        return prod
    ok.sort(key=lambda r: (r.status != "previous_production", r.model_family != prod.model_family))
    return ok[0]


def training_includes(session: Session, entry: ModelRegistry, game_date: date | None) -> bool | None:
    cut = data_cutoff(session, entry)
    if cut is None or game_date is None:
        return None
    return game_date <= cut


# ------------------------------------------------------------------ dev evidence
def dev_evaluation(session: Session, entry: ModelRegistry) -> dict | None:
    """Rolling development metrics of the model's recipe (stored with the lifecycle row or as an evaluation run)."""
    lc = lifecycle_row(session, entry.model_version)
    if lc is not None and lc.dev_metrics:
        return lc.dev_metrics
    run = session.scalars(select(EvaluationRun).where(EvaluationRun.model_version == entry.model_version,
                                                      EvaluationRun.kind == "rolling_dev_evaluation")
                          .order_by(EvaluationRun.created_at.desc())).first()
    if run is not None:
        return run.metrics.get("dev_metrics")
    return None


def ensure_dev_evaluation(session: Session, entry: ModelRegistry) -> dict:
    """Compute (once) and store the rolling development evaluation of a model's frozen recipe."""
    d = dev_evaluation(session, entry)
    if d is not None:
        return d
    from .refit import evaluate_recipe_on_dev
    run = evaluate_recipe_on_dev(session, entry)
    return run.metrics["dev_metrics"]


# ------------------------------------------------------------------ gates
def evaluate_gates(session: Session, entry: ModelRegistry, production: ModelRegistry | None = None) -> dict:
    s = get_settings()
    th = gate_thresholds()
    production = production if production is not None else registry.promoted(session, entry.pitcher_id)
    if production is not None and production.model_version == entry.model_version:
        production = None
    gates = []

    def gate(name, passed, value=None, threshold=None, detail=None):
        gates.append({"gate": name, "passed": bool(passed), "value": value, "threshold": threshold,
                      "detail": detail})

    model = None
    try:
        model = registry.load_model(entry)
        gate("artifact_checksum", True, entry.artifact_sha256[:16])
    except Exception as e:  # noqa: BLE001 - any load failure fails the gate
        gate("artifact_checksum", False, detail=f"{type(e).__name__}: {e}")

    if model is not None:
        from .protocol import load_dated_frame
        df = load_dated_frame(session, entry.pitcher_id, entry.feature_version)
        dev = df[df.game_season.isin(s.development_seasons)]
        P = model.predict_proba(dev)
        finite = bool(np.isfinite(P).all())
        rng = bool((P >= 0).all() and (P <= 1).all())
        err = float(np.abs(P.sum(1) - 1).max()) if len(P) else 0.0
        gate("probabilities_normalized", finite and rng and err <= th["prob_tolerance"], err, th["prob_tolerance"],
             f"{len(P)} development pitches; finite={finite}, within [0,1]={rng}")
    else:
        gate("probabilities_normalized", False, detail="artifact did not load")

    lc = lifecycle_row(session, entry.model_version)
    n_train = lc.n_train_samples if lc is not None else sum(
        int(v.get("n", 0)) for k, v in (entry.dataset_fingerprint or {}).items()
        if k in ("train", "validation") and isinstance(v, dict))
    gate("min_training_samples", n_train >= th["min_training_samples"], n_train, th["min_training_samples"])

    dev = ensure_dev_evaluation(session, entry)
    applied = dev.get("calibration_applied")
    kind = ("chronologically calibrated" if applied else "raw, calibration_applied=false" if applied is False
            else "calibration state not recorded")
    gate("dev_ece", dev["ece"] <= th["max_dev_ece"], dev["ece"], th["max_dev_ece"],
         f"top-label ECE of the deployed output on rolling development predictions ({kind})")

    if production is None:
        gate("dev_log_loss_vs_production", True, dev["log_loss"], None, "no production model to compare against")
        gate("data_not_older_than_production", True, detail="no production model")
    else:
        pdev = ensure_dev_evaluation(session, production)
        same = (pdev.get("eval_seasons"), pdev.get("block_games")) == (dev.get("eval_seasons"), dev.get("block_games"))
        limit = pdev["log_loss"] + th["max_dev_log_loss_regression"]
        gate("dev_log_loss_vs_production", same and dev["log_loss"] <= limit, dev["log_loss"], limit,
             f"production {production.model_version} dev log loss {pdev['log_loss']:.4f}"
             + ("" if same else "; folds differ, not comparable"))
        c_new, c_old = data_cutoff(session, entry), data_cutoff(session, production)
        gate("data_not_older_than_production", c_new is not None and (c_old is None or c_new >= c_old),
             str(c_new), str(c_old))

    seasons = set(dev.get("eval_seasons") or [])
    cal = (entry.calibration or {}).get("fit_split")
    cal_ok = cal in ("development_rolling_oos", "validation")
    gate("development_only_provenance", seasons <= set(s.development_seasons) and cal_ok,
         {"dev_eval_seasons": sorted(seasons), "calibration_fit": cal}, {"development_seasons": s.development_seasons})
    return {"passed": all(g["passed"] for g in gates), "gates": gates, "thresholds": th,
            "production_version": production.model_version if production is not None else None,
            "evaluated_at": _now().isoformat(timespec="seconds")}


def _record(session: Session, entry: ModelRegistry, action: str, from_status: str | None, to_status: str | None,
            replaced: str | None, gates: dict | None, source: str | None) -> ModelPromotionEvent:
    ev = ModelPromotionEvent(pitcher_id=entry.pitcher_id, model_version=entry.model_version, action=action,
                             from_status=from_status, to_status=to_status, replaced_version=replaced,
                             gate_results=gates, source=source,
                             created_at=_now())  # wall clock: the prospective ledger compares it with game dates
    session.add(ev)
    lc = lifecycle_row(session, entry.model_version)
    if gates is not None:
        if lc is not None:
            lc.gate_results = gates
        entry.config = {**(entry.config or {}), "gate_results": {"passed": gates["passed"],
                                                                 "evaluated_at": gates["evaluated_at"],
                                                                 "failed": [g["gate"] for g in gates["gates"]
                                                                            if not g["passed"]]}}
    session.flush()
    return ev


def check_gates(session: Session, version: str) -> dict:
    """Evaluate and record gates without changing any status."""
    e = registry.get_entry(session, None, version)
    g = evaluate_gates(session, e)
    _record(session, e, "gates_evaluated", e.status, e.status, g["production_version"], g, "gate check")
    return g


def promote_with_gates(session: Session, version: str, source: str, force: bool = False) -> dict:
    """Promote a candidate to production only if every gate passes (or `force`, recorded as a manual override)."""
    e = registry.get_entry(session, None, version)
    if e.status == "archived":
        raise ValueError("archived models cannot be promoted")
    if e.status == "promoted":
        return {"promoted": True, "model_version": version, "status": e.status, "note": "already production"}
    prod = registry.promoted(session, e.pitcher_id)
    gates = evaluate_gates(session, e, prod)
    before = e.status
    if not gates["passed"] and not force:
        _record(session, e, "gate_failed", before, before, prod.model_version if prod else None, gates, source)
        log(logger, "promotion_gate_failed", version=version, failed=[g["gate"] for g in gates["gates"]
                                                                       if not g["passed"]])
        return {"promoted": False, "model_version": version, "status": e.status, "gates": gates}
    src = source if gates["passed"] else f"{source} (manual override: gates bypassed)"
    registry.promote(session, version, source=src)
    _record(session, e, "promote", before, "promoted", prod.model_version if prod else None, gates, src)
    sync_aliases(session, e.pitcher_id)
    log(logger, "model_promoted", version=version, replaced=prod.model_version if prod else None,
        gates_passed=gates["passed"])
    return {"promoted": True, "model_version": version, "status": "promoted", "gates": gates,
            "previous_production": prod.model_version if prod else None}


def rollback(session: Session, version: str, source: str = "manual rollback") -> dict:
    """Roll production model `version` back: the previous_production model becomes production again and `version`
    returns to candidate (kept, never deleted)."""
    e = registry.get_entry(session, None, version)
    if e.status != "promoted":
        raise ValueError(f"{version} is not the production model (status {e.status}); only production can be "
                         "rolled back")
    prev = session.scalars(select(ModelRegistry).where(ModelRegistry.pitcher_id == e.pitcher_id,
                                                       ModelRegistry.status == "previous_production")
                           .order_by(ModelRegistry.created_at.desc())).first()
    if prev is None:
        raise LookupError("no previous_production model to roll back to")
    stamp = _now().isoformat(timespec="seconds")
    session.execute(update(ModelRegistry).where(ModelRegistry.pitcher_id == e.pitcher_id,
                                                ModelRegistry.status == "previous_production",
                                                ModelRegistry.model_version != prev.model_version)
                    .values(status="candidate"))
    e.status = "candidate"
    e.config = {**(e.config or {}), "rolled_back_at": stamp, "rolled_back_to": prev.model_version}
    prev.status = "promoted"
    prev.config = {**(prev.config or {}), "promotion_source": f"rollback from {version}: {source}",
                   "promoted_at": stamp}
    session.flush()
    _record(session, e, "rollback", "promoted", "candidate", version, None, source)
    _record(session, prev, "rollback", "previous_production", "promoted", version, None, source)
    from . import tracking
    tracking.get_tracker().sync_status([e, prev])  # status tags + `promoted` alias (no-op if disabled)
    sync_aliases(session, e.pitcher_id)
    log(logger, "model_rolled_back", from_version=version, to_version=prev.model_version)
    return {"rolled_back": version, "production": prev.model_version, "status": {version: "candidate",
                                                                                prev.model_version: "promoted"}}


# ------------------------------------------------------------------ MLflow aliases
def sync_aliases(session: Session, pitcher_id: int) -> dict:
    """Point MLflow aliases `production` / `previous_production` at the registered versions of the corresponding
    Postgres rows (the existing `promoted` alias is maintained by tracking.register_promoted). Best effort."""
    from . import tracking
    t = tracking.get_tracker()
    out = {}
    if not t.enabled or not t.register_models:
        return out
    try:
        c, name = t.client, t.experiment_name(pitcher_id)
        for alias, status in ((PRODUCTION_ALIAS, "promoted"), (PREVIOUS_ALIAS, "previous_production")):
            e = session.scalars(select(ModelRegistry).where(ModelRegistry.pitcher_id == pitcher_id,
                                                            ModelRegistry.status == status)
                                .order_by(ModelRegistry.created_at.desc())).first()
            rid = (e.config or {}).get("mlflow_run_id") if e is not None else None
            mv = None
            if rid:  # a previous production model was registered when it was promoted
                found = c.search_model_versions(f"name='{name}' and run_id='{rid}'")
                mv = found[0] if found else None
            if mv is not None:
                c.set_registered_model_alias(name, alias, mv.version)
                out[alias] = str(mv.version)
            else:
                try:
                    c.delete_registered_model_alias(name, alias)
                except Exception:  # noqa: BLE001 - alias or model absent
                    pass
                out[alias] = None
    except Exception as ex:  # noqa: BLE001 - mirror is best effort
        log(logger, "mlflow_mirror_failed", op="sync_aliases", error=f"{type(ex).__name__}: {ex}"[:500])
    return out


# ------------------------------------------------------------------ views
def stage(status: str) -> str:
    return STAGE_BY_STATUS.get(status, status)


def lifecycle_view(entry: ModelRegistry, session: Session | None = None) -> dict:
    session = session or object_session(entry)
    lc = lifecycle_row(session, entry.model_version) if session is not None else None
    d = {"stage": stage(entry.status), "lifecycle_trained": lc is not None}
    if lc is not None:
        d.update(training_scope=lc.training_scope, data_cutoff=str(lc.data_cutoff),
                 deployment_cutoff=str(lc.deployment_cutoff), n_train_samples=lc.n_train_samples,
                 n_train_games=lc.n_train_games, selected_params=lc.selected_params, dev_metrics=lc.dev_metrics,
                 dev_evaluation_run_id=lc.dev_evaluation_run_id, gate_results=lc.gate_results,
                 retrain_policy=lc.retrain_policy)
    elif session is not None:
        cut = data_cutoff(session, entry)
        d.update(training_scope="season_split", data_cutoff=str(cut) if cut else None,
                 gate_results=(entry.config or {}).get("gate_results"))
    return d


def events(session: Session, pitcher_id: int, limit: int = 50) -> list[dict]:
    rows = session.scalars(select(ModelPromotionEvent).where(ModelPromotionEvent.pitcher_id == pitcher_id)
                           .order_by(ModelPromotionEvent.event_id.desc()).limit(limit)).all()
    return [{"event_id": r.event_id, "model_version": r.model_version, "action": r.action,
             "from_status": r.from_status, "to_status": r.to_status, "replaced_version": r.replaced_version,
             "gates_passed": (r.gate_results or {}).get("passed"), "gate_results": r.gate_results,
             "source": r.source, "created_at": str(r.created_at)} for r in rows]


def overview(session: Session, pitcher_id: int) -> dict:
    rows = session.scalars(select(ModelRegistry).where(ModelRegistry.pitcher_id == pitcher_id)
                           .order_by(ModelRegistry.created_at.desc())).all()

    def v(e):
        return {"model_version": e.model_version, "family": e.model_family, "status": e.status,
                "created_at": str(e.created_at), **lifecycle_view(e, session)}
    by = lambda st: [v(e) for e in rows if e.status == st]  # noqa: E731
    return {"production": (by("promoted") or [None])[0], "previous_production": (by("previous_production") or
                                                                                  [None])[0],
            "candidates": by("candidate"), "archived_count": sum(e.status == "archived" for e in rows),
            "events": events(session, pitcher_id), "gate_thresholds": gate_thresholds(),
            "archive_policy": "keep (models are never deleted)"}
