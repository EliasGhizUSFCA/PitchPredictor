"""Optional MLflow mirror of the experiment record.

Postgres stays the source of truth: `model_registry` and `evaluation_runs` hold every model, metric, fingerprint and
artifact checksum. This module copies those rows into MLflow so experiments can be compared in the MLflow UI. It never
feeds anything back into training, selection, calibration or promotion.

The mirror is best-effort and never breaks the pipeline:

* it is a no-op when `PITCHIQ_MLFLOW_ENABLED=false` or when `mlflow` cannot be imported;
* any MLflow error is logged as a warning (`mlflow_mirror_failed`) and the caller gets `None` back.

Layout: one experiment per pitcher (`<prefix>-<pitcher_id>`), one run per trained model (`pitchiq.run_kind=training`)
and one run per evaluation row (`pitchiq.run_kind=evaluation`). Promoted models are registered in the MLflow model
registry under the experiment name, with the alias `promoted` on the current one.

The MLflow run id is written back into `model_registry.config["mlflow_run_id"]` and
`evaluation_runs.metrics["mlflow_run_id"]` (JSONB, no schema change). `pitchiq mlflow-backfill` mirrors rows that
predate tracking or were written while tracking was off; it is idempotent.
"""
from __future__ import annotations

import contextlib
import functools
import hashlib
import json
import logging
import math
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

from ..config import get_settings
from ..logging_setup import log

logger = logging.getLogger(__name__)

T = "pitchiq."  # tag namespace
PROMOTED_ALIAS = "promoted"
REGISTERED_MODEL_DESCRIPTION = ("PitchIQ promoted next-pitch models. Mirror of Postgres model_registry promotion; "
                                "how each version was promoted is in its `pitchiq.promotion_source` tag.")
BACKFILL_SPLIT_NOTE = "split policy is the current configuration at backfill time"
UNRECORDED_PROMOTION = "unrecorded (promoted before promotion_source was stored)"
_MAX_PARAM_LEN = 6000


def _safe(fn):
    """Mirror failures are logged and swallowed: Postgres is the record, MLflow is a convenience."""

    @functools.wraps(fn)
    def wrapper(self, *a, **kw):
        if not self.enabled:
            return None
        try:
            return fn(self, *a, **kw)
        except Exception as e:  # noqa: BLE001 - deliberately broad, see docstring
            log(logger, "mlflow_mirror_failed", op=fn.__name__, error=f"{type(e).__name__}: {e}"[:500])
            return None

    return wrapper


def _ms(dt: datetime | None) -> int:
    return int(dt.timestamp() * 1000) if dt else int(time.time() * 1000)


def _num(v) -> float | None:
    if isinstance(v, bool) or v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _jsonable(o):
    if hasattr(o, "item"):
        return o.item()
    if isinstance(o, (set, tuple)):
        return list(o)
    if isinstance(o, (datetime, Path)):
        return str(o)
    return repr(o)


def _dumps(obj) -> str:
    return json.dumps(obj, indent=2, sort_keys=True, default=_jsonable)


def _strip_mirror_keys(d: dict | None) -> dict:
    return {k: v for k, v in (d or {}).items() if not str(k).startswith("mlflow_")}


def _db_name(obj) -> str:
    """Name of the Postgres database a row came from (tags runs so test and main DB rows never collide)."""
    from sqlalchemy.orm import object_session
    s = object_session(obj)
    try:
        return str(s.get_bind().url.database) if s is not None else "unknown"
    except Exception:  # noqa: BLE001
        return "unknown"


def _is_complete(run) -> bool:
    """FINISHED and fully logged. Runs mirrored before `mirror_complete` existed were only ever terminated FINISHED
    after their last write, so FINISHED + (for training) the late `artifact_logged` tag also counts."""
    if run.info.status != "FINISHED" or run.data.tags.get(T + "incomplete") == "true":
        return False
    t = run.data.tags
    if t.get(T + "mirror_complete") == "true":
        return True
    return t.get(T + "run_kind") != "training" or T + "artifact_logged" in t


def split_policy() -> str:
    s = get_settings()
    return f"chronological by season: train={s.train_seasons} validation={s.validation_seasons} test={s.test_seasons}"


class Tracker:
    def __init__(self, uri: str | None = None, artifact_root: Path | str | None = None, enabled: bool | None = None,
                 prefix: str | None = None, log_model_artifacts: bool | None = None,
                 register_models: bool | None = None):
        s = get_settings()
        self.uri = uri or s.mlflow_uri
        self.artifact_root = Path(artifact_root) if artifact_root else s.mlflow_artifact_dir
        self.prefix = prefix or s.mlflow_experiment_prefix
        self.log_model_artifacts = s.mlflow_log_model_artifacts if log_model_artifacts is None else log_model_artifacts
        self.register_models = s.mlflow_register_models if register_models is None else register_models
        on = s.mlflow_enabled if enabled is None else enabled
        self.disabled_reason: str | None = None if on else "disabled (PITCHIQ_MLFLOW_ENABLED=false)"
        self._client = None
        self._exp: dict[int, str] = {}

    # ------------------------------------------------------------------ plumbing
    @property
    def client(self):
        if self._client is None and self.disabled_reason is None:
            try:
                os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
                from mlflow.tracking import MlflowClient
            except Exception as e:  # noqa: BLE001 - ImportError or a broken install
                self.disabled_reason = f"mlflow not importable ({type(e).__name__}: {e})"
                log(logger, "mlflow_disabled", reason=self.disabled_reason)
                return None
            if self.uri.startswith("sqlite:///"):
                Path(self.uri[len("sqlite:///"):]).parent.mkdir(parents=True, exist_ok=True)
            try:
                self._client = MlflowClient(tracking_uri=self.uri, registry_uri=self.uri)
            except Exception as e:  # noqa: BLE001 - transient (e.g. locked store): retry on next use
                log(logger, "mlflow_mirror_failed", op="connect", error=f"{type(e).__name__}: {e}"[:500])
                return None
        return self._client

    @property
    def enabled(self) -> bool:
        return self.client is not None

    def status(self) -> dict:
        return {"enabled": self.enabled, "disabled_reason": self.disabled_reason, "tracking_uri": self.uri,
                "experiment_prefix": self.prefix, "source_of_truth": "postgres",
                "ui_command": f"mlflow ui --backend-store-uri {self.uri}"}

    def experiment_name(self, pitcher_id: int) -> str:
        return f"{self.prefix}-{pitcher_id}"

    def experiment_id(self, pitcher_id: int) -> str:
        if pitcher_id in self._exp:
            return self._exp[pitcher_id]
        c, name = self.client, self.experiment_name(pitcher_id)
        exp = c.get_experiment_by_name(name)
        if exp is None:
            loc = (self.artifact_root / name).resolve().as_uri() if self._local_store() else None
            try:
                eid = c.create_experiment(name, artifact_location=loc,
                                          tags={T + "pitcher_id": str(pitcher_id), T + "source_of_truth": "postgres"})
            except Exception:  # noqa: BLE001 - created concurrently by another process
                exp = c.get_experiment_by_name(name)
                if exp is None:
                    raise
                eid = exp.experiment_id
        else:
            eid = exp.experiment_id
        self._exp[pitcher_id] = eid
        return eid

    def _local_store(self) -> bool:
        return self.uri.startswith(("sqlite:", "file:", "/")) or "://" not in self.uri

    def get_run(self, run_id: str | None):
        if not run_id or not self.enabled:
            return None
        try:
            r = self.client.get_run(run_id)
        except Exception:  # noqa: BLE001 - RESOURCE_DOES_NOT_EXIST (e.g. a different store)
            return None
        return r if r.info.lifecycle_stage == "active" else None

    def _search(self, eid: str, flt: str) -> list:
        out, token = [], None
        while True:
            page = self.client.search_runs([eid], flt, max_results=1000, page_token=token)
            out.extend(page)
            token = page.token
            if not token:
                return out

    def _log_batch(self, run_id: str, params: dict, metrics: Iterable[tuple[str, float, int]], ts: int):
        from mlflow.entities import Metric, Param
        ps = [Param(k, str(v)[:_MAX_PARAM_LEN]) for k, v in params.items() if v is not None]
        ms = [Metric(k, v, ts, step) for k, v, step in metrics]
        for i in range(0, max(len(ps), 1), 100):
            if ps[i:i + 100]:
                self.client.log_batch(run_id, params=ps[i:i + 100])
        for i in range(0, len(ms), 1000):
            self.client.log_batch(run_id, metrics=ms[i:i + 1000])

    # ------------------------------------------------------------------ training runs
    @_safe
    def log_model_entry(self, entry, started_at: datetime | None = None, backfilled: bool = False) -> str | None:
        """Mirror one `model_registry` row as an MLflow run. Returns the run id."""
        c = self.client
        eid = self.experiment_id(entry.pitcher_id)
        cfg = entry.config or {}
        start = _ms(started_at or entry.created_at)
        tags = {T + "run_kind": "training", T + "model_version": entry.model_version, T + "family": entry.model_family,
                T + "feature_version": entry.feature_version, T + "status": entry.status,
                T + "pitcher_id": str(entry.pitcher_id), T + "database": _db_name(entry),
                T + "artifact_sha256": entry.artifact_sha256, T + "artifact_path": entry.artifact_path,
                T + "selection": str(cfg.get("selection", "")), T + "split_policy": split_policy(),
                T + "source_of_truth": "postgres:model_registry", T + "backfilled": str(backfilled).lower(),
                "mlflow.source.name": "pitchiq mlflow-backfill" if backfilled else "pitchiq.ml.train.train_family"}
        if cfg.get("ablation"):
            tags[T + "ablation"] = str(cfg["ablation"])
        if backfilled:
            tags[T + "split_policy_note"] = BACKFILL_SPLIT_NOTE
        if entry.status == "promoted":
            tags[T + "promotion_source"] = cfg.get("promotion_source") or UNRECORDED_PROMOTION
        run = c.create_run(eid, start_time=start, tags=tags, run_name=entry.model_version)
        rid = run.info.run_id
        with self._finish_or_fail(rid, start if backfilled else None):
            self._log_training_body(rid, entry, start)
        if entry.status == "promoted":
            self.register_promoted(entry, rid)
        return rid

    def _log_training_body(self, rid: str, entry, start: int) -> None:
        c = self.client
        cfg = entry.config or {}
        fp = entry.dataset_fingerprint or {}
        cal = entry.calibration or {}
        vm = entry.validation_metrics or {}
        used = (entry.feature_schema or {}).get("used_features") or cfg.get("features") or []
        params = {"family": entry.model_family, "feature_version": entry.feature_version, "n_features": len(used),
                  "classes": ",".join(entry.classes or []), "n_classes": len(entry.classes or []),
                  "selection": cfg.get("selection"), "grid_size": len(cfg.get("grid_results") or []) or None,
                  "calibration_method": cal.get("method"), "calibration_fit_split": cal.get("fit_split")}
        params.update({f"hp.{k}": v for k, v in (cfg.get("params") or {}).items()})
        for split, d in fp.items():
            if isinstance(d, dict):
                params[f"data.{split}.n"] = d.get("n")
                params[f"data.{split}.pitch_uid_hash"] = d.get("pitch_uid_hash")
        metrics: list[tuple[str, float, int]] = []
        for k, v in vm.items():
            if _num(v) is not None:
                metrics.append((f"validation_{k}", _num(v), 0))
        ci = vm.get("top1_ci95") or []
        if len(ci) == 2 and _num(ci[0]) is not None:
            metrics += [("validation_top1_ci95_lo", _num(ci[0]), 0), ("validation_top1_ci95_hi", _num(ci[1]), 0)]
        for k in ("temperature", "val_log_loss_before", "val_log_loss_after"):
            if _num(cal.get(k)) is not None:
                metrics.append((f"calibration_{k}", _num(cal[k]), 0))
        if _num(cfg.get("best_iteration")) is not None:
            metrics.append(("best_iteration", _num(cfg["best_iteration"]), 0))
        for i, g in enumerate(cfg.get("grid_results") or []):
            for k in ("val_log_loss", "val_top1"):
                if _num(g.get(k)) is not None:
                    metrics.append((f"grid_{k}", _num(g[k]), i))
        self._log_batch(rid, params, metrics, start)

        c.log_text(rid, _dumps(cfg.get("grid_results") or []), "grid_results.json")
        c.log_text(rid, _dumps(used), "features.json")
        c.log_text(rid, _dumps(entry.feature_schema), "feature_schema.json")
        c.log_text(rid, _dumps(fp), "dataset_fingerprint.json")
        c.log_text(rid, _dumps(_strip_mirror_keys(cfg)), "config.json")
        c.log_text(rid, _dumps(cal), "calibration.json")
        c.log_text(rid, _dumps(vm), "validation_metrics.json")
        self._log_datasets(rid, fp)
        logged, why = self._log_model_file(rid, entry)
        c.set_tag(rid, T + "artifact_logged", str(logged).lower())
        if why:
            c.set_tag(rid, T + "artifact_note", why)

    @contextlib.contextmanager
    def _finish_or_fail(self, rid: str, end_ms: int | None):
        """Terminate a run FINISHED only if everything was logged; otherwise FAILED + `pitchiq.incomplete=true`.

        Backfill re-links runs by tag only when they are FINISHED, so a half-written run is never adopted."""
        try:
            yield
        except BaseException as e:
            try:
                self.client.set_tag(rid, T + "incomplete", "true")
                self.client.set_tag(rid, T + "incomplete_reason", f"{type(e).__name__}: {e}"[:500])
                self.client.set_terminated(rid, "FAILED")
            except Exception as e2:  # noqa: BLE001 - the store itself may be what failed
                log(logger, "mlflow_mark_failed_failed", run_id=rid, error=f"{type(e2).__name__}: {e2}"[:300])
            raise
        self.client.set_tag(rid, T + "mirror_complete", "true")
        self.client.set_terminated(rid, "FINISHED", end_time=end_ms if end_ms is not None else _ms(None))

    def _log_model_file(self, rid: str, entry) -> tuple[bool, str | None]:
        if not self.log_model_artifacts:
            return False, "model artifact copy disabled (PITCHIQ_MLFLOW_LOG_MODEL_ARTIFACTS=false)"
        p = Path(entry.artifact_path)
        if not p.exists():
            return False, "artifact file missing on this machine"
        if hashlib.sha256(p.read_bytes()).hexdigest() != entry.artifact_sha256:
            return False, "artifact checksum mismatch; not copied"
        self.client.log_artifact(rid, str(p), "model")
        return True, None

    def _log_datasets(self, rid: str, fp: dict) -> None:
        try:
            from mlflow.entities import Dataset, DatasetInput, InputTag
        except Exception:  # noqa: BLE001 - older mlflow
            return
        inputs = []
        for split, d in fp.items():
            if not isinstance(d, dict) or not d.get("pitch_uid_hash"):
                continue
            ds = Dataset(name=f"{split}", digest=str(d["pitch_uid_hash"]), source_type="postgres",
                         source=_dumps({"table": "pre_pitch_states", "split": split}),
                         profile=_dumps({"num_rows": d.get("n"), "label_counts": d.get("label_counts")}))
            inputs.append(DatasetInput(ds, tags=[InputTag("mlflow.data.context", split)]))
        if inputs:
            self.client.log_inputs(rid, datasets=inputs)

    # ------------------------------------------------------------------ evaluation runs
    @_safe
    def log_evaluation(self, run, entry=None, backfilled: bool = False) -> str | None:
        """Mirror one `evaluation_runs` row. `entry` is its model_registry row (for family / model run link)."""
        c = self.client
        eid = self.experiment_id(run.pitcher_id)
        split = run.split
        fam = entry.model_family if entry is not None else "unknown"
        start = _ms(run.created_at)
        tags = {T + "run_kind": "evaluation", T + "evaluation_kind": run.kind,
                T + "evaluation_run_id": str(run.evaluation_run_id), T + "model_version": run.model_version,
                T + "family": fam, T + "split": split, T + "pitcher_id": str(run.pitcher_id),
                T + "database": _db_name(run), T + "split_policy": split_policy(),
                T + "source_of_truth": "postgres:evaluation_runs", T + "backfilled": str(backfilled).lower(),
                "mlflow.source.name": "pitchiq mlflow-backfill" if backfilled else f"pitchiq.ml.evaluate.{run.kind}"}
        if backfilled:
            tags[T + "split_policy_note"] = BACKFILL_SPLIT_NOTE
        if entry is not None:
            tags[T + "feature_version"] = entry.feature_version
            # `model_status` is kept current by sync_status / backfill; `model_status_at_mirror` never changes.
            tags[T + "model_status"] = entry.status
            tags[T + "model_status_at_mirror"] = entry.status
            if (entry.config or {}).get("mlflow_run_id"):
                tags[T + "model_run_id"] = entry.config["mlflow_run_id"]
        r = c.create_run(eid, start_time=start, tags=tags, run_name=f"{run.kind}-{run.evaluation_run_id}-{fam}")
        rid = r.info.run_id
        with self._finish_or_fail(rid, start if backfilled else None):
            self._log_evaluation_body(rid, run, start)
        return rid

    def _log_evaluation_body(self, rid: str, run, start: int) -> None:
        c = self.client
        m = run.metrics or {}
        split = run.split
        params = {"kind": run.kind, "split": split, "n_samples": run.n_samples}
        params.update({f"dataset.{k}": (v if not isinstance(v, (dict, list)) else _dumps(v))
                       for k, v in (run.dataset_definition or {}).items()})
        if "seeds" in m:
            params["seeds"] = m["seeds"]
        metrics: list[tuple[str, float, int]] = []

        def add(name, v, step=0):
            if _num(v) is not None:
                metrics.append((name, _num(v), step))

        for k, v in (m.get("summary") or {}).items():
            add(f"{split}_{k}", v)
        ci = (m.get("summary") or {}).get("top1_ci95") or []
        if len(ci) == 2:
            add(f"{split}_top1_wilson_ci95_lo", ci[0]); add(f"{split}_top1_wilson_ci95_hi", ci[1])
        b = m.get("bootstrap") or {}
        for k in ("top1_accuracy_ci95", "log_loss_ci95"):
            if len(b.get(k) or []) == 2:
                base = k.replace("_ci95", "")
                add(f"{split}_{base}_boot_ci95_lo", b[k][0]); add(f"{split}_{base}_boot_ci95_hi", b[k][1])
        add(f"{split}_ece", (m.get("calibration") or {}).get("ece_top_label"))
        for k, v in (m.get("uncalibrated_summary") or {}).items():
            add(f"{split}_uncalibrated_{k}", v)
        add("inference_ms_per_pitch", m.get("inference_ms_per_pitch"))
        add("walk_forward_blocks", m.get("blocks"))
        for res in m.get("results") or []:  # ablation
            cfgname = res.get("config", "?")
            for k in ("val_log_loss", "val_log_loss_sd", "val_top1", "delta_log_loss", "delta_top1", "n_features"):
                add(f"ablation/{cfgname}/{k}", res.get(k))
        self._log_batch(rid, params, metrics, start)

        for k, v in m.items():
            if isinstance(v, (dict, list)) and not str(k).startswith("mlflow_"):
                c.log_text(rid, _dumps(v), f"{k}.json")
        c.log_text(rid, _dumps(run.dataset_definition), "dataset_definition.json")
        notes = {k: v for k, v in m.items() if isinstance(v, str) and not k.startswith("mlflow_")}
        if notes:
            c.log_text(rid, _dumps(notes), "notes.json")

    # ------------------------------------------------------------------ model registry
    @_safe
    def register_promoted(self, entry, run_id: str | None = None) -> str | None:
        """Register a promoted model in the MLflow model registry and point the `promoted` alias at it.

        If the promoted entry cannot be registered (no complete run, or its model file was not logged), the alias is
        removed instead, so it never names a model Postgres no longer promotes."""
        if not self.register_models:
            return None
        c = self.client
        name = self.experiment_name(entry.pitcher_id)
        rid = run_id or (entry.config or {}).get("mlflow_run_id")
        run = self.get_run(rid)
        source = (entry.config or {}).get("promotion_source") or UNRECORDED_PROMOTION
        why = None
        if run is None:
            why = "no mirrored run for the promoted model in this store"
        elif not _is_complete(run):
            why = "mirrored run is incomplete"
        elif run.data.tags.get(T + "artifact_logged") != "true":
            why = "model file not logged: " + run.data.tags.get(T + "artifact_note", "unknown reason")
        if why:
            self._drop_alias(name, keep_run_id=None)
            if run is not None:
                c.set_tag(rid, T + "registry_note", f"not registered ({why}); 'promoted' alias removed")
            log(logger, "mlflow_promoted_not_registered", model_version=entry.model_version, reason=why)
            return None
        try:
            rm = c.get_registered_model(name)
        except Exception:  # noqa: BLE001
            rm = c.create_registered_model(name, tags={T + "pitcher_id": str(entry.pitcher_id)},
                                           description=REGISTERED_MODEL_DESCRIPTION)
        if rm.description != REGISTERED_MODEL_DESCRIPTION:
            c.update_registered_model(name, description=REGISTERED_MODEL_DESCRIPTION)
        existing = c.search_model_versions(f"name='{name}' and run_id='{rid}'")
        if existing:
            mv = existing[0]
        else:
            mv = c.create_model_version(name, source=f"{run.info.artifact_uri}/model", run_id=rid,
                                        tags={T + "model_version": entry.model_version,
                                              T + "family": entry.model_family,
                                              T + "artifact_sha256": entry.artifact_sha256},
                                        description=f"{entry.model_version} (joblib artifact, pitchiq loader)")
        c.set_model_version_tag(name, mv.version, T + "promotion_source", source)
        c.set_tag(rid, T + "promotion_source", source)
        c.set_registered_model_alias(name, PROMOTED_ALIAS, mv.version)
        return str(mv.version)

    def _drop_alias(self, name: str, keep_run_id: str | None) -> None:
        c = self.client
        try:
            mv = c.get_model_version_by_alias(name, PROMOTED_ALIAS)
        except Exception:  # noqa: BLE001 - no registered model or no alias yet
            return
        if mv.run_id != keep_run_id:
            c.delete_registered_model_alias(name, PROMOTED_ALIAS)

    def _refresh_training_tags(self, rid: str, e) -> None:
        self.client.set_tag(rid, T + "status", e.status)
        if e.status == "promoted":
            self.client.set_tag(rid, T + "promotion_source",
                                (e.config or {}).get("promotion_source") or UNRECORDED_PROMOTION)

    def _refresh_evaluation_tags(self, run, e) -> None:
        """Keep `model_status` current; add tags that older mirrors lacked (never alters a mirrored metric)."""
        t, rid = run.data.tags, run.info.run_id
        if t.get(T + "backfilled") == "true" and t.get(T + "split_policy_note") != BACKFILL_SPLIT_NOTE:
            self.client.set_tag(rid, T + "split_policy_note", BACKFILL_SPLIT_NOTE)
        if T + "model_status_at_mirror" not in t and T + "model_status" in t:
            self.client.set_tag(rid, T + "model_status_at_mirror", t[T + "model_status"])
        if e is not None and t.get(T + "model_status") != e.status:
            self.client.set_tag(rid, T + "model_status", e.status)

    @_safe
    def sync_status(self, entries: Iterable) -> int:
        """Refresh status tags on mirrored training runs and on the evaluation runs of those models, then move the
        registry alias for a promoted entry."""
        n = 0
        for e in entries:
            rid = (e.config or {}).get("mlflow_run_id")
            if self.get_run(rid) is not None:
                self._refresh_training_tags(rid, e)
                n += 1
            for r in self._search(self.experiment_id(e.pitcher_id),
                                  f"tags.`{T}run_kind` = 'evaluation' and tags.`{T}model_version` = "
                                  f"'{e.model_version}' and tags.`{T}database` = '{_db_name(e)}'"):
                self._refresh_evaluation_tags(r, e)
            if e.status == "promoted":
                self.register_promoted(e, rid)
        return n

    # ------------------------------------------------------------------ backfill
    def backfill(self, session, pitcher_id: int | None = None, orphan_grace_s: float = 3600.0) -> dict:
        """Mirror every model_registry row (archived included) and every evaluation_runs row into MLflow.

        Idempotent: rows whose stored run id exists in this store are only refreshed (status tags). Rows whose run
        was created but whose id never reached Postgres are found by tag and re-linked, but only if that run is
        complete (FINISHED, `pitchiq.mirror_complete`); incomplete runs are tagged orphaned and the row is re-mirrored.
        """
        from sqlalchemy import select

        from ..models import EvaluationRun, ModelRegistry
        if not self.enabled:
            return {"enabled": False, "disabled_reason": self.disabled_reason}
        q = select(ModelRegistry).order_by(ModelRegistry.created_at, ModelRegistry.model_version)
        if pitcher_id is not None:
            q = q.where(ModelRegistry.pitcher_id == pitcher_id)
        entries = session.scalars(q).all()
        by_version = {e.model_version: e for e in entries}
        rep = {"tracking_uri": self.uri, "models": {"total": len(entries), "created": 0, "relinked": 0,
                                                     "already_mirrored": 0, "failed": 0, "by_status": {}},
               "evaluations": {"total": 0, "created": 0, "relinked": 0, "already_mirrored": 0, "failed": 0,
                               "by_kind": {}},
               "registered_promoted": [], "incomplete_runs_tagged": 0, "orphaned_runs_tagged": 0,
               "experiments": []}
        for e in entries:
            rep["models"]["by_status"][e.status] = rep["models"]["by_status"].get(e.status, 0) + 1
            rid = (e.config or {}).get("mlflow_run_id")
            if self.get_run(rid) is not None:
                self._refresh_training_tags(rid, e)
                rep["models"]["already_mirrored"] += 1
                continue
            found = self._find_complete(e.pitcher_id, {"run_kind": "training", "model_version": e.model_version,
                                                       "database": _db_name(e)}, rep)
            if found is not None:
                rid = found.info.run_id
                self._refresh_training_tags(rid, e)
                self.client.set_tag(rid, T + "orphaned", "false")
                rep["models"]["relinked"] += 1
            else:
                rid = self.log_model_entry(e, backfilled=True)
                if rid is None:
                    rep["models"]["failed"] += 1
                    continue
                rep["models"]["created"] += 1
            e.config = {**(e.config or {}), "mlflow_run_id": rid}
        session.flush()

        q = select(EvaluationRun).order_by(EvaluationRun.evaluation_run_id)
        if pitcher_id is not None:
            q = q.where(EvaluationRun.pitcher_id == pitcher_id)
        runs = session.scalars(q).all()
        rep["evaluations"]["total"] = len(runs)
        for r in runs:
            rep["evaluations"]["by_kind"][r.kind] = rep["evaluations"]["by_kind"].get(r.kind, 0) + 1
            entry = by_version.get(r.model_version) or session.get(ModelRegistry, r.model_version)
            rid = (r.metrics or {}).get("mlflow_run_id")
            existing = self.get_run(rid)
            if existing is not None:
                self._refresh_evaluation_tags(existing, entry)
                rep["evaluations"]["already_mirrored"] += 1
                continue
            found = self._find_complete(r.pitcher_id, {"run_kind": "evaluation",
                                                       "evaluation_run_id": str(r.evaluation_run_id),
                                                       "database": _db_name(r)}, rep)
            if found is not None:
                rid = found.info.run_id
                self._refresh_evaluation_tags(found, entry)
                self.client.set_tag(rid, T + "orphaned", "false")
                rep["evaluations"]["relinked"] += 1
            else:
                rid = self.log_evaluation(r, entry, backfilled=True)
                if rid is None:
                    rep["evaluations"]["failed"] += 1
                    continue
                rep["evaluations"]["created"] += 1
            r.metrics = {**(r.metrics or {}), "mlflow_run_id": rid}
        session.flush()

        for e in entries:
            if e.status == "promoted":
                v = self.register_promoted(e)
                if v:
                    rep["registered_promoted"].append({"model_version": e.model_version,
                                                       "registered_model": self.experiment_name(e.pitcher_id),
                                                       "version": v, "alias": PROMOTED_ALIAS})
        rep["orphaned_runs_tagged"] = self._tag_orphans(session, {e.pitcher_id for e in entries} |
                                                        {r.pitcher_id for r in runs}, orphan_grace_s)
        rep["experiments"] = sorted({self.experiment_name(p) for p in {e.pitcher_id for e in entries} |
                                     {r.pitcher_id for r in runs}})
        return rep

    def _find_complete(self, pitcher_id: int, tags: dict, rep: dict):
        """A complete, unorphaned run carrying `tags`. Incomplete matches are tagged orphaned (never adopted).

        The Postgres row is committed when backfill sees it, and live mirroring happens before commit, so a matching
        run that is not FINISHED is a failed mirror rather than one still being written."""
        flt = " and ".join(f"tags.`{T}{k}` = '{v}'" for k, v in tags.items())
        hit = None
        for run in self._search(self.experiment_id(pitcher_id), flt):
            if run.data.tags.get(T + "orphaned") == "true":
                continue
            if not _is_complete(run):
                self.client.set_tag(run.info.run_id, T + "orphaned", "true")
                self.client.set_tag(run.info.run_id, T + "orphan_reason",
                                    f"incomplete mirror (status {run.info.status}); row re-mirrored")
                rep["incomplete_runs_tagged"] += 1
            elif hit is None:
                hit = run
        return hit

    def _tag_orphans(self, session, pitcher_ids: set[int], grace_s: float) -> int:
        """Runs mirrored from this database whose Postgres row no longer exists (e.g. the transaction rolled back).

        Runs that ended less than `grace_s` ago are skipped: their Postgres transaction may still be open."""
        from sqlalchemy import select

        from ..models import EvaluationRun, ModelRegistry
        db = str(session.get_bind().url.database)
        versions = set(session.scalars(select(ModelRegistry.model_version)).all())
        evals = {str(i) for i in session.scalars(select(EvaluationRun.evaluation_run_id)).all()}
        linked = {(e.config or {}).get("mlflow_run_id") for e in session.scalars(select(ModelRegistry)).all()}
        linked |= {(r.metrics or {}).get("mlflow_run_id") for r in session.scalars(select(EvaluationRun)).all()}
        n = 0
        for p in pitcher_ids:
            for run in self._search(self.experiment_id(p), f"tags.`{T}database` = '{db}'"):
                t = run.data.tags
                if run.info.run_id in linked or t.get(T + "orphaned") == "true":
                    continue
                if (run.info.end_time or run.info.start_time or 0) > (time.time() - grace_s) * 1000:
                    continue
                gone = (t.get(T + "run_kind") == "training" and t.get(T + "model_version") not in versions) or \
                       (t.get(T + "run_kind") == "evaluation" and t.get(T + "evaluation_run_id") not in evals)
                # Either the row is gone, or it exists but links a different run (a duplicate mirror).
                self.client.set_tag(run.info.run_id, T + "orphaned", "true")
                self.client.set_tag(run.info.run_id, T + "orphan_reason",
                                    "postgres row missing" if gone else "duplicate mirror; row links another run")
                n += 1
        return n


@functools.lru_cache(maxsize=4)
def _tracker_for(key: tuple) -> Tracker:
    return Tracker()


def get_tracker() -> Tracker:
    """Process-wide tracker for the current settings."""
    s = get_settings()
    return _tracker_for((s.mlflow_enabled, s.mlflow_uri, str(s.mlflow_artifact_dir), s.mlflow_experiment_prefix,
                         s.mlflow_log_model_artifacts, s.mlflow_register_models))


def mirror_model_entry(entry, started_at: datetime | None = None) -> str | None:
    """Called by train_family after the row is flushed. Writes the run id back into config."""
    rid = get_tracker().log_model_entry(entry, started_at=started_at)
    if rid:
        entry.config = {**(entry.config or {}), "mlflow_run_id": rid}
    return rid


def mirror_evaluation(run, entry=None) -> str | None:
    """Called by the benchmark / walk-forward / ablation code after the row is flushed."""
    rid = get_tracker().log_evaluation(run, entry)
    if rid:
        run.metrics = {**(run.metrics or {}), "mlflow_run_id": rid}
    return rid
