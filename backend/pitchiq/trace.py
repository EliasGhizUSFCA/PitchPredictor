"""Records REAL processing stages (timings, inputs, outputs) for the Engineering Console."""
from __future__ import annotations

import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone

from sqlalchemy.orm import Session

from .models import PipelineTrace


class Trace:
    def __init__(self, kind: str, model_version: str | None = None):
        self.trace_id = str(uuid.uuid4())
        self.kind = kind
        self.model_version = model_version
        self.stages: list[dict] = []
        self._t0 = time.perf_counter()

    @contextmanager
    def stage(self, name: str, inputs: dict | None = None):
        rec = {"stage": name, "status": "running", "started_at": datetime.now(timezone.utc).isoformat(),
               "inputs": inputs or {}, "outputs": {}, "warnings": [], "error": None}
        t = time.perf_counter()
        try:
            yield rec
            rec["status"] = "ok" if not rec["warnings"] else "warning"
        except Exception as e:
            rec["status"] = "error"
            rec["error"] = f"{type(e).__name__}: {e}"
            raise
        finally:
            rec["duration_ms"] = round((time.perf_counter() - t) * 1000, 3)
            rec["model_version"] = self.model_version
            self.stages.append(rec)

    def total_ms(self) -> float:
        return round((time.perf_counter() - self._t0) * 1000, 3)

    def persist(self, session: Session) -> None:
        session.add(PipelineTrace(trace_id=self.trace_id, kind=self.kind, model_version=self.model_version,
                                  stages=self.stages, total_ms=self.total_ms()))
        session.flush()

    def to_dict(self) -> dict:
        return {"trace_id": self.trace_id, "kind": self.kind, "model_version": self.model_version,
                "stages": self.stages, "total_ms": self.total_ms()}
