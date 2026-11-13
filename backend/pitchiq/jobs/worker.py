"""Worker loop: `python -m pitchiq.jobs.worker` (run one or more processes)."""
from __future__ import annotations

import logging
import time
import traceback
from datetime import datetime, timezone

from ..db import get_sessionmaker
from ..logging_setup import configure_logging, log
from ..models import Job
from .queue import claim
from .tasks import TASKS

logger = logging.getLogger(__name__)


class Cancelled(Exception):
    pass


class JobContext:
    def __init__(self, job_id: int, lo: float = 0.0, hi: float = 1.0):
        self.job_id, self.lo, self.hi = job_id, lo, hi
        self._sm = get_sessionmaker()

    def sub(self, lo, hi):
        return JobContext(self.job_id, self.lo + (self.hi - self.lo) * lo, self.lo + (self.hi - self.lo) * hi)

    def progress(self, frac: float, message: str = "", records: int | None = None):
        # Separate short transaction so progress is visible while the job's own transaction is open.
        with self._sm() as s:
            j = s.get(Job, self.job_id)
            j.progress = round(self.lo + (self.hi - self.lo) * max(0.0, min(1.0, frac)), 4)
            if message:
                j.message = message
                j.log = (j.log or [])[-199:] + [{"ts": datetime.now(timezone.utc).isoformat(), "msg": message}]
            if records is not None:
                j.records_processed = records
            s.commit()
            if j.cancel_requested:
                raise Cancelled()

    def note(self, message: str):
        self.progress(0.0 if self.lo == 0 else 0.0, message)

    def cancelled(self) -> bool:
        with self._sm() as s:
            return bool(s.get(Job, self.job_id).cancel_requested)


def run_one() -> bool:
    sm = get_sessionmaker()
    with sm() as s:
        job = claim(s)
        s.commit()
        if job is None:
            return False
        job_id, kind, params = job.job_id, job.kind, dict(job.params or {})
    log(logger, "job_start", job_id=job_id, kind=kind)
    ctx = JobContext(job_id)
    with sm() as s:
        try:
            result = TASKS[kind](s, params, ctx)
            s.commit()
            status, error = "completed", None
        except Cancelled:
            s.rollback()
            status, error, result = "cancelled", None, None
        except Exception as e:  # noqa: BLE001
            s.rollback()
            status, error, result = "failed", f"{type(e).__name__}: {e}\n{traceback.format_exc()[-3000:]}", None
    with sm() as s:
        j = s.get(Job, job_id)
        if status == "failed" and j.attempts < j.max_attempts:
            j.status = "queued"  # safe retry: every task is idempotent (upserts / rebuilds)
            j.error = error
        else:
            j.status = status
            j.error = error
            j.result = _jsonable(result)
            j.finished_at = datetime.now(timezone.utc)
            if status == "completed":
                j.progress = 1.0
        s.commit()
    log(logger, "job_end", job_id=job_id, kind=kind, status=status)
    return True


def _jsonable(x):
    import json
    return json.loads(json.dumps(x, default=str)) if x is not None else None


def main(poll_s: float = 2.0):
    configure_logging()
    log(logger, "worker_started")
    while True:
        if not run_one():
            time.sleep(poll_s)


if __name__ == "__main__":
    main()
