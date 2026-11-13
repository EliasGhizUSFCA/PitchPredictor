"""Postgres-backed job queue: durable, claimable with FOR UPDATE SKIP LOCKED, cancellable, retryable."""
from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.orm import Session

from ..models import Job


def enqueue(session: Session, kind: str, params: dict | None = None, max_attempts: int = 2) -> Job:
    from .tasks import TASKS
    if kind not in TASKS:
        raise ValueError(f"Unknown job kind {kind}. Known: {sorted(TASKS)}")
    j = Job(kind=kind, params=params or {}, status="queued", max_attempts=max_attempts, log=[], progress=0.0,
            records_processed=0, attempts=0, cancel_requested=False)
    session.add(j)
    session.flush()
    return j


def claim(session: Session) -> Job | None:
    j = session.scalars(select(Job).where(Job.status == "queued").order_by(Job.created_at)
                        .with_for_update(skip_locked=True).limit(1)).first()
    if j is None:
        return None
    j.status = "running"
    j.attempts += 1
    j.started_at = datetime.now(timezone.utc)
    session.flush()
    return j


def cancel(session: Session, job_id: int) -> Job:
    j = session.get(Job, job_id)
    if j.status == "queued":
        j.status = "cancelled"
        j.finished_at = datetime.now(timezone.utc)
    elif j.status == "running":
        j.cancel_requested = True
    return j


def retry(session: Session, job_id: int) -> Job:
    old = session.get(Job, job_id)
    if old.status not in ("failed", "cancelled"):
        raise ValueError("Only failed or cancelled jobs can be retried")
    return enqueue(session, old.kind, old.params, old.max_attempts)
