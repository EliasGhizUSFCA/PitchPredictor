from __future__ import annotations

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.orm import Session

from ...db import get_db
from ...jobs import queue
from ...jobs.tasks import DESCRIPTIONS
from ...models import Job

router = APIRouter(tags=["jobs"])


def _view(j: Job, full=False) -> dict:
    d = {"job_id": j.job_id, "kind": j.kind, "description": DESCRIPTIONS.get(j.kind), "params": j.params,
         "status": j.status, "progress": j.progress, "records_processed": j.records_processed, "message": j.message,
         "attempts": j.attempts, "cancel_requested": j.cancel_requested, "created_at": str(j.created_at),
         "started_at": str(j.started_at) if j.started_at else None,
         "finished_at": str(j.finished_at) if j.finished_at else None,
         "error": (j.error or "")[:500] or None}
    if full:
        d.update(result=j.result, log=j.log, error=j.error)
    return d


@router.get("/jobs/kinds")
def kinds():
    return DESCRIPTIONS


@router.get("/jobs")
def list_jobs(limit: int = 50, db: Session = Depends(get_db)):
    return [_view(j) for j in db.scalars(select(Job).order_by(Job.created_at.desc()).limit(limit))]


class JobIn(BaseModel):
    kind: str
    params: dict = {}


@router.post("/jobs")
def create(body: JobIn, db: Session = Depends(get_db)):
    return _view(queue.enqueue(db, body.kind, body.params))


@router.get("/jobs/{jid}")
def get(jid: int, db: Session = Depends(get_db)):
    j = db.get(Job, jid)
    if j is None:
        raise LookupError("job not found")
    return _view(j, full=True)


@router.post("/jobs/{jid}/cancel")
def cancel(jid: int, db: Session = Depends(get_db)):
    return _view(queue.cancel(db, jid))


@router.post("/jobs/{jid}/retry")
def retry(jid: int, db: Session = Depends(get_db)):
    return _view(queue.retry(db, jid))
