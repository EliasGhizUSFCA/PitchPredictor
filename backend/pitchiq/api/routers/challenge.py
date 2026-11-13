from __future__ import annotations

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from ... import challenge as ch
from ...db import get_db
from ...models import PredictionChallenge

router = APIRouter(tags=["challenge"])


class ChallengeIn(BaseModel):
    size: int = Field(25, ge=1, le=1000)
    strategy: str = "natural"
    params: dict | None = None
    seed: int | None = None
    model_version: str | None = None


@router.post("/pitchers/{pid}/challenges")
def create(pid: int, body: ChallengeIn, db: Session = Depends(get_db)):
    c = ch.create_challenge(db, pid, body.size, body.strategy, body.params, body.seed, body.model_version)
    return {"challenge_id": c.challenge_id, "size": c.size, "seed": c.seed, "strategy": c.sampling_strategy,
            "model_version": c.model_version, "split": c.split}


@router.get("/pitchers/{pid}/challenges")
def list_(pid: int, db: Session = Depends(get_db)):
    return [{"challenge_id": c.challenge_id, "size": c.size, "seed": c.seed, "strategy": c.sampling_strategy,
             "model_version": c.model_version, "status": c.status, "created_at": str(c.created_at)}
            for c in db.scalars(select(PredictionChallenge).where(PredictionChallenge.pitcher_id == pid)
                                .order_by(PredictionChallenge.created_at.desc()).limit(50))]


@router.post("/challenges/{cid}/next")
def next_(cid: int, db: Session = Depends(get_db)):
    return ch.next_attempt(db, cid)


class GuessIn(BaseModel):
    guess: str


@router.post("/attempts/{aid}/guess")
def guess(aid: int, body: GuessIn, db: Session = Depends(get_db)):
    ch.set_guess(db, aid, body.guess)
    return {"ok": True}


@router.post("/attempts/{aid}/reveal")
def reveal(aid: int, db: Session = Depends(get_db)):
    return ch.reveal(db, aid)


@router.get("/challenges/{cid}/stats")
def stats(cid: int, db: Session = Depends(get_db)):
    return ch.challenge_stats(db, cid)
