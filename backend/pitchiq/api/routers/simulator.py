from __future__ import annotations

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.orm import Session

from ...db import get_db
from ...ml.inference import SimState, predict_scenario, sweep
from ...models import SavedScenario

router = APIRouter(tags=["simulator"])


@router.post("/pitchers/{pid}/simulate")
def simulate(pid: int, state: SimState, model_version: str | None = None, db: Session = Depends(get_db)):
    return predict_scenario(db, pid, state, model_version)


class CompareReq(BaseModel):
    a: SimState
    b: SimState
    model_version: str | None = None


@router.post("/pitchers/{pid}/simulate/compare")
def compare(pid: int, req: CompareReq, db: Session = Depends(get_db)):
    ra = predict_scenario(db, pid, req.a, req.model_version)
    rb = predict_scenario(db, pid, req.b, req.model_version)
    pa = {d["pitch"]: d["probability"] for d in ra["predictions"]}
    pb = {d["pitch"]: d["probability"] for d in rb["predictions"]}
    diff = {c: (pb[c] - pa[c]) * 100 for c in pa}
    attr_diff = None
    if ra["attribution"] and rb["attribution"]:
        top_b = rb["predictions"][0]["pitch"]
        ga = ra["attribution"]["per_class"][top_b]["by_group"]
        gb = rb["attribution"]["per_class"][top_b]["by_group"]
        attr_diff = {"class": top_b, "by_group": {g: gb.get(g, 0) - ga.get(g, 0) for g in set(ga) | set(gb)}}
    return {"a": ra, "b": rb, "percentage_point_diff": diff,
            "top_changed": ra["predictions"][0]["pitch"] != rb["predictions"][0]["pitch"],
            "attribution_diff": attr_diff, "note": "Differences reflect model sensitivity, not causal effects."}


class SweepReq(BaseModel):
    state: SimState
    field: str
    values: list | None = None
    model_version: str | None = None


DEFAULT_SWEEPS = {
    "pitch_count_before": list(range(0, 121, 5)),
    "count": [[b, s] for b in range(4) for s in range(3)],
    "inning": list(range(1, 10)),
    "bat_order_slot": list(range(1, 10)),
    "stand": ["L", "R"],
    "times_through_order": [1, 2, 3, 4],
    "outs": [0, 1, 2],
    "score_diff": list(range(-5, 6)),
}


@router.post("/pitchers/{pid}/simulate/sweep")
def do_sweep(pid: int, req: SweepReq, db: Session = Depends(get_db)):
    if req.field not in DEFAULT_SWEEPS:
        raise ValueError(f"field must be one of {sorted(DEFAULT_SWEEPS)}")
    vals = req.values or DEFAULT_SWEEPS[req.field]
    if req.field == "count":
        vals = [tuple(v) for v in vals]
    return {**sweep(db, pid, req.state, req.field, vals, req.model_version),
            "note": "Single-variable sensitivity of the trained model; not a causal effect."}


class ScenarioIn(BaseModel):
    name: str
    state: SimState


@router.get("/pitchers/{pid}/scenarios")
def list_scenarios(pid: int, db: Session = Depends(get_db)):
    return [{"scenario_id": s.scenario_id, "name": s.name, "state": s.state, "created_at": str(s.created_at)}
            for s in db.scalars(select(SavedScenario).where(SavedScenario.pitcher_id == pid)
                                .order_by(SavedScenario.created_at.desc()))]


@router.post("/pitchers/{pid}/scenarios")
def save_scenario(pid: int, body: ScenarioIn, db: Session = Depends(get_db)):
    s = SavedScenario(pitcher_id=pid, name=body.name, state=body.state.model_dump(mode="json"))
    db.add(s)
    db.flush()
    return {"scenario_id": s.scenario_id}


@router.delete("/pitchers/{pid}/scenarios/{sid}")
def delete_scenario(pid: int, sid: int, db: Session = Depends(get_db)):
    s = db.get(SavedScenario, sid)
    if s and s.pitcher_id == pid:
        db.delete(s)
    return {"deleted": bool(s)}
