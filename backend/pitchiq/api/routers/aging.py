"""Aging dataset API (read-only). All values are stored Savant / MLB Stats API figures or deltas computed from them."""
from __future__ import annotations

from fastapi import APIRouter, Depends, Query
from sqlalchemy import select, text
from sqlalchemy.orm import Session

from ...db import get_db
from ...models_aging import (AgingCohort, AgingEtlRun, AgingObservation, PitcherBio, PitcherSeasonPitchStats,
                             PitcherSeasonStats)
from ...seasons import modeling_seasons

router = APIRouter(prefix="/aging", tags=["aging"])

_PSS_SKIP = {"source_urls", "updated_at"}


def _row(obj, skip=frozenset()) -> dict:
    return {c.name: (str(v) if hasattr(v, "isoformat") else v)
            for c in obj.__table__.columns if c.name not in skip for v in [getattr(obj, c.name)]}


@router.get("/coverage")
def coverage(db: Session = Depends(get_db)):
    run = db.scalars(select(AgingEtlRun).where(AgingEtlRun.status == "succeeded")
                     .order_by(AgingEtlRun.etl_run_id.desc())).first()
    if run is None:
        return {"etl_run": None, "note": "aging ETL has not run (pitchiq aging-etl)"}
    return {"etl_run": {"etl_run_id": run.etl_run_id, "started_at": str(run.started_at),
                        "finished_at": str(run.finished_at), "params": run.params},
            "coverage": run.coverage, "validation": run.validation, "sources": run.sources, "units": run.units}


@router.get("/pitchers/{pid}")
def pitcher_seasons(pid: int, include_sources: bool = False, db: Session = Depends(get_db)):
    bio = db.get(PitcherBio, pid)
    skip = frozenset() if include_sources else frozenset(_PSS_SKIP | {"values_by_source", "field_sources"})
    seasons = [_row(r, skip) for r in db.scalars(select(PitcherSeasonStats).where(PitcherSeasonStats.pitcher_id == pid)
                                                 .order_by(PitcherSeasonStats.season))]
    types = [_row(r, skip) for r in db.scalars(
        select(PitcherSeasonPitchStats).where(PitcherSeasonPitchStats.pitcher_id == pid)
        .order_by(PitcherSeasonPitchStats.season, PitcherSeasonPitchStats.pitches.desc()))]
    if bio is None and not seasons:
        raise LookupError("pitcher not in aging dataset")
    return {"bio": None if bio is None else _row(bio, {"updated_at"}), "seasons": seasons, "pitch_types": types}


@router.get("/pitchers/{pid}/career-pitches")
def career_pitches(pid: int, db: Session = Depends(get_db)):
    """Per-season, per-pitch-type summary of the pitch-level rows in `pitches` with exact age at each pitch
    (pitch_ages view). Seasons outside the modeling seasons are labelled history_only."""
    ms = set(modeling_seasons())
    rows = db.execute(text("""
        SELECT p.season, p.pitch_type, count(*) AS n, avg(p.release_speed) AS speed, avg(p.release_spin_rate) AS spin,
               avg(p.pfx_x) AS pfx_x, avg(p.pfx_z) AS pfx_z, min(a.age_exact_years) AS age_min,
               max(a.age_exact_years) AS age_max, count(DISTINCT p.game_pk) AS games
        FROM pitches p LEFT JOIN pitch_ages a ON a.pitch_uid = p.pitch_uid
        WHERE p.pitcher_id = :p AND p.pitch_type IS NOT NULL
        GROUP BY p.season, p.pitch_type ORDER BY p.season, n DESC"""), {"p": pid}).all()
    out: dict[int, dict] = {}
    for r in rows:
        s = out.setdefault(int(r.season), {"season": int(r.season),
                                           "role": "modeling" if int(r.season) in ms else "history_only",
                                           "pitches": 0, "age_range": None, "pitch_types": []})
        s["pitches"] += int(r.n)
        if r.age_min is not None:
            lo, hi = float(r.age_min), float(r.age_max)
            s["age_range"] = [min(lo, s["age_range"][0]), max(hi, s["age_range"][1])] if s["age_range"] else [lo, hi]
        s["pitch_types"].append({"pitch_type": r.pitch_type, "n": int(r.n), "games": int(r.games),
                                 "avg_speed_mph": None if r.speed is None else float(r.speed),
                                 "avg_spin_rpm": None if r.spin is None else float(r.spin),
                                 "avg_pfx_x_ft": None if r.pfx_x is None else float(r.pfx_x),
                                 "avg_pfx_z_ft": None if r.pfx_z is None else float(r.pfx_z)})
    for s in out.values():
        for t in s["pitch_types"]:
            t["usage_pct"] = 100.0 * t["n"] / s["pitches"]
    return {"pitcher_id": pid, "source": "Statcast pitch-level rows in `pitches`; ages from pitcher_bio.birth_date",
            "seasons": list(out.values())}


@router.get("/cohorts")
def cohorts(cohort_type: str = "age", scope: str = "pooled", as_of: int | None = None,
            db: Session = Depends(get_db)):
    if as_of is not None:  # recomputed from seasons <= as_of only (no look-ahead in returned_later)
        from ...aging.asof import cohorts_as_of
        rows = [r for r in cohorts_as_of(db, as_of) if r["cohort_type"] == cohort_type and r["scope"] == scope]
        return [{k: (str(v) if hasattr(v, "isoformat") else v) for k, v in r.items()}
                for r in sorted(rows, key=lambda r: r["cohort_key"])]
    rows = db.scalars(select(AgingCohort).where(AgingCohort.cohort_type == cohort_type, AgingCohort.scope == scope)
                      .order_by(AgingCohort.cohort_key)).all()
    return [_row(r) for r in rows]


@router.get("/observations")
def observations(pitch_family: str = "ALL", metric: str = "fastball_avg_speed_mph", pitcher_id: int | None = None,
                 both_qualified: bool | None = None, limit: int = Query(1000, le=20000), as_of: int | None = None,
                 db: Session = Depends(get_db)):
    if as_of is not None:  # pairs with season_to <= as_of; returned_later / next_qualifying_season as of the cutoff
        import json

        from ...aging.asof import load_observations_as_of
        df = load_observations_as_of(db, as_of, pitch_family, metric)
        if pitcher_id is not None:
            df = df[df["pitcher_id"] == pitcher_id]
        if both_qualified is not None:
            df = df[df["both_qualified"] == both_qualified]
        return json.loads(df.head(limit).to_json(orient="records", date_format="iso"))
    q = select(AgingObservation).where(AgingObservation.pitch_family == pitch_family,
                                       AgingObservation.metric == metric)
    if pitcher_id is not None:
        q = q.where(AgingObservation.pitcher_id == pitcher_id)
    if both_qualified is not None:
        q = q.where(AgingObservation.both_qualified == both_qualified)
    rows = db.scalars(q.order_by(AgingObservation.season_from, AgingObservation.pitcher_id).limit(limit)).all()
    return [_row(r) for r in rows]
