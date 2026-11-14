"""As-of-cutoff views of the aging observations and cohorts.

The stored aging_observations / aging_cohorts are built from every loaded season (2015-2026). Most columns of an
observation only use seasons t and t+1, but `returned_later`, `next_qualifying_season` (and the cohorts'
`n_returned_later`) look ahead to any later season. A forecast or backtest with a historical cutoff must therefore not
read the stored columns directly: these helpers keep only pairs with season_to <= cutoff and recompute the look-ahead
columns from seasons <= cutoff (a pitcher whose next qualifying season is after the cutoff has, as of the cutoff, not
returned; `outcome_censored` marks such non-survivors whose return is unknown at the cutoff).
"""
from __future__ import annotations

import pandas as pd
from sqlalchemy import text
from sqlalchemy.orm import Session

from .etl import COHORT_SUMMARY, build_cohorts


def apply_cutoff(obs: pd.DataFrame, cutoff: int) -> pd.DataFrame:
    """Pure function: restrict an observation frame to information available at the end of season `cutoff`."""
    cutoff = int(cutoff)
    out = obs[pd.to_numeric(obs["season_to"]) <= cutoff].copy()
    if out.empty:
        out["outcome_censored"] = pd.Series(dtype=bool)
        return out
    nq = pd.to_numeric(out["next_qualifying_season"], errors="coerce")
    nq = nq.where(nq <= cutoff)
    out["next_qualifying_season"] = nq.astype("Int64")
    survived = out["survived"].astype(bool)
    out["returned_later"] = (~survived) & nq.notna()
    # non-survivors with no qualifying season by the cutoff: retired or not yet returned (right-censored)
    out["outcome_censored"] = (~survived) & nq.isna()
    out["as_of_season"] = cutoff
    return out


def load_observations_as_of(session: Session, cutoff: int, pitch_family: str | None = None,
                            metric: str | None = None) -> pd.DataFrame:
    sql = "SELECT * FROM aging_observations WHERE season_to <= :c"
    params: dict = {"c": int(cutoff)}
    if pitch_family is not None:
        sql += " AND pitch_family = :f"
        params["f"] = pitch_family
    if metric is not None:
        sql += " AND metric = :m"
        params["m"] = metric
    df = pd.read_sql(text(sql + " ORDER BY season_from, pitcher_id"), session.connection(), params=params)
    return apply_cutoff(df, cutoff)


def cohorts_as_of(session: Session, cutoff: int) -> list[dict]:
    """Cohort rows (same shape as aging_cohorts, not stored) recomputed from observations as of `cutoff`."""
    pairs = [("ALL", "pitch_count")] + list(COHORT_SUMMARY)
    cond = " OR ".join(f"(pitch_family = :f{i} AND metric = :m{i})" for i in range(len(pairs)))
    params: dict = {"c": int(cutoff)}
    for i, (f, m) in enumerate(pairs):
        params[f"f{i}"], params[f"m{i}"] = f, m
    df = pd.read_sql(text(f"SELECT * FROM aging_observations WHERE season_to <= :c AND ({cond})"),
                     session.connection(), params=params)
    df = apply_cutoff(df, cutoff)
    if df.empty:
        return []
    rows = build_cohorts(df, None)
    for r in rows:
        r["as_of_season"] = int(cutoff)
    return rows


__all__ = ["apply_cutoff", "load_observations_as_of", "cohorts_as_of"]
