"""Baseball Savant Statcast Search CSV ingestion (the same endpoint pybaseball wraps).

Docs: https://baseballsavant.mlb.com/csv-docs
"""
from __future__ import annotations

import io
import logging
from datetime import date, timedelta

import pandas as pd

from ..http import PoliteClient
from ..logging_setup import log

logger = logging.getLogger(__name__)
SAVANT_CSV = "https://baseballsavant.mlb.com/statcast_search/csv"


def _windows(season: int, days: int = 31):
    start, end = date(season, 3, 1), date(season, 11, 30)
    cur = start
    while cur <= end:
        nxt = min(cur + timedelta(days=days - 1), end)
        yield cur, nxt
        cur = nxt + timedelta(days=1)


def fetch_pitcher_season(client: PoliteClient, pitcher_id: int, season: int, game_types: list[str],
                         use_cache: bool = True) -> pd.DataFrame:
    """Fetch every pitch thrown by `pitcher_id` in `season`, chunked by month (Savant caps rows per request)."""
    frames = []
    for lo, hi in _windows(season):
        params = {
            "all": "true",
            "player_type": "pitcher",
            "pitchers_lookup[]": str(pitcher_id),
            "game_date_gt": lo.isoformat(),
            "game_date_lt": hi.isoformat(),
            "hfGT": "".join(f"{g}|" for g in game_types),
            "hfSea": f"{season}|",
            "type": "details",
        }
        # Only cache completed months: a future/in-progress window could change.
        cacheable = use_cache and hi < date.today() - timedelta(days=3)
        text = client.get(SAVANT_CSV, params, namespace="statcast", suffix=".csv", use_cache=cacheable)
        if not text.strip():
            continue
        df = pd.read_csv(io.StringIO(text), low_memory=False)
        if "pitch_type" not in df.columns or "game_pk" not in df.columns:
            raise RuntimeError(f"Unexpected Statcast response for {lo}..{hi}: columns={list(df.columns)[:5]}")
        if len(df):
            frames.append(df)
            log(logger, "statcast_window", pitcher_id=pitcher_id, season=season, start=str(lo), end=str(hi), rows=len(df))
        if len(df) >= 24000:
            raise RuntimeError("Savant row cap reached; shrink the window size")
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    df = df.drop_duplicates(subset=["game_pk", "at_bat_number", "pitch_number"])
    return df
