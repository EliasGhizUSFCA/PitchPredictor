"""Cut offline aging-ETL fixtures from REAL cached responses (run after a live `pitchiq aging-etl`).

Every Savant leaderboard / grouped-search response the ETL requests for SEASONS is replayed from the raw-response
cache and filtered to PITCHERS; MLB Stats API people are filtered the same way; one real Chris Sale 2019 game is cut
from the cached Statcast pitch-level response. Nothing is edited besides dropping rows.
"""
import io
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pitchiq.aging import etl, sources  # noqa: E402
from pitchiq.http import PoliteClient  # noqa: E402
from pitchiq.ingest import statcast  # noqa: E402

OUT = Path(__file__).parent / "fixtures" / "aging"
SEASONS = [2019, 2020, 2022, 2023]
# Sale; Webb; Houck (SL -> SL+ST in 2023); Sabathia and Felix Hernandez (last season 2019); Flaherty (KC folded into
# CU); Scherzer.
PITCHERS = [519242, 657277, 656557, 282332, 433587, 656427, 453286]
SALE_GAME_2019 = None  # first 2019 regular-season game in the cached response


class Recorder(PoliteClient):
    def __init__(self):
        super().__init__()
        self.seen: dict[str, tuple[str, str]] = {}

    def get(self, url, params=None, *, namespace, suffix=".txt", use_cache=True, retries=3):
        text = super().get(url, params, namespace=namespace, suffix=suffix, use_cache=True, retries=retries)
        self.seen[url + "?" + json.dumps(params or {}, sort_keys=True)] = (text, suffix)
        return text


def _filter_csv(text: str) -> str:
    if not text.strip():
        return text
    df = pd.read_csv(io.StringIO(text.lstrip("﻿")), low_memory=False, dtype=str, keep_default_na=False)
    col = next(c for c in ("pitcher_id", "pitcher", "player_id") if c in df.columns)
    df = df[df[col].astype(str).isin({str(p) for p in PITCHERS})]
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rec = Recorder()
    logs = {k: sources.SourceLog() for k in ("movement", "arsenals", "arsenal_stats", "custom", "search", "people")}
    for s in SEASONS:
        etl.fetch_season(rec, s, logs)
    sources.fetch_people(rec, PITCHERS, logs["people"])
    index = {}
    for i, (key, (text, suffix)) in enumerate(sorted(rec.seen.items())):
        name = f"r{i:03d}{suffix}"
        if suffix == ".json":
            d = json.loads(text)
            keep = ("id", "fullName", "birthDate", "pitchHand", "primaryPosition", "birthCountry", "mlbDebutDate")
            d = {"people": [{k: p.get(k) for k in keep} for p in d.get("people", []) if p["id"] in PITCHERS]}
            (OUT / name).write_text(json.dumps(d))
        else:
            (OUT / name).write_text(_filter_csv(text))
        index[key] = name
    (OUT / "index.json").write_text(json.dumps(index, indent=0, sort_keys=True))
    df = statcast.fetch_pitcher_season(PoliteClient(), 519242, 2019, ["R"])
    first = sorted(df["game_date"].unique())[0]
    g = df[df["game_date"] == first]
    g.to_csv(OUT / "sale_statcast_2019_subset.csv", index=False)
    print(len(index), "responses;", len(g), "Sale 2019 pitches from", first)


if __name__ == "__main__":
    main()
