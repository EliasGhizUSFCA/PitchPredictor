"""Cut offline fixtures for the batter-leaderboard ETL from REAL cached Savant responses (run after a live load).

Keeps every row of the batters in tests/fixtures/statcast_subset.csv plus the first 150 other batters of each
response (enough units for the empirical-Bayes strength estimate). Output: fixtures/batter_lb/<source>_<season>[_hand].csv
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pitchiq import batter_priors as bp  # noqa: E402
from pitchiq.aging.sources import SourceLog  # noqa: E402
from pitchiq.http import PoliteClient  # noqa: E402

OUT = Path(__file__).parent / "fixtures" / "batter_lb"
OUT.mkdir(parents=True, exist_ok=True)
batters = set(pd.read_csv(Path(__file__).parent / "fixtures" / "statcast_subset.csv", low_memory=False)["batter"])
c, log = PoliteClient(), SourceLog()
for season in range(2020, 2027):
    for name, f in (("pitch_type", bp._fetch_pitch_type(c, season, log, True)),
                    ("custom", bp._fetch_custom(c, season, log, True)),
                    ("platoon_L", bp._fetch_platoon(c, season, "L", log, True))):
        d = f.frame
        ids = d["player_id"]
        others = [x for x in pd.unique(ids) if x not in batters][:150]
        d[ids.isin(batters | set(others))].to_csv(OUT / f"{name}_{season}.csv", index=False)
print(log.as_dict())
