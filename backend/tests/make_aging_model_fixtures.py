"""Cut the aging-model test fixtures from the real aging dataset in the main database.

Writes tests/fixtures/aging_model/{pss,psps,bio}.csv: real pitcher_season_stats / pitcher_season_pitch_stats
(four-seam and slider codes only) / pitcher_bio rows for Chris Sale plus 119 other pitchers (seeded sample: 60 whose
careers reach age 33, 59 others, all with >= 4 seasons of >= 100 four-seamers), every season 2015-2026.

Run from backend/: python tests/make_aging_model_fixtures.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import text

from pitchiq.db import session_scope

OUT = Path(__file__).parent / "fixtures" / "aging_model"
SALE = 519242
PSS_COLS = ["pitcher_id", "season", "age_june30_exact", "pitch_count", "games", "games_started", "batters_faced",
            "ip", "quality_flags", "in_zone_pct", "oz_swing_pct", "whiff_pct", "k_pct", "bb_pct", "arm_angle_deg"]
PSPS_COLS = ["pitcher_id", "season", "pitch_type", "pitch_family", "pitches", "avg_speed_mph", "avg_spin_rpm", "ivb_in",
             "break_x_in", "extension_ft", "release_pos_x_ft", "release_pos_z_ft"]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    with session_scope() as s:
        c = s.connection()
        cand = pd.read_sql(text("""
            SELECT p.pitcher_id, count(*) AS n_seasons, max(t.age_june30) AS max_age
            FROM pitcher_season_pitch_stats p JOIN pitcher_season_stats t USING (pitcher_id, season)
            WHERE p.pitch_type = 'FF' AND p.pitches >= 100 GROUP BY p.pitcher_id HAVING count(*) >= 4"""), c)
        rng = np.random.default_rng(7)
        old = cand[(cand["max_age"] >= 33) & (cand["pitcher_id"] != SALE)]["pitcher_id"].to_numpy()
        rest = cand[(cand["max_age"] < 33)]["pitcher_id"].to_numpy()
        ids = [SALE] + sorted(rng.choice(old, 60, replace=False).tolist()) + sorted(
            rng.choice(rest, 59, replace=False).tolist())
        idl = ",".join(map(str, ids))
        pss = pd.read_sql(text(f"SELECT {', '.join(PSS_COLS)} FROM pitcher_season_stats WHERE pitcher_id IN ({idl})"), c)
        psps = pd.read_sql(text(f"SELECT {', '.join(PSPS_COLS)} FROM pitcher_season_pitch_stats WHERE pitcher_id IN "
                                f"({idl}) AND pitch_family IN ('four_seam', 'slider')"), c)
        bio = pd.read_sql(text(f"SELECT pitcher_id, full_name, birth_date, throws FROM pitcher_bio "
                               f"WHERE pitcher_id IN ({idl})"), c)
    pss["quality_flags"] = pss["quality_flags"].map(lambda v: "|".join(v or []))
    pss.to_csv(OUT / "pss.csv", index=False)
    psps.to_csv(OUT / "psps.csv", index=False)
    bio.to_csv(OUT / "bio.csv", index=False)
    print(len(ids), len(pss), len(psps), len(bio))


if __name__ == "__main__":
    main()
