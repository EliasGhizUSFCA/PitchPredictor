# Aging data: MLB-wide longitudinal pitching dataset + Chris Sale career pitches

Step key: `aging-data`. Code: `backend/pitchiq/aging/{sources,families,etl}.py`, ORM `backend/pitchiq/models_aging.py`,
migrations `aging01_dataset` (tables + `pitch_ages` view) and `aging02_bio_url` (widens `pitcher_bio.source_url` to
TEXT), season isolation `backend/pitchiq/seasons.py`, API `backend/pitchiq/api/routers/aging.py` (`/api/aging/*`),
CLI `pitchiq aging-etl`, job kind `aging_etl`, tests `backend/tests/test_aging.py` (fixtures in
`backend/tests/fixtures/aging/`, cut from real responses by `backend/tests/make_aging_fixtures.py`).

Every number below was read from the main database (`pitchiq`) after ETL run **`aging_etl_runs.etl_run_id = 3`**
(status `succeeded`, 7 min 23 s, replayed from the raw-response cache that the first live run filled; coverage JSON
is byte-identical to the earlier run 2). Coverage, validation, source logs and unit descriptions are stored on that
row (`coverage`, `validation`, `sources`, `units`) and served by `GET /api/aging/coverage`.

## Sources (all through `PoliteClient`: 1 req/s per host, raw responses cached under `backend/data/raw`)

Each endpoint and each column used was checked against real responses before use. A few stored values were checked
again against fresh (uncached) live requests in this step: Sale 2024 custom leaderboard (pitch_count 2815, K% 32.1,
BB% 5.6, zone% 50.8, chase% 33.6, whiff% 31.0, FB velo 94.7), Sale 2019 SL movement (947 pitches, 79.0 mph, IVB -4.2
in, HB 14.0 in), Sale 2026 pitch-arsenal-stats (4 pitch types, e.g. SL 1027 pitches, usage 39.9, whiff 38.2, RV/100
1.2) and MLB Stats API birth date 1989-03-30, throws L. All matched the stored rows exactly.

| source key | endpoint | used for | seasons with data |
|---|---|---|---|
| `movement` | `/leaderboard/pitch-movement?year=Y&pitch_type=T&min=1&csv=true` (12 types FF SI FC SL ST SV CH CU FS KN FO SC) | pitches per type, avg_speed, break z / induced z / x, diff vs league | 2015-2026 |
| `arsenals` | `/leaderboard/pitch-arsenals?type=avg_speed\|avg_spin\|n_&min=1` | per-type speed, spin, usage % | 2015-2026 |
| `arsenal_stats` | `/leaderboard/pitch-arsenal-stats?type=pitcher&min=1` | usage, whiff %, K %, put-away %, run value, RV/100, wOBA, xwOBA, hard-hit % | 2017-2026 (2015 and 2016 return a header only) |
| `custom` | `/leaderboard/custom?type=pitcher&min=1&selections=...` | pitcher-season workload + command proxies (list below) | 2015-2026 |
| `search` | `/statcast_search/csv?group_by=name` per season (all pitches) and per pitch type (`hfPT`) | release extension, release position x/z, arm angle (2020+), whiffs/swings per type (fills whiff % for 2015-2016) | 2015-2026 |
| `people` | `statsapi.mlb.com/api/v1/people?personIds=...` (100 ids per request) | birth date, throwing hand, primary position, debut date | - |

Custom selections that return data (verified): `player_age, p_game, p_starting_p, p_formatted_ip, p_out, pa,
pitch_count, k_percent, bb_percent, in_zone_percent, oz_swing_percent, z_swing_percent, swing_percent, whiff_percent,
edge_percent, f_strike_percent, meatball_percent, iz_contact_percent, oz_contact_percent, fastball_avg_speed,
fastball_avg_spin, breaking_avg_speed, breaking_avg_spin, offspeed_avg_speed, offspeed_avg_spin, pitch_hand,
arm_angle, xwoba, p_era`. `release_extension` / `release_pos_*` come back blank from the custom leaderboard, so they are
taken from the grouped Statcast search.

Request log for run 3 (`aging_etl_runs.sources`): movement 144 requests / 41,185 rows, arsenals 36 / 28,311,
arsenal_stats 12 / 32,936, custom 12 / 9,355, search 192 / 51,454, people 28 / 2,778. No failures, no missing
required columns. Empty responses are genuine "no pitcher threw this type" answers (SC 2019/2020/2022/2026, FO
2020-2022) plus arsenal-stats 2015/2016.

## Tables

All value columns carry their unit in the name (`_mph`, `_rpm`, `_in`, `_ft`, `_pct` on a 0-100 scale, `_deg`);
`pitchiq.aging.etl.UNITS` describes each one and is stored on each `aging_etl_runs` row.

* **`pitcher_bio`** (2,778 rows): MLBAM id, full name, birth date, throws, primary position, birth country, debut date,
  `source_url` (the exact bulk request URL, up to 946 characters; migration `aging02_bio_url` removed a 512-char
  truncation that the first version applied), `fetched_at`.
* **`pitcher_season_stats`** (9,836 rows, PK pitcher_id + season): games, GS, outs, IP (= outs / 3, true innings),
  batters faced, pitch count, K %, BB %, zone %, chase (o-swing) %, z-swing %, swing %, whiff %, edge %, first-strike %,
  meatball %, contact rates, fastball / breaking / offspeed speed and spin, extension, release position, arm angle,
  xwOBA, ERA; `age_june30` (completed years) and `age_june30_exact` (days / 365.2425) computed from `pitcher_bio`;
  `savant_player_age` (cross-check only); `measurement_era`, `low_sample`, `quality_flags`, `field_sources` (column ->
  source key), `source_urls` (source key -> exact URL), `fetched_at` (oldest upstream fetch used), `etl_run_id`.
* **`pitcher_season_pitch_stats`** (41,493 rows, PK pitcher_id + season + pitch_type): official leaderboard code,
  `pitch_family`, `pitch_group`, pitches, usage %, speed, spin, break z / IVB / break x and league diffs, extension,
  release position, arm angle, swings, whiffs, whiff %, K %, put-away %, run value, RV/100, wOBA, xwOBA, hard-hit %,
  `values_by_source` (each source's raw value, for audit), `field_sources`, `source_urls`, flags.
* **`aging_observations`** (321,166 rows): one row per (pitcher, unit, metric, season t -> t+1) for every unit that
  qualified in season t, including non-survivors. Unit (`pitch_family` column) is a pitch family, a pitch group
  (`group:fastball|breaking|offspeed`) or `ALL` (pitcher-level metrics). Columns: ages (exact and completed) at both
  June 30s, birth year, `value_from`, `value_to`, `delta`, pitches of the unit in both seasons, workload (pitches, IP)
  in both seasons, `survived`, `unit_present_next`, `both_qualified`, `returned_later`, `next_qualifying_season`,
  `era_from/era_to/era_change`, `short_season`, `reclass_flag`, `codes_from/codes_to`, thresholds used.
* **`aging_cohorts`** (326 rows): by completed age on June 30 and by 5-year birth-year band, pooled and per
  `season_from`: at-risk count, survived, dropped, returned later, era-change count, survival rate, mean age, and
  delta summaries (n / mean / sd / median) for 14 unit-metric pairs (pairs with an era change or a 2020 season are
  excluded from the delta summaries; both seasons must qualify).
* **`aging_etl_runs`**: params, per-source request logs, coverage, validation, units.
* **`pitch_ages`** (view): exact age (`(game_date - birth_date) / 365.2425`) and completed years for every row of
  `pitches` whose pitcher has a birth date. Computed at query time, never stored or typed in.

### Merge precedence (first non-null wins, recorded per row in `field_sources`)

Pitch-type rows: pitches movement > arsenal_stats > search; usage arsenals > arsenal_stats > movement; speed movement >
arsenals > search; spin arsenals > search; break / IVB movement only; extension, release position, arm angle,
swings, whiffs search only; whiff % arsenal_stats > search (whiffs / swings); K %, put-away, run value, wOBA, xwOBA,
hard-hit arsenal_stats only. Season rows: custom first, search as fallback for batters faced, pitch count, K %, BB %,
whiff %; extension and release position from search; arm angle custom > search.

## Rules and definitions

* **Thresholds** (config, recorded on every observation row): a pitch-type season is `low_sample` below
  `PITCHIQ_AGING_MIN_PITCH_TYPE_PITCHES` = 100 pitches; a pitcher-season qualifies at `PITCHIQ_AGING_MIN_SEASON_PITCHES`
  = 500 pitches. Low-sample rows are stored, flagged, never dropped.
* **Measurement era**: `trackman` for 2015-2019, `hawkeye` for 2020+. 2019 -> 2020 pairs carry `era_change = true`;
  2020 (60 games) carries `short_season` on season rows and on both adjoining pairs.
* **Pitch reclassification**: official codes are stored unchanged. `pitch_family` maps SL/ST/SV -> slider, CU/KC/CS ->
  curveball, FF/FA -> four_seam, SI/FT -> sinker, FS/FO -> splitter (plus cutter, changeup, screwball, knuckleball,
  eephus); `pitch_group` maps families to Savant's fastball / breaking / offspeed. Savant leaderboards fold KC (and CS)
  into CU; the per-type grouped searches are folded the same way (pitch-weighted means), so a knuckle-curve pitcher has
  one CU row. ST/SV are not only a 2023+ phenomenon in what Savant serves today: ST pitcher-season rows by season are
  7, 11, 13, 18, 27, 56, 118, 176, 286, 333, 357, 387 (2015-2026), i.e. older seasons are only sparsely labelled as
  sweepers and the share jumps 2021-2023. Family-level deltas are therefore the comparable unit; per-code deltas across
  these seasons are not. `reclass_flag` marks observations whose set of codes inside the unit changed between t and
  t+1: 345 pitcher-seasons for the slider family, 1 for splitter, 1,081 / 837 / 122 for the fastball / breaking /
  offspeed groups (the fastball group includes ordinary four-seam / sinker / cutter mix changes, not only
  relabelling).
* **Age**: age on June 30 of the season (season midpoint), from the MLB Stats API birth date. Savant's own
  `player_age` agrees with the computed completed age on all 9,350 compared pitcher-seasons.
* **Survivor bias**: every pitcher-season that qualified in t and whose season t+1 was fetched is "at risk" and gets an
  observation row even if season t+1 is missing or below threshold (`survived = false`, `value_to`/`delta` null).
  `returned_later` / `next_qualifying_season` separate injuries / demotions with a later comeback from exits.
  Season pairs whose t+1 was not fetched produce no rows, so an unfetched season never looks like a dropout. For
  2025 -> 2026 non-survivors `returned_later` is right-censored (no later season exists yet).

## Coverage (run 3)

Totals: 2,778 pitchers, 9,836 pitcher-seasons (5,309 qualifying at >= 500 pitches), 12 seasons (2015-2026), 41,493
pitch-type rows, 13 distinct official codes, 10 pitch families. Birth dates: 2,778 / 2,778 pitchers (2,778 people
requested, 2,778 returned).

| season | pitchers | qualifying (>=500 p) | pitch-type rows | not low-sample | codes | with zone/chase | with extension | with arm angle | with run value | pitches in type rows |
|---|---|---|---|---|---|---|---|---|---|---|
| 2015 | 735 | 439 | 3,206 | 1,671 | 13 | 724 | 735 | 0 | 0 | 698,221 |
| 2016 | 742 | 439 | 3,123 | 1,672 | 13 | 733 | 742 | 0 | 0 | 711,700 |
| 2017 | 755 | 464 | 3,229 | 1,745 | 13 | 741 | 755 | 0 | 3,063 | 717,833 |
| 2018 | 799 | 470 | 3,279 | 1,712 | 13 | 771 | 798 | 0 | 3,081 | 717,850 |
| 2019 | 831 | 475 | 3,347 | 1,783 | 12 | 803 | 830 | 0 | 3,168 | 727,406 |
| 2020 | 735 | 166 | 2,921 | 968 | 11 | 705 | 735 | 715 | 2,698 | 263,262 |
| 2021 | 909 | 484 | 3,678 | 1,806 | 12 | 845 | 909 | 906 | 3,408 | 708,615 |
| 2022 | 871 | 473 | 3,536 | 1,800 | 11 | 808 | 871 | 871 | 3,280 | 706,840 |
| 2023 | 863 | 479 | 3,659 | 1,864 | 13 | 799 | 863 | 863 | 3,387 | 716,512 |
| 2024 | 855 | 473 | 3,744 | 1,864 | 13 | 800 | 855 | 854 | 3,510 | 708,540 |
| 2025 | 873 | 478 | 3,824 | 1,941 | 13 | 809 | 873 | 873 | 3,604 | 708,940 |
| 2026 | 868 | 469 | 3,947 | 1,991 | 12 | 812 | 868 | 868 | 3,737 | 713,266 |

Arm angle exists only from 2020 (Hawk-Eye); run value per pitch type only from 2017 (arsenal-stats). The pitcher-season
union contains pitchers who appear in the grouped search but not in the custom leaderboard (`no_custom_row`: 11 to 64
per season); those have no zone / chase values.

Observations by unit (rows / at-risk pitcher-seasons): ALL 75,153 / 4,840; group:fastball 42,637 / 4,835;
group:breaking 38,632 / 4,380; four_seam 35,861 / 4,069; slider 29,593 / 3,347; group:offspeed 23,411 / 2,656;
sinker 22,760 / 2,586; changeup 19,965 / 2,267; curveball 18,372 / 2,089; cutter 11,223 / 1,267; splitter 3,474 / 392;
knuckleball 68 / 8; screwball 9 / 1; eephus 8 / 2.

Survival (ALL / pitch_count rows): 4,840 at-risk pitcher-seasons, 3,141 survived (qualified in t+1), 1,699 dropped, of
which 568 returned in a later season; 475 at-risk pairs span the 2019 -> 2020 era change, 641 involve 2020.

Pooled survival by completed age on June 30 of season t (cohorts, `cohort_type='age'`, `scope='pooled'`), with the
raw mean change in season fastball velocity (mph) and K % (points) among pairs where both seasons qualified and no era
change / 2020 season is involved. These are descriptive survivor-only means, not aging curves: they are exactly the
quantities that survivor bias distorts, and modelling them is the next step's job.

| age | at risk | survived | dropped | survival | n (FB velo pairs) | mean d FB velo | mean d K % |
|---|---|---|---|---|---|---|---|
| 24 | 334 | 236 | 98 | 0.707 | 211 | +0.03 | 0.00 |
| 26 | 535 | 361 | 174 | 0.675 | 329 | -0.10 | +0.14 |
| 28 | 486 | 322 | 164 | 0.663 | 296 | -0.21 | -0.41 |
| 30 | 379 | 246 | 133 | 0.649 | 228 | -0.24 | -0.59 |
| 32 | 294 | 202 | 92 | 0.687 | 185 | -0.30 | -0.76 |
| 34 | 173 | 107 | 66 | 0.618 | 102 | -0.18 | -0.88 |
| 35 | 117 | 62 | 55 | 0.530 | 56 | -0.33 | -1.37 |
| 36 | 81 | 38 | 43 | 0.469 | 34 | -0.26 | -1.80 |
| 37 | 48 | 28 | 20 | 0.583 | 26 | -0.52 | -1.40 |
| 38 | 34 | 14 | 20 | 0.412 | 13 | -0.31 | +0.25 |

(Full table, ages 19-45, and birth-year bands 1970-1974 ... 2000-2004 are in `aging_cohorts`; small cells such as
ages >= 39 have fewer than 15 at-risk pitcher-seasons each.)

## Cross-source validation (stored in `aging_etl_runs.validation`, run 3)

* Pitches per (pitcher, season, type), sources compared where >= 2 report it: 41,195 compared, 1 disagreement
  (pitcher 622663, 2022 SL: movement and arsenal-stats 331, grouped search 324; flagged
  `pitch_count_sources_disagree`, the movement value is stored).
* Speed, movement vs arsenals leaderboard: 40,928 compared, 87 differ by > 0.25 mph (agreement 0.9979); 52 of them
  are 2015-2016, mostly CU / SL / CH / KN (mean absolute gap 2.4 mph for the 26 CU rows), consistent with those
  leaderboards folding codes differently in early seasons. Flagged `speed_sources_disagree`; values from both sources
  kept in `values_by_source`.
* Season pitch count, custom vs grouped search: 9,350 compared, 0 disagreements. IP from `p_formatted_ip` vs `p_out`:
  9,350 compared, 0 mismatches. Savant age vs computed June 30 age: 9,350 compared, 0 differences.
* Duplicate source keys collapsed: 15 pitch-type rows and 5 pitcher-season rows, all switch-pitcher Pat Venditte
  (519381; 2015, 2016, 2018, 2019, 2020), who is listed once per throwing hand; values that agree are kept, values that differ are nulled, never averaged without weights;
  flagged `multiple_source_rows_per_key`.
* Other flags: 558 pitcher-seasons are position players pitching (`position_player`); 4,527 pitcher-seasons are below
  500 pitches (`low_sample`).

## Chris Sale career pitch-level data

`pitchiq aging-etl` also ingests Sale's pitch-level Statcast for `PITCHIQ_CAREER_SEASONS` (default 2015-2022) through
the existing ingestion pipeline (`ingest_statcast`) into `pitches`, `games`, `plate_appearances`, `pitch_sequences`:
16,015 pitches in 158 games. Rows per season and comparison with Savant's season pitch count (custom leaderboard):

| season | pitch rows | typed | Savant pitch_count | exact age range at game date |
|---|---|---|---|---|
| 2015 | 3,323 | 3,320 | 3,323 | 26.035 - 26.508 |
| 2016 | 3,431 | 3,359 | 3,431 | 27.015 - 27.510 |
| 2017 | 3,428 | 3,423 | 3,428 | 28.017 - 28.493 |
| 2018 | 2,525 | 2,524 | 2,525 | 28.997 - 29.493 |
| 2019 | 2,466 | 2,465 | 2,466 | 29.994 - 30.372 |
| 2020 | 0 (did not pitch: Tommy John surgery) | - | - | - |
| 2021 | 740 | 736 | 736 | 32.376 - 32.513 |
| 2022 | 102 | 102 | 102 | 33.285 - 33.298 |
| 2023 | 1,672 | 1,670 | 1,670 | 34.005 - 34.498 |
| 2024 | 2,818 | 2,815 | 2,815 | 35.004 - 35.475 |
| 2025 | 2,031 | 2,031 | 2,031 | 35.993 - 36.499 |
| 2026 | 2,575 | 2,574 | 2,574 | 36.992 - 37.485 |

(2023-2026 were already in `pitches` from the main ingest; they are listed for the age / count check.) Ages come from
the `pitch_ages` view using birth date 1989-03-30 from the MLB Stats API.

### Isolation from next-pitch modeling

`Settings.seasons` (2023-2026) remain the only modeling seasons; `Settings.career_seasons` are history-only
(`pitchiq/seasons.py`). Every path that feeds the next-pitch product filters `pitches` to modeling seasons: feature
frame (`features/store.load_pitch_frame`, used by history features and the simulator), ingestion validation and game
feeds, analytics and its filter options, pitcher list / summary counts, video discovery and coverage. The career ingest
refuses modeling seasons, restores `pitchers.seasons` to the modeling list and records
`pitchers.extra.history_only_seasons` = [2015, 2016, 2017, 2018, 2019, 2021, 2022]. After the run the main DB still has
`pre_pitch_states` train 4,490 (2023-2024), validation 2,031 (2025), test 2,575 (2026), i.e. unchanged and with no rows
before 2023. Tests assert that the feature frame and a full feature rebuild are identical before and after
history-only rows are added, that no pre-pitch states are created for them, and that analytics / video coverage
exclude them. `GET /api/aging/pitchers/{id}/career-pitches` labels each season `modeling` or `history_only`.

## Running

```
cd backend
alembic upgrade heads
pitchiq aging-etl                       # all PITCHIQ_AGING_SEASONS + Sale career seasons, cached responses
pitchiq aging-etl --seasons 2024 2025 --no-career --min-season-pitches 300
pitchiq aging-etl --refresh             # bypass the raw-response cache (424 leaderboard/search/people requests
                                        # at 1 req/s per host, plus Sale's monthly Statcast windows)
```

Job: `POST /api/jobs` with kind `aging_etl` (params: seasons, career_seasons, career, search_pitch_types, min_pitches,
min_season_pitches, refresh, refresh_bio, pitcher_id). API: `/api/aging/coverage`, `/api/aging/pitchers/{id}`,
`/api/aging/pitchers/{id}/career-pitches`, `/api/aging/cohorts?cohort_type=age|birth_year_band&scope=pooled|<season>`,
`/api/aging/observations?pitch_family=&metric=&pitcher_id=&both_qualified=`.

## Notes for downstream (aging models / forecasts)

* The tables contain every season through 2026. A forecast made "as of" a historical cutoff must use only rows with
  `season <= cutoff` (observations: `season_to <= cutoff`) and must not use pooled cohorts, which include all
  seasons; per-season cohorts (`scope = '<season_from>'`) can be filtered. Re-running the ETL never mixes in future
  seasons implicitly, but nothing in these tables is cut off by itself.
* 2026 next-pitch results are retrospective (previously viewed); this dataset does not change that status.

## Tests

`backend/tests/test_aging.py` (14 tests, offline): family / era / age helpers, KC -> CU folding, the full ETL on
fixtures cut from real responses (seasons 2019, 2020, 2022, 2023; Sale, Webb, Houck, Sabathia, Felix Hernandez,
Flaherty, Scherzer; one real Sale 2019 game for the pitch-level path), stored values equal source values, low-sample /
short-season flags, dropout rows for Sabathia and Hernandez after 2019, era-change flags, Houck's SL -> SL+ST
reclassification flag, cohort counts equal observation counts, idempotent rebuild, exact per-pitch ages, history-only
isolation (feature frame, full feature rebuild, analytics, video coverage, pre-pitch states), modeling seasons refused
by the career ingest, exact (untruncated) bulk-people provenance URL, job kind and API. Full suite on
`pitchiq_test_b`: 80 passed.

## Limitations

* Savant leaderboards are served as currently classified and computed; Savant can revise historical pitch
  classifications and leaderboard values. `fetched_at` records when each response was fetched; re-running with
  `--refresh` may change values.
* The TrackMan -> Hawk-Eye change (2019 -> 2020) and the 2020 short season are flagged, not corrected; spin and break
  levels across that boundary are not strictly comparable.
* Pitch-type speed / spin from different leaderboards disagree for a small number of 2015-2016 rows (see validation);
  the movement leaderboard value is used.
* Pitch-type extension, release position and whiff % per type come from grouped Statcast searches; values for a
  folded CU row (CU + KC + CS) are pitch-weighted means.
* Survival is defined by a pitch-count threshold (500), not by roster status, so a reliever who throws fewer pitches
  after a role change counts as "dropped" for the ALL unit; `returned_later` and the per-unit rows help separate this.
* Pitcher-seasons that appear only in the grouped search (not in the custom leaderboard) lack zone / chase / edge
  metrics.
* No MLflow run is logged for the ETL: it is data ingestion, not a model or an evaluation; its provenance lives in
  `aging_etl_runs`.
