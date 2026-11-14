# System B: pitch characteristic & aging forecast engine

Step key: `aging-model`. Code: `backend/pitchiq/aging_model/{panel,methods,backtest,analysis,pipeline}.py`, ORM
`backend/pitchiq/models_aging_forecast.py`, migration `aging03_forecast` (down_revision `vcv01_video_alignment`),
API `backend/pitchiq/api/routers/aging_forecast.py`, CLI `pitchiq aging-model`, job kind `aging_model`, tests
`backend/tests/test_aging_model.py` (fixtures in `backend/tests/fixtures/aging_model/`, cut from the real aging
dataset by `backend/tests/make_aging_model_fixtures.py`).

Every number below was read from the main database (`pitchiq`) after run **`aging_backtests.backtest_key =
ag2026-20261114014710-b47841`** (status `succeeded`, 22.9 min wall clock on 4 processes while another agent's job
shared the CPUs; 0 method failures). Input: aging ETL run 5 (`docs/aging-data.md`), panel fingerprint
`rows_hash 01fc7e26...` (329,421 pitcher-season-unit-metric rows, 181,647 qualified, 2,423 pitchers, 2015-2026). An
earlier run, `ag2026-20261114013241-567e32`, is also stored; it is the same engine before conformalized intervals
were added for the two native-quantile methods (see "Intervals"), and it is used below only for that comparison.
Its `aging_selected` model is `archived`; the current one is `aging-selected-c2026-20261114014710-b47841`
(`promoted`).

**Evaluation labels.** Development = target seasons 2018-2025 (method selection, Marcel tuning, interval
calibration). Target season 2026 = **retrospective, previously viewed**: it is scored and stored separately
(`retrospective_2026_previously_viewed`) and never used for selection, tuning or calibration.

## What is forecast

Unit of analysis: pitcher-season (season-level Savant leaderboards, `pitcher_season_stats` /
`pitcher_season_pitch_stats`, merged as described in `docs/aging-data.md`). Families are pitch-weighted means over
the official codes of a family (SL/ST/SV -> slider, so the 2023 sweeper relabelling does not create breaks).

| unit | targets |
|---|---|
| four_seam, sinker, slider, changeup (Sale's arsenal; the panel holds every family) | velocity `avg_speed_mph`, spin `avg_spin_rpm`, induced vertical break `ivb_in`, horizontal break magnitude `break_x_in` (Savant's `pitcher_break_x` is unsigned), release extension `extension_ft`, release height `release_pos_z_ft`, release side `release_pos_x_armside_ft` (mirrored so arm side is positive for both hands) |
| ALL (pitcher-season) | zone % `in_zone_pct`, chase % `oz_swing_pct`, whiff % `whiff_pct`, K % `k_pct`, BB % `bb_pct`, arm angle `arm_angle_deg` (Hawk-Eye, 2020+) |
| Sale only, from his pitch-level rows | location dispersion per family: SD of `plate_x`, SD of `plate_z` (ft) |

34 population targets + 8 location targets. Only **qualified** seasons enter the models: a family >= 100 pitches, a
pitcher-season >= 500 pitches (both scaled to 37 / 185 for the 60-game 2020 season). Position players are excluded.
Forecasts are therefore **conditional on the unit qualifying in the target season**; the engine does not forecast
whether Sale pitches, and it infers nothing about injuries.

Horizons: h = 1, 2, 3 seasons. **Rest of current season: not produced.** The cutoff season (2026) is complete
(today is 2026-11-14), and the data are season-level leaderboards; an in-season horizon would need partial-season
population data that this dataset does not contain. Every forecast row says so in `assumptions.rest_of_season`.

Sale's ages come from his MLB Stats API birth date (1989-03-30), age on June 30: 2027 38.251, 2028 39.253, 2029
40.253 (completed 38 / 39 / 40).

## Methods (`pitchiq/aging_model/methods.py`)

| method | description |
|---|---|
| `no_change` | y(t+h) = y(t) |
| `marcel` | last 3 seasons weighted 5/4/3 x sample size, regressed to the season-t population mean with k pseudo-pitches; k chosen from {0, 50, 150, 400, 1000, 3000} by h=1 pseudo-forecast MAE on pairs that end <= the tuning season (for the 2026 forecast: <= 2025). Chosen k: four-seam velocity 50, K % 1000, BB % 1000 |
| `age_curve` | population delta method: y(t) + sum of population mean yearly changes at each age, from consecutive-season pairs of survivors (same era, no 2020), restricted cubic spline in age (knots 23.5 / 27 / 30.5 / 36, linear tails), deltas winsorized at 1/99 % |
| `age_curve_ipw` | the same with inverse-probability-of-continuation weights (logistic model on season-t age spline, value standardized within season, log workload, starter share; fitted on t not in {2019, 2020}; weights clipped at p >= 0.05, trimmed at the 99th percentile, mean 1) |
| `mixedlm` | statsmodels MixedLM on levels: age spline + Hawk-Eye era + 2020 + log workload + starter share fixed effects; random intercept and random slope on (age-30)/10 per pitcher; forecast = fixed part at the projected age (season-t workload / role carried forward) + the pitcher's BLUPs |
| `lgbm_quantile` | LightGBM quantile regression (q = 0.1 / 0.5 / 0.9) of y(t+h) - y(t) on the history at t (value, previous changes, gap, sample size, workload, role, age, h, season-t population mean and the pitcher's distance to it, era flags incl. "crosses Hawk-Eye", target-2020 flag); fixed parameters (300 trees, lr 0.05, 15 leaves) |
| `eb_kalman` | empirical-Bayes local-level model: the pitcher's true level follows the IPW population age curve plus a random walk (variance q); an observed season adds season noise a and sampling noise b / n. a, b, q from a variogram of within-pitcher changes (lags 1-3, NNLS); prior = season population mean with between-pitcher variance; Kalman filter over the pitcher's qualified seasons, Gaussian predictive. A Hawk-Eye level shift is estimated by difference-in-differences (2019 -> 2021 vs same-era 2-year changes) when the cutoff is >= 2021 |

No method forces a decline: every age effect is estimated. For Sale's four-seam velocity the fitted EB components at
the 2026 cutoff are a = 0.095 mph^2, b = 25.8 mph^2 x pitches, q = 0.500 mph^2 per season, prior variance 4.86,
Hawk-Eye shift +0.15 mph; for K %: a = 1.00, b = 2,136, q = 4.41 (points^2).

### Separating age from workload, era, measurement and season noise

* **Season noise vs persistent change**: the EB variogram splits observed change into season noise (a), sampling
  noise (b / n) and persistent drift (q) per target (stored in `assumptions.method_state` of each `eb_kalman` row).
* **Workload / role**: MixedLM and LightGBM include season-t workload and starter share; the analysis below adds
  them to the delta model.
* **Era / measurement**: TrackMan (2015-2019) vs Hawk-Eye (2020+): pair-based estimates (delta curves, variogram,
  Marcel tuning) exclude pairs that cross 2019 -> 2020 or touch 2020; MixedLM has Hawk-Eye and 2020 fixed effects;
  LightGBM has era-crossing features; EB estimates the shift. The sweeper reclassification is absorbed by family
  aggregation. Selection uses only forecasts within one era (see below).
* **Calendar time**: see "Age vs calendar year".

## Backtest (`pitchiq/aging_model/backtest.py`)

Rolling origins c = 2016 ... 2025: each method is fitted on qualified rows with season <= c and forecasts every unit
that qualified in c for c+1, c+2, c+3. Scored: target seasons 2018-2026 where the unit qualified (1,273,589 scored
forecasts over 34 targets x 7 methods). Methods are compared on their common support (rows every method forecast).
Metrics per target / method / horizon: MAE and RMSE of the median, bias, coverage and width of the 80 % interval
(q10-q90), pinball loss at q10 / q50 / q90 and their mean (a 3-point CRPS approximation). Breakdowns stored in
`aging_backtest_metrics` (15,310 rows in this run): subsets `all_targets`, `dev`, `dev_same_era`,
`selection`, `retro_2026`; age band at the origin (<=26, 27-30, 31-33, 34-36, 37+) on `dev_same_era`; target season;
Sale's own rows.

**Intervals.** Methods without their own distribution (no_change, marcel, age_curve(_ipw), mixedlm) get empirical
intervals: the method's own residuals at earlier origins (origin c' with c' + h <= c, i.e. outcomes known at the
cutoff; same horizon; the unit's age band if it has >= 150 residuals, else all ages; Gaussian with the observed
h-season change SD before 30 residuals exist). The method's point stays the median. `lgbm_quantile` and
`eb_kalman` have their own quantiles; in the first run (`...013241-567e32`) their raw 80 % intervals undercovered
(selection subset, h = 1 / 2 / 3: LightGBM 0.665 / 0.614 / 0.598, EB 0.870 / 0.811 / 0.789), so the current run
conformalizes them (CQR: widen q10 / q90 by the 80 % quantile of max(q10 - y, y - q90) of earlier forecasts, same
outcome-known rule). Coverage after CQR: LightGBM 0.852 / 0.780 / 0.811, EB 0.895 / 0.840 / 0.865. CQR improved
LightGBM's relative pinball (h = 1: 1.031 -> 1.019) and fixed EB's long-horizon undercoverage, but EB's h = 1
pinball got worse (1.015 -> 1.042) because its intervals became wider than needed for the >= 32 age band.

For the 2026-cutoff forecasts, intervals use residuals of development targets only (<= 2025).

**Selection** (one method per target, used for all horizons so the 2027-2029 trajectory and Monte Carlo paths come
from one model): lowest mean over h = 1..3 of the method's mean pinball loss relative to `no_change`, on the
`selection` subset = development targets 2018-2025, excluding target season 2020 and forecasts that cross
TrackMan -> Hawk-Eye, origin age >= 32 (Sale's band). A horizon with < 100 rows in that subset falls back to
`dev_same_era` (all ages); this happened at h = 3 for every sinker, slider and changeup target and at h = 2 for
every changeup target.

### Results (development, selection subset; mean over the 34 targets of pinball relative to no_change)

| method | h=1 | h=2 | h=3 | coverage80 h=1 / 2 / 3 |
|---|---|---|---|---|
| no_change | 1.000 | 1.000 | 1.000 | 0.857 / 0.864 / 0.868 |
| marcel | 1.057 | 0.998 | 0.995 | 0.844 / 0.862 / 0.860 |
| age_curve | 1.043 | 1.073 | 1.094 | 0.847 / 0.850 / 0.872 |
| age_curve_ipw | 1.044 | 1.077 | 1.105 | 0.847 / 0.853 / 0.870 |
| mixedlm | 1.095 | 1.054 | 1.099 | 0.831 / 0.848 / 0.837 |
| lgbm_quantile (CQR) | 1.019 | 1.034 | 0.994 | 0.852 / 0.780 / 0.811 |
| eb_kalman (CQR) | 1.042 | 1.103 | 1.171 | 0.895 / 0.840 / 0.865 |

On average no single method beats no-change across characteristics: release point, extension, horizontal break,
spin and arm angle are best forecast by their last value, and population age curves add error there. The gains are
target-specific. Selected per target (count of 34): no_change 17, lgbm_quantile 5, marcel 4, eb_kalman 3,
mixedlm 2, age_curve_ipw 2, age_curve 1. Mean pinball skill vs no_change of the selected model (development
selection subset, `aging_model_registry.validation_metrics`): +2.0 % (h=1), +3.9 % (h=2), +5.0 % (h=3); per target:

| target | selected | rel. pinball (mean h) | h=1 MAE selected / no_change (n) | h=1 coverage80 |
|---|---|---|---|---|
| ALL k_pct | lgbm_quantile | 0.800 | 3.174 / 3.724 pts (463) | 0.827 |
| ALL whiff_pct | lgbm_quantile | 0.794 | 2.744 / 3.141 pts (463) | 0.834 |
| ALL bb_pct | marcel | 0.820 | 1.593 / 1.906 pts (463) | 0.810 |
| ALL oz_swing_pct | lgbm_quantile | 0.845 | 2.505 / 2.927 pts (463) | 0.834 |
| ALL in_zone_pct | mixedlm | 0.850 | 2.166 / 2.445 pts (463) | 0.786 |
| ALL arm_angle_deg | no_change | 1.000 | 2.341 deg (279) | 0.781 |
| four_seam avg_speed_mph | age_curve | 0.917 | 0.634 / 0.686 mph (411) | 0.869 |
| sinker avg_speed_mph | age_curve_ipw | 0.957 | 0.654 / 0.676 mph (303) | 0.851 |
| slider avg_speed_mph | eb_kalman | 0.959 | 0.956 / 0.994 mph (308) | 0.867 |
| changeup avg_speed_mph | eb_kalman | 0.989 | 0.730 / 0.739 mph (191) | 0.895 |
| four_seam ivb_in | mixedlm | 0.959 | 0.821 / 0.764 in (411) | 0.871 |
| slider ivb_in | marcel | 0.966 | 1.365 / 1.366 in (308) | 0.847 |
| changeup break_x_in | marcel | 0.955 | 0.934 / 0.905 in (191) | 0.906 |
| sinker avg_spin_rpm | marcel | 0.974 | 43.7 / 43.5 rpm (303) | 0.855 |
| changeup avg_spin_rpm | eb_kalman | 0.995 | 59.6 / 59.6 rpm (191) | 0.843 |
| slider avg_spin_rpm | lgbm_quantile | 0.993 | 61.1 / 55.6 rpm (308) | 0.883 |
| sinker release_pos_x_armside_ft | lgbm_quantile | 0.996 | 0.178 / 0.161 ft (303) | 0.845 |
| four_seam release_pos_x_armside_ft | age_curve_ipw | 0.999 | 0.160 / 0.158 ft (411) | 0.864 |
| all other 16 targets | no_change | 1.000 | e.g. four-seam spin 44.3 rpm, extension 0.086 ft, release height 0.081 ft | 0.81-0.93 |

The last four non-no-change selections win by less than 1 % relative pinball and have a worse median MAE than
no-change: they are practical ties (the rule has no parsimony margin; adding one would be a protocol change made
on development data, documented here as a known weakness). For K %, whiff %, BB %, chase and zone the population
models clearly beat no-change (15-21 % lower pinball), i.e. command / bat-missing rates regress substantially; for
velocity the age curve / EB methods beat no-change by 4-8 %, increasingly with horizon (four-seam h=3 MAE in the
first run: EB 0.928 vs no-change 1.222 mph).

Age bands (development, same era, h = 1, MAE). Four-seam velocity: no_change 0.769 / 0.719 / 0.682 / 0.752 / 0.576
mph for <=26 / 27-30 / 31-33 / 34-36 / 37+ (n 889 / 892 / 382 / 145 / 33); EB 0.750 / 0.704 / 0.628 / 0.661 / 0.491;
age_curve 0.768 / 0.711 / 0.645 / 0.682 / 0.524. K %: no_change 3.700 / 3.770 / 3.835 / 3.715 / 3.018 points; LightGBM
3.293 / 3.325 / 3.307 / 3.081 / 2.345 (n 773 / 854 / 410 / 157 / 44). The 37+ band has only 33-44 forecasts per
target at h = 1.

**Retrospective 2026 (previously viewed; all ages).** Mean relative pinball of the selected methods vs no-change:
0.980 / 0.983 / 0.966 (h = 1 / 2 / 3), mean coverage80 0.851 / 0.846 / 0.828. Examples (h=1): K % MAE 3.187 vs
3.623 (n 324), whiff % 2.374 vs 2.675, zone % 2.700 vs 3.604 (but coverage 0.679), four-seam velocity 0.647 vs
0.644 mph (no gain in 2026).

### Sale's own backtest (stored in `aging_backtest_predictions`, every method / origin / horizon)

Four-seam velocity, selected `age_curve`, h = 1 (origin value -> median [80 %] vs actual): 2017->2018 94.8 -> 94.62
[93.30, 95.75] vs 95.2; 2018->2019 95.2 -> 94.90 [93.60, 95.93] vs 93.4 (outside); 2023->2024 93.9 -> 93.59 [92.34,
94.68] vs 94.8 (outside); 2024->2025 94.8 -> 94.44 [93.27, 95.37] vs 94.8; retrospective 2025->2026 94.8 -> 94.46
[93.37, 95.43] vs 96.1 (outside). The population curve expected a decline in each year; Sale's velocity rose in
2024 and 2026. K %, selected `lgbm_quantile`, h = 1: 2023->2024 29.4 -> 27.67 [22.31, 30.83] vs 32.1 (outside),
2024->2025 32.1 -> 29.07 [24.92, 32.91] vs 32.4, retrospective 2025->2026 32.4 -> 28.93 [23.94, 32.45] vs 30.9.
Sale has 2-9 scored forecasts per target and horizon; his own backtest is illustrative, not a basis for selection.

## Sale forecasts (cutoff 2026; `pitch_characteristic_forecasts`, selected method, median [80 % interval])

| target | method | 2026 actual | 2027 (age 38) | 2028 (39) | 2029 (40) |
|---|---|---|---|---|---|
| four-seam velocity (mph) | age_curve | 96.1 | 95.77 [94.68, 96.74] | 95.44 [94.11, 97.16] | 95.10 [93.73, 97.23] |
| sinker velocity (mph) | age_curve_ipw | 95.6 | 95.23 [94.25, 96.35] | 94.83 [93.46, 96.49] | 94.40 [92.98, 96.50] |
| slider velocity (mph) | eb_kalman | 79.8 | 79.54 [78.01, 81.08] | 79.43 [77.27, 81.59] | 79.33 [76.69, 81.96] |
| changeup velocity (mph) | eb_kalman | 88.5 | 87.97 [86.49, 89.46] | 87.81 [85.76, 89.86] | 87.66 [85.18, 90.13] |
| four-seam spin (rpm) | no_change | 2351 | 2351 [2264, 2404] | 2351 [2252, 2465] | 2351 [2237, 2483] |
| slider spin (rpm) | lgbm_quantile | 2529 | 2521 [2415, 2603] | 2522 [2377, 2631] | 2514 [2344, 2659] |
| sinker spin (rpm) | marcel | 2211 | 2169 [2100, 2240] | 2169 [2079, 2277] | 2169 [2067, 2291] |
| changeup spin (rpm) | eb_kalman | 1859 | 1840 [1725, 1955] | 1820 [1669, 1972] | 1796 [1615, 1977] |
| four-seam IVB (in) | mixedlm | 9.9 | 9.30 [7.83, 10.59] | 8.90 [7.13, 10.38] | 8.50 [6.59, 10.14] |
| slider IVB (in) | marcel | -4.2 | -3.87 [-5.90, -1.31] | -3.87 [-7.06, -0.95] | -3.87 [-7.15, -0.38] |
| four-seam extension (ft) | no_change | 6.61 | 6.61 [6.48, 6.77] | 6.61 [6.40, 6.90] | 6.61 [6.43, 6.95] |
| four-seam release height (ft) | no_change | 4.94 | 4.94 [4.79, 5.04] | 4.94 [4.71, 5.12] | 4.94 [4.67, 5.13] |
| K % | lgbm_quantile | 30.9 | 29.12 [23.85, 32.15] | 27.61 [22.22, 31.66] | 26.46 [20.93, 31.67] |
| whiff % | lgbm_quantile | 30.4 | 28.18 [23.99, 31.96] | 27.50 [22.87, 32.25] | 26.99 [21.95, 32.30] |
| chase % | lgbm_quantile | 36.5 | 32.05 [28.77, 37.26] | 31.54 [27.73, 37.74] | 31.05 [27.06, 37.96] |
| zone % | mixedlm | 49.5 | 50.29 [47.43, 53.83] | 50.29 [46.58, 54.34] | 50.28 [46.64, 54.64] |
| BB % | marcel | 4.8 | 5.83 [3.71, 8.60] | 5.83 [3.11, 9.19] | 5.83 [2.92, 9.08] |
| arm angle (deg) | no_change | 10.9 | 10.90 [6.70, 14.03] | 10.90 [4.67, 15.03] | 10.90 [3.80, 15.90] |

All 786 forecast rows (126 selected: 34 targets + 8 location targets x 3 seasons; the rest are the other methods'
forecasts for comparison) store `quantiles` (q05-q95 where the method has them; LightGBM q10/q50/q90),
`distribution` (normal mean/sd, or piecewise-linear quantile function with Gaussian tails), 200 `sample_draws`
(seeded, `draws_seed`), `model_version`, `selection_model_version`, `projected_age` and `assumptions` (cutoff, data
used, tuning/calibration through 2025, selection basis and its development score, interval source, workload/role
carried forward, Hawk-Eye era, 162 games, no forced decline, no injury inference, sparse population support at
38+). Draws are marginal per season (no joint path model across seasons, except that one method generates all
three seasons).

**Location consistency (Sale only, own history).** Plate-location SD per family from his 2015-2026 pitch-level rows
(`location_panel`). No population pitch-level data exist, so there is no age curve; methods: no-change, Marcel-weighted
recent 3 seasons, career mean; Gaussian interval with sd = sqrt(h) x SD of his own one-season changes (development
targets). Selection per metric by development MAE (43 scored forecasts pooled over families and horizons):
`plate_x_sd_ft` no_change 0.0478 vs career mean 0.0492 vs recent3 0.0499 ft; `plate_z_sd_ft` no_change 0.0459 vs
0.0473 / 0.0465. Coverage of these intervals is poor for plate_z (development h=1 0.556, h=2 0.500); they are
small-sample, own-history numbers. 2027 forecasts (no_change): four-seam plate_x SD 0.73 [0.65, 0.81], plate_z SD
0.68 [0.63, 0.73]; slider 0.91 [0.83, 0.99] / 0.66 [0.61, 0.70].

## Survivor bias and age vs calendar year (`aging_analyses`, data through 2026; reported, not used for selection)

Cumulative population change from age 30 to 36 under each estimator (sum of yearly changes, or level difference):

| target | survivors (delta) | IPW (delta) | + season FE | + season FE + workload, IPW | MixedLM era FE | MixedLM season FE | cross-section 36 vs 30 | pairs |
|---|---|---|---|---|---|---|---|---|
| four-seam velocity (mph) | -1.64 | -1.39 | -1.66 | -1.10 | -1.78 | -2.34 | -1.50 | 3,001 |
| sinker velocity (mph) | -1.49 | -1.35 | -1.51 | -1.23 | -1.86 | -2.33 | -1.52 | 1,734 |
| slider velocity (mph) | -1.16 | -1.04 | -1.22 | -0.96 | -1.27 | -1.58 | -1.52 | 2,250 |
| four-seam spin (rpm) | +5.1 | +26.6 | +5.4 | +39.6 | -14.6 | -49.0 | -15.9 | 3,001 |
| slider spin (rpm) | +160.3 | +172.5 | +196.9 | +220.2 | +20.2 | -61.8 | +7.9 | 2,250 |
| K % (pts) | -5.69 | -3.84 | -5.59 | -2.99 | -3.30 | -3.21 | -1.89 | 2,874 |
| whiff % (pts) | -3.74 | -2.91 | -3.61 | -2.31 | -2.84 | -2.79 | -1.83 | 2,874 |
| BB % (pts) | +1.95 | +1.37 | +1.98 | +1.14 | -0.28 | -0.32 | -0.85 | 2,874 |
| zone % (pts) | -2.60 | -2.11 | -2.46 | -1.93 | -0.35 | -0.33 | +0.10 | 2,874 |

**Survivor bias.** Continuation (qualifying again next season, t not in 2019/2020) falls with age: four-seam unit
0.698 (<=26), 0.678 (27-30), 0.664 (31-33), 0.618 (34-36), 0.510 (37+, 52 of 102); K %: 0.723 / 0.673 / 0.703 /
0.593 / 0.521 (62 of 119). It depends on the season-t value: pitchers in the lowest K % tercile continue at 0.568,
the highest at 0.778; four-seam velocity 0.605 vs 0.755. IPW corrects that observable selection. Because low-value
pitchers who do continue tend to move back toward the mean, upweighting them makes the curves *less* steep (K %
30-36 from -5.69 to -3.84 points; four-seam velocity -1.64 to -1.39 mph). IPW cannot correct selection on the
unobserved change itself (a pitcher who lost velocity and therefore lost his job). The returners diagnostic
bounds that only weakly: pitchers aged >= 32 who missed a season and qualified again within 3 seasons changed
four-seam velocity by -0.29 mph per year (n 22) vs -0.30 for survivors (n 520); slider velocity -0.31 (n 17) vs
-0.15 (n 397); K % -0.46 (n 30) vs -1.08 points (n 577). In the backtest, IPW did not help: `age_curve_ipw` is within
0.01 of `age_curve` in relative pinball at every horizon (1.044 / 1.077 / 1.105 vs 1.043 / 1.073 / 1.094).

**Age vs calendar year.** In consecutive-season changes, cohort (birth year) cancels, and season effects are
identified across birth cohorts because pitchers of many ages share each season. But age = season - birth year, so
the *average* yearly change is the same constant whether it is called "aging" or "calendar trend": first
differences cannot separate a linear calendar trend from the mean age effect. The season-FE delta model therefore
codes target-season effects as deviations summing to zero, which attributes the mean change to age (an assumption,
not a finding). The deviations are real findings: four-seam velocity season effects are within -0.22 ... +0.13
mph/year (no strong calendar shocks), K % +0.37 ... +0.60 points in 2016-2019 transitions and -0.86 for 2021->2022.
For slider spin the 2015->2016 transition carries +166 rpm, i.e. a measurement change in 2015 Statcast spin
data; it dominates the slider-spin delta curve (+160 rpm from 30 to 36), which is not aging. The level MixedLM with
season fixed effects, which leans on the assumption that pitcher random effects are uncorrelated with age, gives
the opposite sign (-62 rpm) and a steady league-wide level increase of +229 (2016) to +409 rpm (2026) relative to
2015. Four-seam velocity level season effects rise steadily (+0.06 in 2016 to +1.54 mph in 2026 vs 2015), which in
the season-FE level model moves part of the decline from calendar to age (-2.34 vs -1.78 mph from 30 to 36). In
short: the curvature of aging and the calendar deviations are identified; the split of the average trend between
age and calendar time is an assumption, and the selected forecasts use the per-target method the backtest prefers
(for spin that is no-change, which embeds no age or calendar trend). Workload: higher season-t workload predicts a
more negative next-season change (log workload coefficient -0.23 mph for four-seam velocity, -1.36 points for K %),
starter share a more positive one (+0.39 mph, +1.21 points).

## Storage, registry, MLflow

* `aging_backtests` (run, config, selection per target and horizon, summary, failures, panel fingerprint).
* `aging_model_registry`: one row per method (`aging_<method>`, candidate), `aging_location_own_history`, and
  `aging_selected` (promoted; config holds the per-target selection and member versions). Artifacts (joblib, sha256
  checked) under `backend/artifacts/aging/<backtest_key>/`: fitted state at the cutoff per target (curve
  coefficients, EB components, Marcel k and its grid scores, MixedLM fixed effects / RE covariance, LightGBM boosters
  as text). These are separate from `model_registry` on purpose: next-pitch code resolves models per pitcher there
  (the end-to-end test asserts no `aging%` family appears in it).
* `aging_backtest_runs` (shape of `evaluation_runs`): per method `development` (selection subset),
  `development_same_era_all_ages` and `retrospective_2026_previously_viewed`, plus the location backtest; 23 rows in
  this run (ids 47-69). `aging_backtest_metrics` (every breakdown), `aging_backtest_predictions` (Sale's rows),
  `aging_analyses` (34 rows), `pitch_characteristic_forecasts` (786 rows), 5,688 rows of Sale's backtest forecasts.
* MLflow: the existing `pitchiq.ml.tracking.Tracker` logs these rows unchanged into experiment
  **`pitchiq-aging-519242`** (prefix `<PITCHIQ_MLFLOW_EXPERIMENT_PREFIX>-aging`, so `pitchiq mlflow-backfill` of the
  next-pitch experiments never touches them): 9 training runs (with artifacts; the promoted selected model is
  registered with alias `promoted`) and 23 evaluation runs per engine run, tagged `pitchiq.system=aging_forecast` and
  `pitchiq.split_policy=rolling origin ...`. Run ids are written back to `config.mlflow_run_id` /
  `metrics.mlflow_run_id`. Store (`backend/mlruns/mlflow.db`) after the two engine runs: 64 aging runs (18 training,
  46 evaluation); registered model `pitchiq-aging-519242` version 2 (= `aging-selected-c2026-20261114014710-b47841`)
  carries alias `promoted`.

## API / CLI / job

* `GET /api/pitchers/{pid}/aging/history[?cutoff=]`: every target's season values (qualified or not), n, exact age,
  plus location dispersion from pitch-level rows.
* `GET /api/pitchers/{pid}/aging/forecast[?cutoff=&backtest_key=&all_methods=&include_draws=&unit=&metric=]`:
  latest successful run's selected (or all) forecast rows with quantiles, distribution, assumptions, optional draws.
* `GET /api/pitchers/{pid}/aging/backtest[?unit=&metric=&breakdown_type=subset|age_band_dev_same_era|target_season|
  target_pitcher|...|all&include_predictions=&include_analyses=]`: selection, metrics, model entries, evaluation
  runs (with MLflow ids), Sale's backtest rows, analyses.
* `pitchiq aging-model [--families ...] [--cutoff 2026] [--origins ...] [--methods ...] [--workers 4]
  [--no-analyses] [--no-mirror]`; job `POST /api/jobs` kind `aging_model` (same params). Re-running the engine
  creates a new `backtest_key`; the API serves the latest succeeded run (or `backtest_key=`).
  The uvicorn process already running on :8000 does not auto-reload; the endpoints are served after a restart
  (verified here with FastAPI's TestClient against the main database).

## Tests (`backend/tests/test_aging_model.py`, offline, real-data fixtures)

20 tests: panel aggregation / normalisation (family pitch-weighted means, arm-side mirroring, 2020 scaling);
restricted cubic spline linear tails; pinball / coverage arithmetic; for every method, forecasts from origin 2021
are identical when all later seasons are changed by +7; full backtest: perturbing 2026 values leaves development
metrics, the selection, Marcel's k (tuned through origin 2024) and the final forecast's empirical and CQR interval
widths unchanged while retrospective metrics change; perturbing seasons >= 2023 leaves Sale's backtest forecasts
(points and intervals) from origins <= 2022 unchanged; selection subset excludes 2020 / cross-era / young / 2026
rows; EB shrinkage (one small-sample season is pulled toward the population, a long large-sample history is not);
IPW weights (survivors only, mean 1); reproducible Monte Carlo draws with correct quantiles; location own-history
backtest; end-to-end on the test database (real rows for Sale + 119 pitchers inserted in the rolled-back session):
forecasts 2027-2029 with exact projected ages, draws, assumptions, registry entries with MLflow run ids, promoted
selected model, development / retrospective evaluation rows, analyses, no rows in `model_registry`; MLflow
experiment `pitchiq-aging-519242` with training and evaluation runs and the aging split policy tag; the three API
endpoints (with `get_db` overridden to the test session) and a 404 for an unknown pitcher; job kind registered.

## Limitations

* Forecasts are conditional on qualifying (>= 100 pitches of the family / >= 500 pitches); scored backtest rows are
  survivors by construction. The engine does not model availability or injuries.
* Population support at Sale's forecast ages is thin: 39 / 16 / 13 qualifying pitcher-seasons at completed ages
  38 / 39 / 40 in the whole dataset; the 37+ backtest band has 33-44 h = 1 forecasts per target. The 2028-2029
  forecasts lean on the linear tail of the age spline and on Sale's own history.
* Several selections are practical ties with no-change (relative pinball 0.99-1.00, worse median MAE): slider spin
  (lgbm), sinker and four-seam release side, changeup spin. Their medians are within a few rpm / hundredths of a foot
  of no-change but their intervals differ.
* CQR made EB intervals too wide at h = 1 for the >= 32 band (coverage 0.895); zone % intervals undercovered in the
  2026 retrospective (0.679). Interval calibration is pooled over ages for horizons where the 34+ band has < 150
  residuals.
* Sale's 2015-2016 fastball is labelled SI (sinker) by Statcast (1,739 / 1,860 pitches) and FF only from 2017, so his
  sinker family has a classification break; his four-seam series starts in 2017. Not corrected (no source for a
  re-label); the 2026-cutoff forecasts use only his recent seasons in practice (Marcel: last 3; Kalman: filtered).
* 2015 Statcast spin (TrackMan) appears inconsistent with 2016+ (slider spin season effect +166 rpm for 2015->2016);
  pair-based estimates include 2015->2016 pairs. The backtest selects no-change / LightGBM for spin targets.
* Mixing of method by target means the forecast trajectory is coherent within a target but different targets of the
  same pitch can come from different models (e.g. four-seam velocity from the age curve, IVB from MixedLM); draws
  are not jointly simulated across targets.
* Location dispersion is the raw SD of plate location (includes intended location changes), own history only, 2-9
  seasons per family; its intervals are approximate and undercover for plate_z.
* Rest-of-season forecasts are not produced (season complete; no partial-season population data).
* The earlier run `ag2026-20261114013241-567e32` (before CQR) remains stored and mirrored for the interval
  comparison; its selected model is archived.
