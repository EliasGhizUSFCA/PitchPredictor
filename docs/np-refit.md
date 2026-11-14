# np-refit: next-pitch training protocol, final refit, scheduled retraining, promotion gates, rollback

Every number below comes from runs on the main database (`pitchiq`, Chris Sale 519242, 9,096 Statcast rows, 9,090
labeled pre-pitch states, 97 games, 2023-2026). The runs are stored in `evaluation_runs`, `model_registry`,
`model_lifecycle` and `model_promotion_events`, and mirrored to MLflow (`backend/mlruns`, experiment `pitchiq-519242`).

## 1. Evaluation roles

`pre_pitch_states.split` still holds `train` (2023-24), `validation` (2025) and `test` (2026). On top of that,
`Settings` defines an evaluation role for each season (`pitchiq/ml/protocol.py`):

| role | seasons | allowed use | label in API / docs |
|---|---|---|---|
| `development` | 2023, 2024, 2025 | model selection, hyperparameters, calibration decision, promotion gates, retraining-policy choice | "development (chronological, ...)" |
| `retrospective_viewed` | 2026 | reporting only. 2026 was the "locked test" season and has been viewed twice | "retrospective (previously viewed)" |
| `prospective` | later seasons, plus the prospective ledger | honest future evaluation | "prospective" |

`GET /api/protocol` (and `pitchiq protocol`) returns the roles, how each split maps to a role, the rolling protocol
and the gate thresholds. `/api/pitchers/{pid}/evaluations`, `/api/evaluations/{id}` and the leaderboard now include
`evaluation_role` and `evaluation_label` fields. No existing field was removed.

## 2. Rolling-origin development evaluation (the selection objective)

* The folds cover the games of 2024 and 2025 (`dev_eval_seasons`), cut into consecutive blocks of 5 whole games
  (`dev_block_games`). That gives 10 folds and 4,846 scored pitches.
* Each fold's model is trained on every labeled pitch from games dated strictly before the block's first game. The
  window expands, 2023 is always in training, and no game is ever split across train and score. A fold is skipped
  if its training window is missing any pitch class. No fold was skipped on the real data.
* Primary metric: pooled out-of-sample multiclass log loss. Also stored: Brier, top-1, top-2, top-label ECE,
  per-class precision/recall and per-fold metrics.
* Grids (`DEV_GRIDS`):
  * frequency: alpha in {5, 20, 50} x 2 back-off level sets
  * logreg: C in {0.01, 0.03, 0.1, 0.3, 1}
  * lightgbm: num_leaves in {7, 15, 31} x min_child_samples in {20, 60, 120}, with boosting rounds 25..800 in steps
    of 25. Each fold is fit once with 800 rounds and scored at every checkpoint. A test checks this is identical to
    refitting with that many rounds. There is no early stopping on scored data.
* Calibration is chronological. Fold f is calibrated with a temperature fit only on the out-of-sample predictions of
  folds before f (fold 0 uses T=1). **Calibration rule, fixed before the final fit:** the pooled temperature is
  deployed only if chronologically calibrated dev log loss is below raw dev log loss. Otherwise the deployed model
  outputs raw probabilities (T=1). On the real data, calibration made things worse for every family, so every
  final-fit model deploys raw outputs. This matches the earlier finding that the T fit on 2025 hurt 2026.

### Results per family (selected configuration, pooled over 10 folds, 4,846 pitches)

These are `evaluation_runs` kind `rolling_dev_evaluation`, split `development`.

| family | run | selected hyperparameters | log loss (deployed = raw) | chrono-calibrated log loss | Brier | top-1 | top-2 | ECE |
|---|---|---|---|---|---|---|---|---|
| lightgbm | 29 | num_leaves 15, min_child_samples 120, n_estimators 125 | **1.0713** | 1.0746 | 0.6051 | 0.4998 | 0.8293 | 0.0121 |
| logreg | 28 | C 0.01 | 1.0849 | 1.0911 | 0.6144 | 0.4781 | 0.8298 | 0.0257 |
| frequency | 27 | alpha 50, levels stand / stand+count / stand+count+prev1 | 1.0900 | 1.0937 | 0.6161 | 0.4726 | 0.8283 | 0.0248 |
| majority | 26 | - | 1.1445 | 1.1451 | 0.6418 | 0.4016 | 0.8289 | 0.0112 |

Per-class recall (lightgbm): FF 0.533, SL 0.663, CH 0.018, SI 0.000. For logreg: CH 0.043, SI 0.011. The models still
almost never put the most probability on a changeup or sinker.

Per-fold raw log loss for lightgbm (run 29):

| fold | games through | n_train | log loss |
|---|---|---|---|
| 0 | 2024-04-26 | 1,670 | 1.175 |
| 1 | 2024-05-26 | 2,135 | 1.232 |
| 2 | 2024-06-27 | 2,630 | 1.226 |
| 3 | 2024-07-31 | 3,108 | 1.159 |
| 4 | 2024-08-28 | 3,584 | 1.049 |
| 5 | 2025-03-27 | 4,090 | 1.127 |
| 6 | 2025-04-25 | 4,569 | 1.013 |
| 7 | 2025-05-23 | 5,042 | 0.896 |
| 8 | 2025-08-30 | 5,547 | 0.859 |
| 9 | 2025-09-28 | 6,049 | 0.994 |

The early-2024 folds have small training windows (1.7k-2.6k pitches). Only fold 0 is trained on 2023 alone; folds 1
and later already include the 2024 games before their block (fold 1, the worst at 1.232, has 2023 plus five 2024
games). These early folds pull the pooled rolling dev log loss above the single-season 2025 numbers.

The selected configuration's score is the minimum over its grid, so it is optimistic by the selection effect.

For comparison, the legacy season-split production recipe (`519242-lightgbm-v1-20261113T205951`) was scored on the
same folds (run 19): raw 1.0772. Its stored headline, 1.0827, is chronologically calibrated, because that run used the
earlier (WIP) code with calibration always on.

## 3. Final fit (deployable models)

`pitchiq refit` (job `refit_next_pitch`) does three things:

1. Selects each family's recipe on the rolling dev folds.
2. Refits that recipe on **all labeled pitches from games dated before the deployment cutoff**. Today that is all of
   2023-2026: 9,090 pitches, 97 games, data cutoff 2026-09-23. **This fixes "the final model never learned 2025"**
   (nor 2026).
3. Takes the temperature only from the out-of-sample dev predictions, never from the final training data, and
   applies the calibration rule above.

Each model stores the following in `model_lifecycle` and `model_registry.config/calibration/validation_metrics`:

* data cutoff and deployment cutoff
* n_train and n_train_games
* selected hyperparameters
* dev metrics
* the dev evaluation run id

The same values go to MLflow as tags `pitchiq.data_cutoff`, `pitchiq.training_scope` and
`pitchiq.deployment_cutoff`, plus metrics `n_train` and `dev_*`. The generic mirror tags every model run with the
season-split `pitchiq.split_policy` ("train=[2023, 2024] validation=[2025] test=[2026]"); for lifecycle models it is
overwritten with the actual scope (e.g. "deployment_refit: fit on all labeled games before 2026-11-14 (data cutoff
2026-09-23, seasons [2023, 2024, 2025, 2026]); hyperparameters selected on rolling-origin development folds
[2024, 2025]"), and a `pitchiq.evaluation_roles` tag (development 2023-2025, retrospective_viewed 2026) is added. The
rolling-dev and replay evaluation runs get the same two tags. The 8 existing lifecycle model runs and 21 rolling-dev /
replay evaluation runs were re-tagged in place.

| family | model version | status |
|---|---|---|
| lightgbm | `519242-lightgbm-v1-20261114T001434-refit` | **production** (stored status `promoted`) |
| logreg | `519242-logreg-v1-20261114T000935-refit` | candidate |
| frequency | `519242-frequency-v1-20261113T235849-refit` | candidate |
| majority | `519242-majority-v1-20261113T235818-refit` | candidate |

The previous WIP run of this step created the `...20261113T22xxxx-refit` versions. Their recipes are identical (the
raw numbers reproduced exactly), but they predate the calibration rule: they apply the pooled temperature, e.g. T=0.943
for lightgbm. They are kept. `519242-lightgbm-v1-20261113T221307-refit` is now `previous_production`, the rollback
target. The WIP evaluation runs 15-25 are kept and superseded by runs 26-35.

Refit models are trained on 2026, so they cannot have an honest 2026 score:

* `holdout_benchmark` refuses to score a lifecycle model on a split at or before its data cutoff.
* `registry.latest_by_family` (benchmarks, walk-forward, ablation) excludes lifecycle models.
* Historical/blind views (challenge, `predict_historical`) default to the newest model whose data ended before the
  pitch's game. `predict_historical` reports `training_data_includes_game`. When no such model exists (challenges on
  the train split 2023-24 or the validation split 2025: no model's data ended before them), the default falls back to
  production, i.e. the refit trained on those very pitches. The challenge therefore reports this: each attempt's
  `prediction` carries `training_data_includes_game`, `evaluation_sample` ("in-sample (model trained on these
  pitches)" / "partly in-sample" / "out-of-sample") and `model_data_cutoff`, and `challenge_stats` adds
  `model_trained_on_these_pitches`, `in_sample_pitches`, `evaluation_sample` and an in-sample warning in `note`. On
  the real DB: train and validation challenges default to `...001434-refit` (in-sample); test challenges default to
  the legacy `519242-lightgbm-v1-20261113T205951` (data cutoff 2025-09-28, out-of-sample).
* `POST /api/pitchers/{pid}/models/compare` uses the newest model per family including refits, always includes the
  production model, and adds `training_scope`, `data_cutoff` and (for a historical pitch) `training_data_includes_game`
  to each row. On the real DB it now returns the four refits (all in-sample for any 2023-2026 pitch).
* Simulator support/OOD diagnostics (`ood_report`) and the unknown-previous-pitch marginalization weights use the
  model's actual fit set: for lifecycle models every labeled pitch from games before the deployment cutoff, for
  season-split models the train split. For production `...001434-refit`, `training_rows` is now 9,090 (it reported
  the 4,485 train-split rows before); the response adds `training_support`.

## 4. Promotion gates, rollback, MLflow aliases

Stages: `candidate` -> production (stored as `promoted`, so existing consumers keep working; API views add `stage`)
-> `previous_production` (one rollback slot) -> `candidate`. Models are never deleted (archive policy: keep).

The gates were fixed in `Settings` before any gated promotion. They read development evidence only.

| gate | rule |
|---|---|
| `artifact_checksum` | the artifact loads and its SHA-256 matches |
| `probabilities_normalized` | on all 6,516 labeled development pitches: finite, in [0,1], each row sums to 1 within 1e-6 |
| `min_training_samples` | at least 2,000 |
| `dev_ece` | dev top-label ECE of the deployed output at most 0.05 (raw ECE when calibration_applied=false, else chronologically calibrated ECE) |
| `dev_log_loss_vs_production` | candidate dev log loss at most production's + 0.005, on the same folds |
| `data_not_older_than_production` | the candidate's data cutoff is not before production's |
| `development_only_provenance` | dev folds and calibration evidence lie inside 2023-2025 |

A failing candidate stays `candidate`. Its gate results are written to `model_lifecycle.gate_results`, the
`model_registry.config.gate_results` summary and a `gate_failed` event in `model_promotion_events`.

* `POST /api/models/{v}/promote` (and `pitchiq promote`) is gated. Failing gates return 409 with the gate results.
  `force=true` bypasses them and is recorded as a manual override.
* `POST /api/models/{v}/gates` and `pitchiq gates` evaluate and record the gates without changing status.
* `POST /api/models/{v}/rollback` and `pitchiq rollback` work only on the production model. previous_production
  becomes production and the rolled-back model returns to candidate.
* MLflow aliases `production` and `previous_production` (plus the existing `promoted` alias) move with every
  promotion and rollback. After this run: `production` = v3 = `...20261114T001434-refit` and `previous_production` =
  v2 = `...20261113T221307-refit`, checked against the MLflow registry.

Real promotion, event in `model_promotion_events`: lightgbm `...001434-refit` passed all 7 gates:

* dev ECE 0.0121 (raw ECE: the deployed output is raw because calibration_applied=false; the chronologically
  calibrated ECE is 0.0325). The detail text stored with that promotion (event 5) wrongly described the value as
  "chronologically calibrated"; the wording was fixed and the gates were re-evaluated with `check_gates` (event 6,
  action `gates_evaluated`, all 7 passed, same values, no status change), which also replaced
  `model_lifecycle.gate_results`. Event 5 is kept unchanged as the historical record.
* dev log loss 1.0713, against the limit 1.0796 = production 1.0746 + 0.005
* 9,090 training samples
* max row-sum error 3.3e-16

`refit_all` tries families in ascending dev log loss and promotes the first one that passes. Here that was lightgbm,
so the other three were not attempted.

`train_all`'s existing auto-promotion only runs when no production model exists (bootstrap). Every later promotion
goes through the gates.

## 5. Scheduled retraining

Job `retrain_next_pitch` (params `policy`, `force`, `reselect`, `promote`) and CLI `pitchiq retrain --policy ...`:

| policy | a refit is due when |
|---|---|
| `after_each_start` | at least 1 completed start after production's data cutoff and before today |
| `every_6_starts` | at least 6 such starts |
| `weekly` | at least 7 days since the last fit and at least 1 new start |
| `manual` | the operator runs it and at least 1 new start exists (`--force` overrides) |

A retrain refits production's frozen recipe on all data before today. `--reselect` re-runs the rolling dev selection
first. The new model goes through the same gates. Every rebuild of the ingestion features (job `build_features`,
CLI `features`) then calls `refit.after_ingest`:

1. It writes the prospective ledger first.
2. It then runs the policy check with `Settings.auto_retrain_policy`, default `after_each_start`, chosen on the 2025
   replay below.

`manual` disables the automatic check. A season-split production model is never re-selected implicitly. On the real
DB, `pitchiq retrain --policy after_each_start` returned `due: false` (0 completed starts after 2026-09-23).

### Policy replay (evaluation_runs kind `retraining_policy_replay`)

The replay (`ml/replay.py`) simulates the live rule above, `refit.policy_due`, day by day and continuously across
2025 and 2026:

* At the first 2025 game (2025-03-27) a model is deployed, fit on all 4,485 earlier pitches (2023-24). This
  `initial_fit` is not counted as a refit.
* On every calendar day after that, before that day's game, the policy is checked exactly as live: new completed
  starts are the games after the deployed model's data cutoff and before that day; the last-fit time is the day of
  the last (re)fit; a due refit trains on all games dated before that day. So `weekly` is "at least 7 days since the
  last fit and at least 1 new start" (not calendar Mondays), `every_6_starts` counts starts since the deployed cutoff
  across the offseason, and no policy gets a free season-start refit.
* `manual`: live, the job runs only when an operator triggers it. The replay assumes no operator trigger, so the
  2025-03-27 deployment serves both seasons. This is the "never retrain" lower bound, not a model of a particular
  operator's habits.
* `refits` counts refit events between the previous replayed season's last game (exclusive) and this season's last
  game, so the 2026 count includes offseason refits triggered by late-2025 starts. `models` is the number of distinct
  models that served the season's games.

Each fit uses the family's dev-selected recipe and raw outputs (the deployed outputs, given the calibration rule).
The 95% CI is a game-cluster bootstrap with 300 resamples. The simulation rule is stored with each run
(`metrics.simulation`, `dataset_definition.simulation`).

An earlier version of the replay (runs 30-35) did not follow the live rule: it refit `weekly` on calendar Mondays and
gave every policy, including `manual`, an uncounted fit at each season start. Those runs are kept but superseded by
runs 36-41 below. The 2025 after_each_start, every_6_starts and manual numbers are unchanged (their 2025 schedules
were the same); weekly 2025 and all of 2026 changed, most of all 2026 manual (1.0265 before, 1.0640 now, because it
no longer gets a 2026 season-start fit).

**2025, development.** 21 games, 2,031 pitches. Runs 36 (frequency), 38 (logreg), 40 (lightgbm).

| family | policy | refits | models | log loss | 95% CI | Brier | top-1 | top-2 | ECE |
|---|---|---|---|---|---|---|---|---|---|
| lightgbm | after_each_start | 20 | 21 | **0.9409** | 0.889-1.003 | 0.5474 | 0.5623 | 0.8872 | 0.0357 |
| lightgbm | weekly | 17 | 18 | 0.9424 | 0.891-1.004 | 0.5479 | 0.5657 | 0.8882 | 0.0384 |
| lightgbm | every_6_starts | 3 | 4 | 0.9548 | 0.911-1.011 | 0.5531 | 0.5579 | 0.8872 | 0.0436 |
| lightgbm | manual | 0 | 1 | 0.9719 | 0.937-1.010 | 0.5620 | 0.5559 | 0.8897 | 0.0681 |
| logreg | after_each_start | 20 | 21 | 0.9576 | 0.906-1.013 | 0.5620 | 0.5411 | 0.8922 | 0.0245 |
| logreg | weekly | 17 | 18 | 0.9585 | 0.908-1.014 | 0.5622 | 0.5396 | 0.8912 | 0.0250 |
| logreg | every_6_starts | 3 | 4 | 0.9669 | 0.920-1.020 | 0.5650 | 0.5347 | 0.8932 | 0.0318 |
| logreg | manual | 0 | 1 | 0.9780 | 0.935-1.025 | 0.5681 | 0.5332 | 0.8927 | 0.0258 |
| frequency | after_each_start | 20 | 21 | 0.9763 | 0.935-1.025 | 0.5659 | 0.5367 | 0.8897 | 0.0574 |
| frequency | weekly | 17 | 18 | 0.9767 | 0.935-1.025 | 0.5661 | 0.5342 | 0.8897 | 0.0551 |
| frequency | every_6_starts | 3 | 4 | 0.9809 | 0.941-1.028 | 0.5680 | 0.5352 | 0.8897 | 0.0588 |
| frequency | manual | 0 | 1 | 0.9905 | 0.953-1.035 | 0.5728 | 0.5185 | 0.8897 | 0.0520 |

lightgbm 2025 refit days: weekly 2025-04-03 ... 2025-06-19 every 7 days, then 2025-08-31 ... 2025-09-28 (no starts
in between); every_6_starts 2025-04-26, 2025-05-30, 2025-09-11.

**2026, retrospective (previously viewed).** 27 games, 2,574 pitches. Runs 37, 39, 41 (state carried over from the
2025 replay). Not used for any choice.

| family | policy | refits | models | log loss | 95% CI | Brier | top-1 | top-2 | ECE |
|---|---|---|---|---|---|---|---|---|---|
| lightgbm | after_each_start | 27 | 27 | **0.9968** | 0.966-1.023 | 0.5852 | 0.5012 | 0.8380 | 0.0243 |
| lightgbm | weekly | 26 | 26 | 0.9974 | 0.967-1.024 | 0.5855 | 0.5031 | 0.8384 | 0.0219 |
| lightgbm | every_6_starts | 4 | 5 | 1.0040 | 0.974-1.032 | 0.5882 | 0.5023 | 0.8353 | 0.0179 |
| lightgbm | manual | 0 | 1 | 1.0640 | 1.038-1.087 | 0.6130 | 0.4701 | 0.8182 | 0.0209 |
| logreg | after_each_start | 27 | 27 | 1.0213 | 0.989-1.049 | 0.5985 | 0.4829 | 0.8322 | 0.0335 |
| logreg | weekly | 26 | 26 | 1.0215 | 0.989-1.049 | 0.5985 | 0.4825 | 0.8318 | 0.0340 |
| logreg | every_6_starts | 4 | 5 | 1.0253 | 0.991-1.054 | 0.6000 | 0.4837 | 0.8302 | 0.0352 |
| logreg | manual | 0 | 1 | 1.0588 | 1.029-1.085 | 0.6100 | 0.4841 | 0.8283 | 0.0225 |
| frequency | after_each_start | 27 | 27 | 1.0170 | 0.988-1.042 | 0.5993 | 0.4790 | 0.8318 | 0.0134 |
| frequency | weekly | 26 | 26 | 1.0171 | 0.988-1.042 | 0.5993 | 0.4794 | 0.8318 | 0.0124 |
| frequency | every_6_starts | 4 | 5 | 1.0197 | 0.991-1.044 | 0.6005 | 0.4786 | 0.8310 | 0.0138 |
| frequency | manual | 0 | 1 | 1.0507 | 1.027-1.072 | 0.6134 | 0.4596 | 0.8205 | 0.0156 |

lightgbm 2026: after_each_start's 27 refits include the one on 2025-09-29 after the last 2025 start; weekly's first
is 2025-10-05; every_6_starts serves the first 2026 games with its 2025-09-11 model and refits on 2026-04-07 (the
6th start since its 2025-09-10 data cutoff: three late-2025 starts, 2025-09-16/22/28, plus the first three of 2026), 2026-05-15, 2026-06-29 and 2026-08-09.

What the replay shows:

* In 2025, refitting during the season beats never refitting for every family. For lightgbm, after_each_start is
  0.031 lower in log loss than manual.
* after_each_start and weekly are practically tied (difference 0.0015 or less in 2025, 0.0006 or less in 2026), far
  inside the bootstrap CIs. Weekly refits a few days later but slightly less often.
* every_6_starts sits in between.
* Not retraining at all (manual without an operator trigger) costs the most in 2026 (lightgbm 1.0640 vs 0.9968),
  because the 2025-03-27 model never sees 2025 or 2026.
* The 2026 retrospective numbers show the same ordering. The lightgbm after_each_start value of 0.9968 is consistent
  with the earlier walk-forward result of 1.001.
* `Settings.auto_retrain_policy = "after_each_start"` was chosen from 2025 (development) only, where it is lowest for
  all three families under the live-rule replay. The margin over weekly is far inside the bootstrap CIs.

## 6. Prospective ledger

When features are rebuilt after ingestion, `prospective.record_prospective` scores every pitch of every game with the
model that was production on the game's date. It does this only if that model was created before the game and its
data cutoff is before the game date. Production timing comes from promotion events, which use wall-clock timestamps.

* Predictions go to `pitch_predictions` with context `prospective`. They are idempotent and are never back-filled with
  a later model.
* Metrics are at `GET /api/pitchers/{pid}/prospective`, CLI `pitchiq prospective`, and job `prospective_ledger`.
  Each model's rows are scored on that model's own classes. A label outside them (e.g. a new pitch type that feature
  rebuilds map to a class the model lacks) is stored with `prob_actual` 0, charged the clipped log loss and counted in
  `n_actual_outside_model_classes` (plus `representable_only` metrics). The `all` aggregate uses the union of the
  models' classes with probability 0 for a class a model does not output.
* Current state on the real DB: 97 games checked, 0 recorded, because every ingested game is older than every
  production model. The ledger fills as 2027 starts are ingested. That is the first evidence no decision has seen.

## 7. Other API / CLI

* `GET /api/pitchers/{pid}/lifecycle`: production, previous_production, candidates, events, thresholds.
* `GET /api/pitchers/{pid}/dev-evaluation`: latest rolling dev run per model.
* `GET /api/pitchers/{pid}/retraining-replay`
* `pitchiq dev-eval <version>`: scores a frozen recipe on the rolling folds.
* `pitchiq replay-policies`

## Tests

`backend/tests/test_np_refit.py` has 25 tests, run offline on the fixture DB. Registry changes happen in a transaction
that is rolled back, with a throwaway MLflow store. They cover:

* roles
* chronological folds that never split a game and refuse 2026
* LightGBM checkpoint equivalence
* chronological calibration and the calibration rule
* final fit on all data before the cutoff
* MLflow tags
* in-sample refusal
* gated promotion and previous_production
* a failing gate leaving the model a candidate
* rollback and alias moves
* policy due rules
* retrain and `after_ingest`
* the policy replay simulating the live `policy_due` rule (weekly = 7 days + a new start, every_6_starts counted
  across the offseason, no season-start refit, manual = the single initial fit for both seasons)
* the challenge's in-sample flags when the default falls back to a refit trained on the pitches
* simulator support (`ood_report.training_rows`) equal to the refit's training set
* model comparison including the production refit
* prospective scoring with a label outside the model's classes and with differing class lists
* the dev_ece gate detail matching the calibration state, MLflow `split_policy` / `evaluation_roles` tags
* the benchmark job label (no "locked test")
* the prospective ledger (trained-before rule, exact probabilities, idempotence)
* the API

Full suite: 98 passed with `PITCHIQ_DATABASE_URL=postgresql+psycopg://pitchiq:pitchiq@localhost:5432/pitchiq_test`.

## Limitations

* Hyperparameters were selected on rolling folds that include 2025 games. The 2025 policy replay is therefore
  optimistic in absolute terms, though the bias is shared by all policies.
* The final fit's LightGBM rounds (125) were chosen on fold training windows of 1.7k-6k pitches, but the final model
  trains on 9.1k pitches. The best round count for the larger set may differ. There is no in-sample check of that.
* 2026 is retrospective (previously viewed). There is not yet any clean out-of-sample evaluation of a final-fit
  model. That needs the prospective ledger, i.e. 2027 games.
* The default retraining policy and the gate thresholds were set by this step's author from development evidence.
  Changing them is a protocol change and should be recorded.
* With only 4 classes and single-pitcher data, CH/SI recall stays near zero for every family and policy. This is a
  modelling limitation that the protocol does not fix.
* The replay checks the policy once per calendar day before that day's game. Live, the check runs whenever the
  ingestion job runs (`after_ingest`), so a live weekly schedule can drift from the replay's by however late
  ingestion happens. The replay's `manual` assumes no operator trigger at all.
* Review fixes after commit 023a0f9: replay runs 30-35 (old schedule) and promotion event 5's dev_ece detail text are
  kept as historical records and superseded by runs 36-41 and gate event 6.
