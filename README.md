# PitchIQ AI

MLB next-pitch prediction research platform. The proof of concept is **Chris Sale (MLBAM 519242)**, using his
real 2023–2026 regular-season Statcast pitches, official MLB pitch-video references, and models evaluated on a locked
held-out season.

Every probability shown by the app is a genuine model output. Every accuracy figure comes from a stored
evaluation run. Nothing here is mocked.

```
backend/   Python: ingestion, PostgreSQL schema + migrations, features, ML, video discovery, jobs, FastAPI
frontend/  Next.js + TypeScript + Tailwind + Recharts dashboard (see frontend/README.md)
```

## Quick start (local)

Requirements: Python ≥ 3.11, PostgreSQL 16, Node 20+. Apple Silicon works (all models are CPU-first).

```bash
cd backend && pip install -e ".[dev]"
```

```bash
createdb pitchiq   # or: docker compose up -d db
```

```bash
pitchiq bootstrap   # migrate → ingest → feeds → validate → features → video library → train → benchmark → walk-forward → ablation
```

```bash
pitchiq serve        # API on :8000 (OpenAPI docs at /docs)
```

```bash
pitchiq worker       # background job worker (needed for jobs started from the UI)
```

```bash
cd frontend && npm install && npm run dev   # dashboard on :3000, proxies /api to :8000
```

The full stack also runs with `docker compose up --build`. Configuration uses `PITCHIQ_*` environment variables
(see `backend/.env.example`), including the pitcher, the seasons, and the train/validation/test season split.

Tests (offline; they run the real pipeline on fixtures cut from real Statcast and MLB Stats API responses, against a
separate `pitchiq_test` database built from the Alembic migrations):

```bash
cd backend && createdb pitchiq_test && python -m pytest -q
```

## Architecture

| Layer | Module | Notes |
|---|---|---|
| Ingestion | `pitchiq/ingest/` | Baseball Savant Statcast CSV (the endpoint pybaseball wraps), fetched month by month with a polite rate-limited client and a raw-response cache. MLB Stats API live feeds supply batting-order slots, batter metadata, and per-game validation evidence. |
| Validation | `ingest/pipeline.validate` | Checks Statcast pitch counts against the live feed, PA pitch-number continuity, counts at the start of each PA, and lineup coverage. Reports are persisted. |
| Database | `pitchiq/models.py`, `alembic/` | 19 tables: pitchers, batters, games, game_lineups, plate_appearances, pitches, pre_pitch_states, pitch_sequences, video_assets, pitch_predictions, prediction_challenges, challenge_attempts, model_registry, evaluation_runs, jobs, pipeline_traces, saved_scenarios, data_validation_reports. Natural-key constraints, FKs, CHECKs, indexes. The full original Statcast row is kept in `pitches.raw` for provenance. |
| Features | `pitchiq/features/` | One function (`make_features`) builds every feature from the pre-pitch situation plus history strictly before the pitch. Both historical reconstruction and the simulator call it. |
| ML | `pitchiq/ml/` | Majority class, hierarchical smoothed frequency baseline, multinomial logistic regression, LightGBM. Temperature-scaling calibration. Exact linear attributions for logreg; TreeSHAP (LightGBM `pred_contrib`) for GBM. Artifacts are stored with SHA-256 checksums and registered in `model_registry`. |
| Evaluation | `ml/evaluate.py`, `ml/metrics.py` | Full holdout benchmark (stores every prediction), walk-forward evaluation, validation-only ablation. Top-1/top-2, log loss, multiclass Brier, ECE + reliability curves, confusion matrix, per-class P/R/F1, breakdowns with Wilson CIs, game-level cluster bootstrap. |
| Video | `pitchiq/video/` | Provider adapters → metadata matcher with confidence levels → `video_assets`, coverage reporting, availability checks, review queue. |
| Challenge | `pitchiq/challenge.py` | Seeded sampling from the locked test split; the prediction is frozen before anything about the outcome is read; reveal and live scoring with baselines on the same pitches. |
| Jobs | `pitchiq/jobs/` | Postgres-backed queue (`FOR UPDATE SKIP LOCKED`) with progress, logs, cancellation, and safe retry. All tasks are idempotent. |
| Observability | `pitchiq/trace.py` | Each inference records its real stages (timings, inputs, outputs, warnings) to `pipeline_traces`. |
| API | `pitchiq/api/` | FastAPI: pitchers/analytics, blind replay, simulator (predict, compare, sweep, saved scenarios), models/leaderboard/compare, challenges, videos, jobs, traces. |

Adding a pitcher means running `pitchiq --pitcher <id> bootstrap`. Nothing in the code is Sale-specific: pitch
classes are derived from that pitcher's own training data.

## Data

* **Source:** Baseball Savant Statcast search CSV (https://baseballsavant.mlb.com/csv-docs), regular season (`game_type=R`).
* **Loaded:** 9,096 rows across 97 games (2023: 1,672 · 2024: 2,818 · 2025: 2,031 · 2026: 2,575). Season totals agree with
  the MLB Stats API season pitch counts.
* **Validation:** 6 games have one more Statcast row than the live feed has pitches. Each of those 6 rows is a pitch-clock
  `automatic_ball` / `automatic_strike` with no pitch type. These rows are kept and move the count, but they are not
  prediction targets and are not counted as thrown pitches. Batting-order slots resolved for 100% of plate appearances.
* **Arsenal:** Statcast classifies Sale's pitches as FF, SL, CH, SI in every season (no class falls under the 1% "OTHER" threshold).

## Features (all known before the pitch)

Batter handedness · batting-order slot · balls, strikes, count · inning · pitches already thrown this game
(distinct from the pitch number within the PA) · times through the order · outs, runners, score differential, home/away ·
previous 1–3 pitch types · whether the previous pitch was in the same PA, its result, and its velocity · this game's
mean four-seam velocity so far · smoothed in-game usage shares · usage over the previous 3 games · prior-games usage in
this exact count · prior-games transition probabilities from the previous pitch type · smoothed history against this
batter · pitch types already thrown in this PA · days of rest.

Statcast measurements of the **target pitch** (velocity, spin, movement, location, result, events) are listed in
`FORBIDDEN_SOURCE_COLUMNS` and are only read after that pitch's features are frozen, to update state for the *next* pitch.

Leakage safeguards, each enforced by a test:

- truncation invariance: a pitch's features are identical when computed from only the rows up to it, with its own outcome scrambled
- changing future rows never changes earlier features
- no game spans splits, and splits are chronological
- the inference path never selects the label: corrupting stored labels does not change predictions
- the API never returns the label or video reference before reveal

## Models and evaluation methodology

* **Split:** train 2023–24 (4,485 pitches) · validation 2025 (2,031) · **locked test 2026 (2,574)**.
* **Selection:** hyperparameter grids, early stopping, calibration temperature, and model promotion all use
  validation only. The test split is read only by the benchmark.
* **Official metrics:** the full holdout benchmark predicts every eligible 2026 pitch with frozen models. Challenge
  sessions are interactive demonstrations on subsets.

### Results (real; from `evaluation_runs`)

Locked 2026 holdout, n = 2,574, every model scored on the same pitches. 95% CIs come from a game-level bootstrap.

| Model | Top-1 | Top-2 | Log loss | Brier | ECE |
|---|---|---|---|---|---|
| Majority class | 41.4% | 81.3% | 1.213 | 0.658 | 0.035 |
| Frequency baseline | 46.0% [44.3, 47.9] | 82.1% | 1.054 | 0.622 | 0.078 |
| Logistic regression | 48.5% [46.4, 50.9] | 83.0% | 1.056 | 0.614 | 0.062 |
| LightGBM (promoted on validation) | 47.4% [45.5, 49.3] | 81.7% | 1.063 [1.03, 1.09] | 0.622 | 0.081 |

Walk-forward on 2026 (refit every 6 games on all earlier games; hyperparameters and temperature frozen):

| Model | Top-1 | Log loss | Brier |
|---|---|---|---|
| Frequency | 48.0% | 1.031 | 0.612 |
| Logistic regression | 47.9% | 1.018 | 0.602 |
| **LightGBM** | **50.5%** [48.5, 53.0] | **1.001** | **0.593** |

How to read these numbers:

- All learned models clearly beat the majority class.
- On the static holdout, frequency, logreg, and LightGBM have overlapping CIs, so they **cannot be reliably ranked**.
- Validation favoured LightGBM (0.952 log loss vs 0.964 for frequency), so it stays promoted. Re-selecting on test results would be tuning on the locked set.
- 2026 is harder than 2025 because Sale's mix moved back toward CH/SI. Walk-forward refitting recovers most of the gap, which says recency matters more than model family.
- The temperature fit on 2025 is below 1 for every model and slightly hurts 2026 log loss (uncalibrated is better), another symptom of season-to-season drift.

**Ablation** (validation, LightGBM, 3 seeds; Δ log loss when a group is removed, positive means the group helps):

- sequence +0.037 · handedness +0.020 · usage +0.013 · count +0.012
- situation, pitch count, TTO, inning, context: +0.001 to +0.003
- lineup slot +0.001
- batter history −0.002 (no measurable benefit)

### Evaluation log (transparency)

1. The first trained set (v1) used numeric `season` as a model input. It was benchmarked once on the test split. Logreg
   collapsed (41.1% top-1, 1.283 log loss) because a linear model extrapolates `season` into unseen years.
2. `season` was removed from model inputs. The basis was the a-priori extrapolation argument plus validation evidence:
   dropping the context group improved logreg **validation** log loss by 0.036. Models were retrained and re-benchmarked.
   The v1 models are kept as `archived` in the registry along with their benchmark runs.

Because the test split has now been viewed twice, the 2026 numbers are slightly optimistic for logreg. A clean
confirmation needs a new untouched partition, for example the 2027 season.

## Video access (what is and is not done)

* **How discovery works:** for every game, the official **MLB Stats API live feed** supplies each pitch's Statcast `playId`.
  Candidates are matched to Statcast pitches on game_pk, pitcher, batter, at-bat number, and pitch number, plus the pre-pitch count and
  the feed's pitch classification. **Titles are never used.**
* **Confidence levels:** `exact_verified` (everything agrees), `strong_review` (IDs agree, a secondary check disagrees,
  queued for human review), `possible`, `no_match`. Only `exact_verified` clips enter challenges.
* **Current library:** 9,090 candidates → **9,074 exact verified**, 16 awaiting review. The review-queue clips are at-bats where a
  pitch-clock call shifts the numbering or the classification differs. Availability is checked lazily and rate-limited;
  all spot checks so far found official video.
* **Playback: Mode B, official external link.** PitchIQ links to Baseball Savant's public per-pitch video page
  (`/sporty-videos?playId=…`) and only shows that link **after** the prediction is frozen and the user reveals.
  It does **not** extract, download, re-host, or embed the media file: no embedding or download licence has been
  verified for this integration. MLB Film Room has no adapter: mlb.com's robots.txt disallows its `/search` and `/api/` paths, and no public, supported
  automated access method could be verified. A licensed-provider adapter can be added behind the same interface
  (`video/providers/base.py`).
* **Mode A, in-app playback:** available for clips an operator is licensed to use. Point `PITCHIQ_LICENSED_VIDEO_DIR` at a folder
  with `manifest.csv` (file, game_pk, at_bat_number, pitch_number, pitcher_id, batter_id, license_reference). Rows
  without a licence reference are ignored. Pre-pitch pause timestamps can be corrected through `PUT /api/videos/{id}/alignment`.
* **Mode C, data-only:** used automatically when no verified clip is available. The app works with zero clips.

## Known limitations / not yet built

* **Automatic video alignment:** the OpenCV/FFmpeg delivery-start detector is not built. No licensed controllable footage is
  available to develop or validate it against. The schema, Mode A player path, and manual-correction endpoint exist.
* **Sequential model** (GRU/LSTM/Transformer): not built. The tabular models and evaluation harness come first, and the
  ablation shows the sequence features already carry most of the signal.
* **LLM explanations:** not implemented. The narrative endpoint is a deterministic template over stored attribution values,
  and it never produces probabilities.
* **Experiment tracking** lives in Postgres (`model_registry`, `evaluation_runs` with dataset fingerprints and grid results)
  rather than MLflow.
* **Simulator assumptions:** pitcher history is taken as of a chosen date. Missing previous-pitch velocity is imputed with that
  pitch type's historical mean. If the pitch count is above 0 and no previous pitch is entered, the output averages the
  model over the training distribution of previous pitch and result for that count; the response says so. Simulator
  outputs show model sensitivity, not causal effects, and are never mixed into accuracy statistics.
* **Expansion to 10 pitchers** is supported by the architecture but not yet run. The selection criterion (for example, top 10 by
  pitches thrown among qualified starters in the latest season) is to be documented when it is executed.
* **Batter identity** enters only as smoothed batter-vs-pitcher history, which showed no validation benefit. Raw IDs are not used.

## Data and rights

Statcast and MLB Stats API data © MLB Advanced Media. They are used here for non-commercial research, with
source identifiers kept on every row. Video remains on MLB's own pages. PitchIQ stores only references and metadata.
