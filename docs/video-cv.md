# Video alignment and pre-pitch pause detection (Mode A), OpenCV + FFmpeg

**Read this first: every accuracy number on this page comes from synthetic clips.** No licensed real broadcast
footage exists in this project. The 9,090 assets in the main database are all `external_link` references to Baseball
Savant pages, and PitchIQ never downloads or trims MLB media. So the detector was developed and validated only
against broadcast-like clips rendered by `backend/tests/video_synth.py`, which also gives the exact ground truth.
These numbers say how well the detector recovers events in *those renders*. They are not a measurement on real MLB
broadcasts. Real footage will be harder: real bodies and uniforms, crowd and graphics motion, zooms and pans, replays,
dissolves, and pitchers who never come to a clean stop.

## What was built

| Piece | Where |
|---|---|
| Detector (decode → scene cuts → view classification → pitcher ROI → motion energy → onset / release → confidence; optional score-bug region) | `backend/pitchiq/video/alignment.py` (`DETECTOR_VERSION = "vcv-1.0"`) |
| Applying alignment to assets, reveal gate, FFmpeg trimming + cache | `backend/pitchiq/video/align_service.py` |
| Scoring against ground truth, storing reports, MLflow mirror | `backend/pitchiq/video/alignment_eval.py` |
| Tables `video_alignment_runs`, `video_alignment_validations` | `backend/pitchiq/models_video_cv.py`, migration `vcv01_video_alignment` (down_revision `aging02_bio_url`) |
| API | `backend/pitchiq/api/routers/video_cv.py` and the protected `GET /api/videos/{id}/media` in `routers/videos.py` |
| Job / CLI | job `align_videos` (`jobs/tasks.py`), `pitchiq align-videos [--force] [--asset-ids ...] [--limit N]` |
| Synthetic generator + validation runner | `backend/tests/video_synth.py` |
| Tests | `backend/tests/test_video_cv.py` (new), `backend/tests/test_video.py` (licensed-media test updated to the new protection) |

### Detector pipeline (`alignment.py`)

1. **Decode.** `ffprobe` reads the frame rate and duration. OpenCV reads the frames. Each frame is downscaled to 256 px
   wide and reduced to three things: an H-S histogram (16×8 bins), coarse 64×36 grass and dirt colour maps (HSV
   thresholds), and a 3×3-blurred grayscale image. Analysis stops after 90 s.
2. **Scene cuts.** Bhattacharyya distance between consecutive histograms. A cut is a local maximum above
   max(0.35, median + 10·MAD), with at least 0.3 s between cuts. Only hard cuts are modelled. Dissolves and wipes are not.
3. **Center-field view classification** (a documented heuristic, not a learned model). The broadcast pitcher/batter
   view looks from behind the pitcher toward home plate. Five cues are computed on the shot-averaged colour maps,
   each ramped to [0, 1]: grass across the middle band, dirt (the mound) at the bottom centre, grass on both flanks
   of the mound, no dirt across the middle band, and dirt around home plate in the upper centre. The view score is
   their geometric mean, and 0 if any cue is 0. A shot counts as center-field when its score is ≥ 0.5. The
   home-plate cue was added because without it the reverse view (camera behind home plate, plate dirt at the
   bottom) passed as center-field.
4. **Pitcher ROI.** The column range of the largest mound-dirt blob (bottom 45 % of the frame), and the rows from 40 %
   of the frame height to the bottom. This keeps out the batter, catcher and umpire near home plate, whose pre-pitch
   movement (bat waggle) would otherwise look like a delivery.
5. **Motion energy.** For each pair of consecutive frames:
   * Estimate the global camera translation by phase correlation. Shift both frames half-way, so both get the same
     interpolation blur.
   * Remove the median global change (brightness flicker).
   * Soft-threshold |difference| at max(3 grey levels, 3·the clip's noise σ), where σ is estimated from the median
     absolute difference.
   * Average over the ROI.
   * Apply a 5-tap temporal median. One bad frame gives two bad differences, so a 3-tap median is not enough.
6. **Delivery.**
   * **Release candidate:** the motion peak, which must reach 0.018 of the pitcher's share of the ROI. The pitcher's
     share is estimated as the non-grass, non-dirt pixels.
   * **Set position:** the latest 0.4 s window before the peak whose mean motion is within max(2 % of the peak,
     3 noise σ) of the stillest window.
   * **Onset:** a Page CUSUM (k = 2, h = 8, on motion standardised by max(σ_set, 1 % of the peak)) scanned forward
     from the set window. The onset is the last zero of the CUSUM before its alarm, refined backward (≤ 0.3 s) to the
     first frame above the set position's noise level (2 σ).
7. **Outputs.**
   * `onset_time`.
   * `release_time` (the motion peak).
   * `pre_pitch_pause_time = onset − 0.12 s`.
   * `pitch_end_time = release + 0.55 s`, capped at the end of the shot. This is a fixed ball-flight allowance,
     not a detection.
   * `clip_start_time` (start of the chosen center-field shot).
8. **Confidence.** The minimum of five components:
   * view score;
   * peak SNR, ramp(5 → 20);
   * set-position stillness: the share of quiet frames in the 0.6 s before the onset, ramp(0.5 → 0.85);
   * onset-to-release duration inside 0.45–2.0 s;
   * room between the shot start and the pause.

   Below 0.5 the clip is `unaligned`, the pre-pitch timestamps are cleared, and the reasons are recorded.
9. **Score-bug region (optional).** A high-edge, temporally constant rectangle near the top or bottom border. It is
   only labelled "candidate" when the clip has a scene cut that the region persists through. There is no OCR and no
   guarantee that the region is a scoreboard.

### How alignment is applied (`align_service.py`)

* **Scope.** Only assets with `playback_method='licensed_local'`, a `licensed:` rights reference, and an existing
  local file. Anything else gets 422.
* **Recording.** Every attempt writes a `video_alignment_runs` row. The row holds the full result: cuts, per-shot
  view scores, the ROI, the motion series, confidence components, reasons, the file fingerprint and the runtime.
* **Applying.** A confident result sets `synchronization_status='auto_aligned'` and writes the timestamps and
  `alignment_confidence` to `video_assets`.
* **Low confidence or decode failure.** The asset becomes `'unaligned'` with no pause point.
* **User corrections.** An asset corrected through `PUT /api/videos/{id}/alignment` (`user_corrected`) is never
  overwritten unless `force` is passed. The run is still recorded, with `applied=false`.
* **Idempotency.** `align_videos` skips an asset that already has an applied run for the same detector version and
  file fingerprint.

### Server-side spoiler protection

* `GET /api/videos/{id}/media`, the default (`segment=pre_pitch`):
  * Serves an FFmpeg re-encode of exactly `floor(pause·fps)` frames from the start of the clip, so it ends before
    `pre_pitch_pause_time`.
  * Video only: no audio, subtitle, data streams or metadata.
  * Cached under `data/video_segments/{asset}-{hash(file, pause)}.mp4`. A new pause, from a correction or a
    re-alignment, produces a new file and deletes the old one.
  * The encoded duration is checked with ffprobe and the segment is rejected if it is longer than the pause.
  * Unaligned clips get **409**: no pre-pitch segment exists. They use the data-only flow, then the full clip after
    reveal.
* `GET /api/videos/{id}/media?segment=full` serves the original file **only if a prediction for that pitch_uid has
  been revealed**, otherwise **403**. "Revealed" means either a challenge attempt with `revealed=true`, or a
  `pitch_predictions` row with `revealed_at` set in context `challenge` or `dashboard`. Prospective-ledger and
  benchmark rows do not unlock video. PitchIQ has no user accounts, so the gate is per pitch, not per viewer.
* Media responses carry `Cache-Control: private, no-store`.
* **Challenge attempt view (pre-reveal).**
  * `video` gains `pre_pitch_segment_available` and `pre_pitch_media_url = /api/attempts/{attempt_id}/pre-pitch-media`.
    These are only set for in-app clips that are aligned.
  * The URL is scoped to the attempt on purpose. The video asset id is still not exposed before reveal, because
    other endpoints, such as the video search with labels, can tie an asset id to the pitch type.
  * That endpoint only ever serves the pre-pitch segment.
* **Reveal payload.**
  * `video_reference.media_url` is now `/api/videos/{id}/media?segment=full`. The existing frontend player keeps
    working: it seeks to `pre_pitch_pause_time`.
  * Added fields: `pre_pitch_media_url`, `synchronization_status`, `alignment_confidence` and `pitch_end_time`.
* **Video search rows.** Added `full_media_url`, `pre_pitch_segment_available`, `alignment_confidence` and
  `pre_pitch_pause_time`. `media_url` keeps its path, but by default it now returns the pre-pitch segment.
* **New endpoints.**
  * `POST /api/videos/{id}/align?force=`
  * `GET /api/videos/{id}/alignment` (current values plus the latest run's diagnostics)
  * `GET /api/videos/alignment/validations` (stored accuracy reports, each labelled with its `data_kind`)

## Validation on synthetic clips

### Generator (`tests/video_synth.py`, `synth-v1`)

Each clip is rendered with known ground truth and encoded with libx264 (yuv420p). The structure is:

* An optional lead-in shot (crowd or player close-up), then a hard cut.
* The center-field view:
  * backstop/crowd band; home-plate dirt circle with batter (optional bat waggle) and catcher; mowed grass; mound
    with rubber;
  * an articulated pitcher seen from behind, who stands and nods, comes set, holds a static set position, and
    delivers. The delivery is a leg lift starting from zero velocity, a stride, arm cocking, and acceleration to
    release at maximum hand speed, followed by a follow-through. A ball travels to the catcher.
* An optional tail close-up after another cut.

Randomised per seed:

* frame rate: 24, 25, 29.97, 30, 50, 59.94 or 60 fps;
* resolution: 480×270 to 960×540;
* sensor noise σ: 0–5;
* sub-pixel camera jitter: 0–0.8 px random walk;
* CRF: 18–30;
* set-position length: 0.7–2.5 s;
* onset-to-release time: 0.8–1.25 s;
* pitcher scale and offset, uniform colours;
* score bug: present on 70 % of clips, in one of four corners.

Negative kinds: crowd only, close-up only, wide outfield view, reverse (behind-home) view with a delivery, and
center-field view where the pitcher comes set but never delivers.

### Protocol

* **Development set:** seeds 1–30, plus one negative per kind (seeds 900000–900004). These were the only clips looked
  at while choosing the motion measure and every threshold and margin above.
* **Evaluation set:** a fresh draw from the same generator, seeds 10001–10060, plus two negatives per kind (seeds
  950000–950009). It was run once, after the parameters were frozen.
* Nothing was changed after the evaluation run. Its worst-case pause lead (below) is reported, not tuned away.
* **Error definition:** onset error = detected onset − true onset. A "gross error" is |error| > 200 ms. "Failure" means
  unaligned or a gross error on a clip that contains a delivery.
* **Command:** `cd backend && python tests/video_synth.py --n-dev 30 --n-eval 60 --neg-per-kind 2 --store`.

### Results

Stored in `video_alignment_validations` (main DB `pitchiq`: id 1 = development, id 2 = evaluation). Also mirrored
to MLflow experiment `pitchiq-video-cv`: run `bd4eb2d7814643739166073f28ceb413` (development) and run
`041ad9c32d0c4ba6844a2ad2569aed05` (evaluation).

| Synthetic clips | development (id 1) | evaluation (id 2) |
|---|---|---|
| clips (delivery / negative) | 35 (30 / 5) | 70 (60 / 10) |
| delivery clips auto-aligned | 30 / 30 | 60 / 60 |
| failure rate (unaligned + gross error) | 0.0 % | 0.0 % |
| onset \|error\| median | 43.8 ms (1.7 frames) | **48.3 ms** (1.6 frames) |
| onset \|error\| p90 / p95 / max | 67.1 / 70.3 / 83.8 ms | **69.9** / 78.6 / 117.9 ms |
| onset signed error median (p10, p90) | +43.8 (+27.4, +67.1) ms | +48.0 (+19.6, +69.9) ms |
| release (motion-peak) \|error\| median / p90 | 64.6 / 141.0 ms | 63.0 / 97.9 ms |
| pause point before the true onset | 100 % | 100 % (smallest lead 2.1 ms, median 72.0 ms) |
| negatives falsely aligned | 0 / 5 | 0 / 10 |
| scene-cut recall / false cuts | 1.0 / 0 | 1.0 / 0 |
| score bug found (IoU ≥ 0.5) | 13 / 17 | 38 / 45 (median IoU 0.9) |
| score-bug false positives on clips without one | 2 | 4 |
| median runtime per clip (4 parallel workers) | 1.97 s | 1.24 s |

Evaluation onset |error| by frame rate (median / p90, ms):

| 24 (n=13) | 25 (n=9) | 29.97 (n=5) | 30 (n=11) | 50 (n=14) | 59.94 (n=5) | 60 (n=3) |
|---|---|---|---|---|---|---|
| 54.1 / 69.7 | 44.8 / 56.6 | 44.4 / 50.6 | 40.3 / 60.4 | 50.2 / 75.2 | 60.2 / 104.1 | 48.3 / 62.0 |

All 10 evaluation negatives were unaligned:

* the crowd, close-up, outfield and behind-home clips with "no center-field (pitcher/batter) view found";
* both no-delivery clips with "no delivery-sized motion", relative peak 0.0065 and 0.0070 against the 0.018 threshold.

### Reading the numbers

* **The onset is detected late, by design and consistently.** The rendered leg lift starts with zero velocity, so
  the first frames after the true onset barely change any pixels. The evaluation signed errors range from −52 to
  +118 ms. The 0.12 s pause margin turns that into a pause point that preceded the true onset on every clip.
  However, the smallest lead was only 2.1 ms on one clip, so the safety margin is thin in the worst case. For real
  footage, a larger `AlignmentParams.pause_margin_s` (for example 0.2 s) would be prudent. The pre-pitch segment
  only needs to stop somewhere inside the set position. The margin was not changed after the evaluation run.
* **The release estimate is the motion-energy peak, not a detected release.** It tends to land in the
  follow-through: median +62 ms, range −218 to +110 ms. Treat `release_frame_timestamp` as approximate.
* The generator and the detector were written together, by the same author, for this step. Fresh seeds test
  generalisation **within this generator's distribution only**.

## Run on the real data

`pitchiq align-videos` on the main database reported `licensed_local_assets: 0`. All 9,090 video assets are
`external_link` / `not_applicable`, and `PITCHIQ_LICENSED_VIDEO_DIR` is unset. So there was nothing real to align,
and no real-footage accuracy exists. When an operator adds licensed clips, run `pitchiq videos`, then
`pitchiq align-videos`, or the job `align_videos`. Review any `unaligned` results, or correct them with
`PUT /api/videos/{id}/alignment`.

## Tests (`tests/test_video_cv.py`)

* **Accuracy on 10 synthetic clips.** These use separate test seeds 30001–30010, disjoint from development and
  evaluation. The test asserts:
  * at least 9 of 10 are aligned;
  * median |onset error| ≤ 100 ms and max ≤ 200 ms;
  * the pause precedes the true onset;
  * the release is within 350 ms;
  * scene cuts are within 1.5 frames;
  * the center-field view is recognised.
* **Negatives.** No-pitcher-view clips (crowd, close-up, behind-home) and the no-delivery clip end up `unaligned`.
* **Undecodable file.** Gives `unaligned` with the error recorded.
* **View-classifier cues** on constructed colour maps.
* **End-to-end media protection:**
  * before alignment, 409;
  * after `align_videos`, `auto_aligned`, with a run row and idempotent skip;
  * the default media is a segment whose ffprobe duration is ≤ the pause and whose frame count is
    `floor(pause·fps)`;
  * `segment=full` gives 403 before reveal;
  * the challenge pre-reveal view exposes only the attempt-scoped URL (no asset id), and that URL serves the segment;
  * after reveal, the full clip gives 200 and is byte-identical to the licensed file;
  * a user correction produces a shorter segment (≤ 1.0 s) and is kept by re-alignment.
* **Unaligned clip:** 409 for the segment, 403 for the full clip, then 200 after a dashboard reveal.
* **External assets** are rejected by align/alignment (422) and media (404).
* The job is registered.
* A validation report is stored, labelled synthetic, and mirrored to MLflow.

The existing test in `tests/test_video.py` that expected the unprotected 200 now expects 409 (unaligned) and
403 (full clip before reveal).

## Limitations

* Accuracy is on **synthetic clips only**. The view classifier's colour rules are a plausible description of the
  center-field shot, but they are uncalibrated on real broadcasts. Real-world problems include: different
  stadium dirt and grass colours and lighting; night games; shadows; the pitcher covering the mound; overlays; and
  non-standard camera angles. All of these can make it misclassify. A misclassified clip is labelled `unaligned`
  rather than given a wrong pause point only to the extent that the confidence components catch it.
* Not handled: dissolves and wipes, replays inside a clip, zooms during the delivery, pickoff throws and step-offs.
  A step-off produces large motion and could be accepted as a delivery. Only the no-delivery case (coming set and
  holding) was tested. Also not handled: pitchers without a complete stop, such as quick pitches. Multiple
  center-field shots are resolved by picking the most confident one.
* `pitch_end_time` is a fixed allowance after the release estimate.
* Score-bug detection is a candidate region only: no OCR, and false positives occurred (4 on evaluation clips
  without a score bug).
* The pre-pitch segment drops audio.
* The reveal gate is per pitch, because PitchIQ has no user accounts.
* Media is served from local licensed files only. External or official links are never fetched, trimmed or proxied.
* The running uvicorn API on :8000 does not auto-reload. It must be restarted to expose the new routes and the
  protected media behaviour.
* The frontend was not changed. The challenge UI can use `video.pre_pitch_media_url` before reveal.
