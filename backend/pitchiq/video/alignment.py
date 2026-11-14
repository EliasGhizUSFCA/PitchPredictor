"""Automatic pre-pitch alignment for controllable, licensed clips (Mode A). OpenCV + FFmpeg.

Pipeline (all heuristics are documented here and in docs/video-cv.md):

1. **Decode** (`decode`): ffprobe for frame rate / duration, OpenCV for frames. Each frame is downscaled to
   `analysis_width` and reduced to: an HSV histogram (scene cuts), coarse colour-class maps (grass / dirt, for the
   view classifier) and a blurred grayscale image (motion energy, score-bug detection).
2. **Scene cuts** (`detect_scene_cuts`): Bhattacharyya distance between consecutive H-S histograms. A cut is a
   local maximum above max(`cut_abs_threshold`, median + `cut_mad_k` * MAD). Hard cuts only; dissolves are not
   modelled.
3. **View classification** (`classify_view`): the broadcast "center-field" view looks from behind the pitcher toward
   home plate, so it has (a) grass across the middle band between mound and plate, (b) a dirt mound at the bottom
   centre and (c) grass flanking the mound. Each cue is mapped to [0, 1]; the view score is their geometric mean.
   Crowd shots, close-ups, outfield views and the reverse (behind-home) view fail at least one cue.
4. **Pitcher ROI** (`pitcher_roi`): the column range of the mound blob, the rows from above the mound up to the
   lower `roi_top` of the frame. This excludes the batter/catcher near home plate, whose pre-pitch movement would
   otherwise look like a delivery.
5. **Motion energy** (`motion_energy`): fraction of ROI pixels whose frame difference (after removing the median
   global brightness change) exceeds `diff_threshold` grey levels.
6. **Delivery onset** (`detect_delivery`): release candidate = the motion peak; the set position = the latest
   low-motion window of `still_window_s` before it; onset = Page CUSUM change point (last zero of the CUSUM before
   its alarm) on the standardised motion series scanned forward from the set window.
7. **Release** = the motion-energy peak; **pre_pitch_pause_time** = onset - `pause_margin_s`;
   **pitch_end_time** = release + `ball_flight_s` (capped at the end of the segment).
8. **Confidence**: the weakest of view score, peak SNR, stillness of the set position, plausibility of the
   onset->release duration and the room left before the pause. Below `min_confidence` the clip is "unaligned"
   (no pre-pitch segment is served; manual correction via PUT /api/videos/{id}/alignment).
9. Optional **score-bug region** (`detect_scoreboard`): a high-edge, temporally constant rectangle near a frame
   border. Detection only; no OCR, no guarantee.

Validated only on synthetic clips (tests/video_synth.py). See docs/video-cv.md.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import numpy as np

DETECTOR_VERSION = "vcv-1.0"


@dataclass
class AlignmentParams:
    analysis_width: int = 256
    max_seconds: float = 90.0
    # scene cuts
    hist_bins: tuple = (16, 8)
    cut_abs_threshold: float = 0.35
    cut_mad_k: float = 10.0
    min_segment_s: float = 0.5
    # view classifier
    view_threshold: float = 0.5
    # motion
    diff_threshold: float = 3.0  # minimum soft threshold (grey levels); raised to 3x the clip's noise sigma
    roi_top: float = 0.40
    smooth_s: float = 0.05
    motion_compensation: bool = True
    median_taps: int = 5
    # delivery detection
    still_window_s: float = 0.4
    still_tol_frac: float = 0.02  # set window: mean motion within this fraction of the peak of the stillest window
    still_tol_sigma: float = 3.0  # ... or within this many noise sigmas
    cusum_k: float = 2.0
    cusum_h: float = 8.0
    sigma_floor: float = 0.00002
    sigma_peak_frac: float = 0.01
    min_peak_rel: float = 0.018  # peak energy / pitcher figure fraction of the ROI
    min_delivery_s: float = 0.45
    max_delivery_s: float = 2.0
    skip_after_cut_s: float = 0.15
    still_check_s: float = 0.6
    refine_sigma: float = 2.0
    refine_max_s: float = 0.3
    pause_margin_s: float = 0.12
    ball_flight_s: float = 0.55
    min_confidence: float = 0.5
    detect_scoreboard: bool = True

    def as_dict(self) -> dict:
        return {k: (list(v) if isinstance(v, tuple) else v) for k, v in asdict(self).items()}


@dataclass
class Decoded:
    fps: float
    n_frames: int
    duration: float
    width: int
    height: int
    hists: np.ndarray  # (N, bins) float32
    gray: np.ndarray  # (N, h, w) uint8, blurred
    grass: np.ndarray  # (N, 36, 64) uint8 0/1 coarse maps
    dirt: np.ndarray
    truncated: bool = False


@dataclass
class AlignmentResult:
    detector_version: str = DETECTOR_VERSION
    status: str = "unaligned"  # auto_aligned | unaligned
    alignment_confidence: float = 0.0
    fps: float | None = None
    n_frames: int | None = None
    clip_duration: float | None = None
    scene_cuts: list = field(default_factory=list)
    segments: list = field(default_factory=list)
    chosen_segment: int | None = None
    clip_start_time: float | None = None  # start of the chosen center-field shot
    onset_time: float | None = None
    release_time: float | None = None
    pre_pitch_pause_time: float | None = None
    pitch_end_time: float | None = None
    confidence_components: dict = field(default_factory=dict)
    roi: list | None = None
    scoreboard_region: dict | None = None
    motion: dict | None = None  # chosen segment's motion series (for diagnostics / UI)
    reasons: list = field(default_factory=list)
    error: str | None = None

    def to_dict(self) -> dict:
        return json.loads(json.dumps(asdict(self), default=float))


# ------------------------------------------------------------------ decode
def ffprobe(path: str | Path) -> dict:
    """Frame rate, duration, size and stream info via ffprobe (raises RuntimeError if unreadable)."""
    exe = shutil.which("ffprobe")
    if exe is None:
        raise RuntimeError("ffprobe not found on PATH")
    r = subprocess.run([exe, "-v", "error", "-select_streams", "v:0", "-show_entries",
                        "stream=avg_frame_rate,r_frame_rate,nb_frames,width,height,duration:format=duration",
                        "-of", "json", str(path)], capture_output=True, text=True, timeout=60)
    if r.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {r.stderr.strip()[:300]}")
    d = json.loads(r.stdout or "{}")
    if not d.get("streams"):
        raise RuntimeError("no video stream")
    st = d["streams"][0]

    def rate(s):
        try:
            a, b = s.split("/")
            return float(a) / float(b) if float(b) else None
        except (ValueError, AttributeError):
            return None
    fps = rate(st.get("avg_frame_rate")) or rate(st.get("r_frame_rate"))
    dur = st.get("duration") or d.get("format", {}).get("duration")
    return {"fps": fps, "duration": float(dur) if dur else None, "width": st.get("width"), "height": st.get("height"),
            "nb_frames": int(st["nb_frames"]) if str(st.get("nb_frames", "")).isdigit() else None}


def _coarse_masks(hsv_small: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    h, s, v = hsv_small[..., 0], hsv_small[..., 1], hsv_small[..., 2]
    grass = (h >= 35) & (h <= 85) & (s >= 60) & (v >= 40)
    dirt = (h >= 5) & (h <= 25) & (s >= 60) & (s <= 220) & (v >= 60) & (v <= 235)
    return grass.astype(np.uint8), dirt.astype(np.uint8)


def decode(path: str | Path, params: AlignmentParams | None = None) -> Decoded:
    p = params or AlignmentParams()
    meta = ffprobe(path)
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError("OpenCV could not open the file")
    fps = meta["fps"] or cap.get(cv2.CAP_PROP_FPS) or 0.0
    if not fps or fps <= 0 or fps > 480:
        raise RuntimeError(f"unusable frame rate {fps}")
    max_frames = int(p.max_seconds * fps)
    hists, grays, grass, dirt = [], [], [], []
    w0 = h0 = None
    truncated = False
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        if len(grays) >= max_frames:
            truncated = True
            break
        if w0 is None:
            h0, w0 = fr.shape[:2]
        aw = p.analysis_width
        ah = max(int(round(h0 * aw / w0)), 8)
        small = cv2.resize(fr, (aw, ah), interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, list(p.hist_bins), [0, 180, 0, 256]).flatten()
        hists.append(hist / max(hist.sum(), 1.0))
        g = cv2.GaussianBlur(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), (3, 3), 0)
        grays.append(g)
        gm, dm = _coarse_masks(cv2.resize(hsv, (64, 36), interpolation=cv2.INTER_NEAREST))
        grass.append(gm)
        dirt.append(dm)
    cap.release()
    if len(grays) < 3:
        raise RuntimeError("too few decodable frames")
    n = len(grays)
    return Decoded(fps=float(fps), n_frames=n, duration=n / fps, width=int(w0), height=int(h0),
                   hists=np.asarray(hists, np.float32), gray=np.asarray(grays), grass=np.asarray(grass),
                   dirt=np.asarray(dirt), truncated=truncated)


# ------------------------------------------------------------------ scene cuts + views
def hist_distances(hists: np.ndarray) -> np.ndarray:
    d = np.zeros(len(hists), np.float32)
    for i in range(1, len(hists)):
        d[i] = cv2.compareHist(hists[i - 1], hists[i], cv2.HISTCMP_BHATTACHARYYA)
    return d


def detect_scene_cuts(hists: np.ndarray, fps: float, params: AlignmentParams | None = None) -> list[int]:
    """Frame indices that start a new shot."""
    p = params or AlignmentParams()
    d = hist_distances(hists)
    if len(d) < 3:
        return []
    med = float(np.median(d[1:]))
    mad = float(np.median(np.abs(d[1:] - med))) * 1.4826
    thr = max(p.cut_abs_threshold, med + p.cut_mad_k * mad)
    min_gap = max(int(0.3 * fps), 2)
    cuts: list[int] = []
    for i in range(1, len(d)):
        if d[i] > thr and d[i] >= d[max(i - 1, 0)] and d[i] >= d[min(i + 1, len(d) - 1)]:
            if cuts and i - cuts[-1] < min_gap:
                if d[i] > d[cuts[-1]]:
                    cuts[-1] = i
                continue
            cuts.append(i)
    return cuts


def _frac(m: np.ndarray, y0, y1, x0, x1) -> float:
    H, W = m.shape
    sub = m[int(y0 * H):max(int(y1 * H), int(y0 * H) + 1), int(x0 * W):max(int(x1 * W), int(x0 * W) + 1)]
    return float(sub.mean()) if sub.size else 0.0


def classify_view(grass_mean: np.ndarray, dirt_mean: np.ndarray, params: AlignmentParams | None = None) -> dict:
    """Center-field (pitcher/batter) view heuristic on segment-averaged coarse colour maps."""
    p = params or AlignmentParams()
    mid_grass = _frac(grass_mean, 0.38, 0.70, 0.0, 1.0)
    mound = _frac(dirt_mean, 0.80, 1.0, 0.38, 0.62)
    flank = 0.5 * (_frac(grass_mean, 0.72, 1.0, 0.0, 0.22) + _frac(grass_mean, 0.72, 1.0, 0.78, 1.0))
    mid_dirt = _frac(dirt_mean, 0.38, 0.70, 0.0, 1.0)
    plate = _frac(dirt_mean, 0.12, 0.36, 0.33, 0.67)

    def ramp(x, lo, hi):
        return float(np.clip((x - lo) / (hi - lo), 0.0, 1.0))
    c = {"mid_grass": ramp(mid_grass, 0.30, 0.60), "mound_dirt": ramp(mound, 0.15, 0.40),
         "flank_grass": ramp(flank, 0.25, 0.55), "no_mid_dirt": ramp(0.5 - mid_dirt, 0.0, 0.3),
         "plate_dirt": ramp(plate, 0.10, 0.30)}
    score = float(np.prod([max(v, 1e-6) for v in c.values()]) ** (1 / len(c)))
    if min(c.values()) == 0:
        score = 0.0
    view = "center_field" if score >= p.view_threshold else "other"
    return {"view": view, "score": round(score, 4), "cues": {k: round(v, 3) for k, v in c.items()},
            "raw": {"mid_grass": round(mid_grass, 3), "mound_dirt": round(mound, 3), "flank_grass": round(flank, 3),
                    "mid_dirt": round(mid_dirt, 3), "plate_dirt": round(plate, 3)}}


def pitcher_roi(dirt_mean: np.ndarray, params: AlignmentParams | None = None) -> list[float]:
    """Normalised [x0, y0, x1, y1] around the mound and the pitcher standing on it."""
    p = params or AlignmentParams()
    H, W = dirt_mean.shape
    m = (dirt_mean > 0.5).astype(np.uint8)
    m[: int(0.55 * H)] = 0
    m[:, : int(0.15 * W)] = 0
    m[:, int(0.85 * W):] = 0
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=4)
    if n > 1:
        k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        x, y, w, h, _a = stats[k]
        cx, mw = (x + w / 2) / W, w / W
    else:
        cx, mw = 0.5, 0.4
    half = float(np.clip(0.55 * mw, 0.12, 0.3))
    return [max(cx - half, 0.0), p.roi_top, min(cx + half, 1.0), 1.0]


def motion_energy(gray: np.ndarray, roi: list[float], params: AlignmentParams | None = None) -> np.ndarray:
    """e[i] = mean soft-thresholded absolute frame difference in the ROI between frames i-1 and i (e[0] = 0).

    Camera motion is compensated first: the global translation between consecutive full frames is estimated by
    phase correlation and frame i-1 is shifted onto frame i. The median global change is then removed (brightness
    flicker), the noise floor is estimated per clip from the median absolute difference (most ROI pixels are
    static), and a 5-frame temporal median suppresses short spikes (one bad frame produces two bad differences).
    Units: grey levels / 255 averaged over the ROI.
    """
    p = params or AlignmentParams()
    n, H, W = gray.shape
    x0, y0, x1, y1 = int(roi[0] * W), int(roi[1] * H), int(np.ceil(roi[2] * W)), int(np.ceil(roi[3] * H))
    e = np.zeros(n, np.float64)
    if n < 2 or x1 <= x0 or y1 <= y0:
        return e
    win = cv2.createHanningWindow((W, H), cv2.CV_32F)
    f = gray.astype(np.float32)
    diffs = np.empty((n - 1, y1 - y0, x1 - x0), np.float32)
    for i in range(1, n):
        prev, cur = f[i - 1], f[i]
        if p.motion_compensation:
            # copies: this OpenCV build applies the window to its inputs in place
            (dx, dy), _resp = cv2.phaseCorrelate(prev.copy(), cur.copy(), win)
            if abs(dx) < 0.25 * W and abs(dy) < 0.25 * H:
                # shift both frames half-way so both get the same interpolation blur
                Mp = np.float32([[1, 0, dx / 2], [0, 1, dy / 2]])
                Mc = np.float32([[1, 0, -dx / 2], [0, 1, -dy / 2]])
                prev = cv2.warpAffine(prev, Mp, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
                cur = cv2.warpAffine(cur, Mc, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        diffs[i - 1] = cur[y0:y1, x0:x1] - prev[y0:y1, x0:x1]
    diffs = np.abs(diffs - np.median(diffs.reshape(n - 1, -1), axis=1)[:, None, None])
    noise = float(np.median(diffs)) * 1.4826 / 0.6745  # median |d| -> sigma of d (Gaussian)
    tau = max(p.diff_threshold, 3.0 * noise)
    e[1:] = np.clip(diffs - tau, 0, None).reshape(n - 1, -1).mean(axis=1) / 255.0
    k = p.median_taps
    if k > 1 and n > k:
        pad = np.pad(e[1:], (k // 2, k // 2), mode="edge")
        e[1:] = np.median(np.lib.stride_tricks.sliding_window_view(pad, k), axis=1)
    return e


def figure_fraction(grass_mean: np.ndarray, dirt_mean: np.ndarray, roi: list[float]) -> float:
    """Share of the ROI that is neither grass nor dirt (the pitcher, roughly), from the segment-mean colour maps."""
    H, W = grass_mean.shape
    x0, y0, x1, y1 = int(roi[0] * W), int(roi[1] * H), int(np.ceil(roi[2] * W)), int(np.ceil(roi[3] * H))
    g, dm = grass_mean[y0:y1, x0:x1], dirt_mean[y0:y1, x0:x1]
    return float(np.clip(1.0 - (g + dm), 0, 1).mean()) if g.size else 0.0


def _smooth(x: np.ndarray, w: int) -> np.ndarray:
    if w <= 1:
        return x.copy()
    k = np.ones(w) / w
    return np.convolve(np.pad(x, (w // 2, w - 1 - w // 2), mode="edge"), k, mode="valid")


def detect_delivery(e: np.ndarray, fps: float, params: AlignmentParams | None = None,
                    figure_frac: float | None = None) -> dict:
    """Onset/release on one segment's motion series (index 0 = first frame of the segment)."""
    p = params or AlignmentParams()
    n = len(e)
    out: dict = {"found": False}
    w = max(int(round(p.smooth_s * fps)), 1)
    s = _smooth(e, w)
    skip = max(int(round(p.skip_after_cut_s * fps)), 1)
    L = max(int(round(p.still_window_s * fps)), 4)
    if n < skip + L + 4:
        out["reason"] = "segment too short"
        return out
    pk = skip + int(np.argmax(s[skip:]))
    peak = float(s[pk])
    out.update(peak_index=pk, peak_energy=peak)
    rel = peak / max(figure_frac, 0.01) if figure_frac is not None else None
    out["peak_rel"] = rel
    if rel is not None and rel < p.min_peak_rel:
        out["reason"] = f"no delivery-sized motion (relative peak {rel:.4f} < {p.min_peak_rel})"
        return out
    min_gap = int(round(p.min_delivery_s * fps * 0.5))
    last_end = pk - min_gap  # set window must end at least this far before the peak
    starts = list(range(skip, last_end - L + 1))
    if not starts:
        out["reason"] = "no room for a set position before the motion peak"
        return out
    means = np.array([e[a:a + L].mean() for a in starts])
    vmin = float(means.min())
    amin = starts[int(np.argmin(means))]
    wmin = e[amin:amin + L]
    sig_min = float(np.median(np.abs(wmin - np.median(wmin))) * 1.4826)
    tol = max(p.still_tol_frac * (peak - vmin), p.still_tol_sigma * max(sig_min, p.sigma_floor))
    ok = np.where(means <= vmin + tol)[0]
    a0 = starts[int(ok[-1])]  # latest still window
    win = e[a0:a0 + L]
    mu0 = float(np.median(win))
    sig = float(np.median(np.abs(win - mu0)) * 1.4826)
    sig_eff = max(sig, p.sigma_floor, p.sigma_peak_frac * (peak - mu0))
    z = (s - mu0) / sig_eff
    S, alarm, last_zero = 0.0, None, a0
    for t in range(a0, pk + 1):
        S = max(0.0, S + z[t] - p.cusum_k)
        if S == 0.0:
            last_zero = t
        if S > p.cusum_h:
            alarm = t
            break
    if alarm is None:
        out["reason"] = "no change point before the motion peak"
        return out
    onset = last_zero + 1  # first frame of the rise
    # refine backward to the first frame above the set position's noise level (bounded look-back)
    lvl = mu0 + p.refine_sigma * max(sig, p.sigma_floor)
    lim = max(a0, onset - int(round(p.refine_max_s * fps)))
    while onset - 1 > lim and s[onset - 1] > lvl:
        onset -= 1
    # stillness of the set position: share of frames under the CUSUM drift level in the window before the onset
    ws = max(skip, onset - int(round(p.still_check_s * fps)))
    still_frac = float(np.mean(z[ws:onset] < p.cusum_k)) if onset > ws else 0.0
    still_span_s = (onset - ws) / fps
    snr = (peak - mu0) / max(sig, p.sigma_floor)
    out.update(found=True, onset_index=onset, alarm_index=alarm, set_window=[a0, a0 + L], baseline=mu0,
               baseline_sigma=sig, sigma_eff=sig_eff, snr=float(snr), still_frac=still_frac, still_span_s=still_span_s,
               smooth_w=w, release_index=pk)
    return out


def detect_scoreboard(gray: np.ndarray, cuts: list[int], fps: float) -> dict | None:
    """Static high-edge rectangle near a border, persisting through the clip (score bug). No OCR."""
    n, H, W = gray.shape
    idx = np.linspace(0, n - 1, min(n, 90)).astype(int)
    st = gray[idx].astype(np.float32)
    tstd = st.std(axis=0)
    med = np.median(st, axis=0).astype(np.uint8)
    edges = np.abs(cv2.Sobel(med, cv2.CV_32F, 1, 0, ksize=3)) + np.abs(cv2.Sobel(med, cv2.CV_32F, 0, 1, ksize=3))
    cand = ((tstd < 3.0) & (edges > 60)).astype(np.uint8)
    band = np.zeros_like(cand)
    band[: int(0.3 * H)] = 1
    band[int(0.7 * H):] = 1
    cand &= band
    cand = cv2.morphologyEx(cand, cv2.MORPH_CLOSE, np.ones((5, 9), np.uint8))
    k, lab, stats, _ = cv2.connectedComponentsWithStats(cand, connectivity=8)
    best = None
    for j in range(1, k):
        x, y, w, h, a = stats[j]
        frac = w * h / (W * H)
        if not (0.004 <= frac <= 0.12) or w < 2 * h:
            continue
        fill = a / max(w * h, 1)
        if fill < 0.35:
            continue
        sc = fill * min(1.0, frac / 0.01)
        if best is None or sc > best[0]:
            best = (sc, x, y, w, h, fill)
    if best is None:
        return None
    sc, x, y, w, h, fill = best
    persistent_across_cuts = len(cuts) > 0
    return {"box": [round(x / W, 4), round(y / H, 4), round((x + w) / W, 4), round((y + h) / H, 4)],
            "fill": round(float(fill), 3),
            "confidence": "candidate" if persistent_across_cuts else "low (no scene cut to confirm persistence)",
            "ocr": None, "note": "region detection only; no OCR and no guarantee it is a scoreboard"}


# ------------------------------------------------------------------ top level
def _ramp(x, lo, hi):
    return float(np.clip((x - lo) / (hi - lo), 0.0, 1.0))


def align_decoded(dec: Decoded, params: AlignmentParams | None = None) -> AlignmentResult:
    p = params or AlignmentParams()
    fps = dec.fps
    res = AlignmentResult(fps=fps, n_frames=dec.n_frames, clip_duration=dec.duration)
    if dec.truncated:
        res.reasons.append(f"clip truncated to the first {p.max_seconds:.0f} s for analysis")
    cuts = detect_scene_cuts(dec.hists, fps, p)
    res.scene_cuts = [round(c / fps, 4) for c in cuts]
    bounds = [0] + cuts + [dec.n_frames]
    best = None
    for si in range(len(bounds) - 1):
        a, b = bounds[si], bounds[si + 1]
        seg = {"index": si, "start": round(a / fps, 4), "end": round(b / fps, 4)}
        if (b - a) / fps < p.min_segment_s:
            seg.update(view="too_short", score=0.0)
            res.segments.append(seg)
            continue
        vc = classify_view(dec.grass[a:b].mean(axis=0), dec.dirt[a:b].mean(axis=0), p)
        seg.update(view=vc["view"], score=vc["score"], cues=vc["cues"])
        if vc["view"] == "center_field":
            roi = pitcher_roi(dec.dirt[a:b].mean(axis=0), p)
            e = motion_energy(dec.gray[a:b], roi, p)
            fig = figure_fraction(dec.grass[a:b].mean(axis=0), dec.dirt[a:b].mean(axis=0), roi)
            det = detect_delivery(e, fps, p, figure_frac=fig)
            seg["delivery"] = {k: v for k, v in det.items() if k in ("found", "reason", "snr", "peak_energy", "peak_rel")}
            if det["found"]:
                onset = (a + det["onset_index"]) / fps
                release = (a + det["release_index"]) / fps
                pause = onset - p.pause_margin_s
                delivery_s = release - onset
                comp = {
                    "view": vc["score"],
                    "peak_snr": _ramp(det["snr"], 5.0, 20.0),
                    "set_stillness": _ramp(det["still_frac"], 0.5, 0.85) * _ramp(det["still_span_s"], 0.2, 0.5),
                    "delivery_timing": 1.0 if p.min_delivery_s <= delivery_s <= p.max_delivery_s else
                    _ramp(delivery_s, p.min_delivery_s * 0.5, p.min_delivery_s) if delivery_s < p.min_delivery_s
                    else _ramp(-delivery_s, -p.max_delivery_s * 1.5, -p.max_delivery_s),
                    "pause_room": _ramp(pause - a / fps, 0.0, 0.3),
                }
                conf = float(min(comp.values()))
                cand = dict(si=si, a=a, b=b, onset=onset, release=release, pause=pause, conf=conf, comp=comp,
                            roi=roi, e=e, det=det)
                if best is None or conf > best["conf"]:
                    best = cand
        res.segments.append(seg)
    if p.detect_scoreboard:
        try:
            res.scoreboard_region = detect_scoreboard(dec.gray, cuts, fps)
        except Exception as ex:  # noqa: BLE001 - optional diagnostic must never fail alignment
            res.reasons.append(f"scoreboard detection failed: {ex}")
    if best is None:
        if not any(s.get("view") == "center_field" for s in res.segments):
            res.reasons.append("no center-field (pitcher/batter) view found")
        else:
            res.reasons.append("center-field view found but no delivery detected: " + "; ".join(
                s["delivery"].get("reason", "") for s in res.segments if "delivery" in s))
        return res
    a, b, fps_ = best["a"], best["b"], fps
    res.chosen_segment = best["si"]
    res.clip_start_time = round(a / fps_, 4)
    res.onset_time = round(best["onset"], 4)
    res.release_time = round(best["release"], 4)
    res.pre_pitch_pause_time = round(max(best["pause"], 0.0), 4)
    res.pitch_end_time = round(min(best["release"] + p.ball_flight_s, b / fps_), 4)
    res.confidence_components = {k: round(v, 4) for k, v in best["comp"].items()}
    res.alignment_confidence = round(best["conf"], 4)
    res.roi = [round(x, 4) for x in best["roi"]]
    det = best["det"]
    step = max(1, len(best["e"]) // 600)
    res.motion = {"segment_start": round(a / fps_, 4), "fps": fps_, "step": step,
                  "energy": [round(float(x), 5) for x in best["e"][::step]],
                  **{k: det[k] for k in ("onset_index", "release_index", "alarm_index", "set_window", "baseline",
                                         "baseline_sigma", "snr", "peak_energy", "peak_rel", "still_frac")}}
    if best["conf"] >= p.min_confidence:
        res.status = "auto_aligned"
    else:
        weakest = min(best["comp"], key=best["comp"].get)
        res.reasons.append(f"low confidence {best['conf']:.2f} < {p.min_confidence} (weakest: {weakest})")
    return res


def align_file(path: str | Path, params: AlignmentParams | None = None) -> AlignmentResult:
    """Run the full detector on a local file. Never raises: decode failures give an 'unaligned' result."""
    p = params or AlignmentParams()
    try:
        dec = decode(path, p)
    except Exception as ex:  # noqa: BLE001 - corrupt/unsupported media -> unaligned with the error recorded
        r = AlignmentResult(error=f"{type(ex).__name__}: {ex}"[:500])
        r.reasons.append("could not decode the clip")
        return r
    return align_decoded(dec, p)
