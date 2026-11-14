"""Synthetic broadcast-like pitch clips with known ground truth, for validating pitchiq.video.alignment.

THESE CLIPS ARE SYNTHETIC. No licensed real broadcast footage is available in this project, so the alignment
detector can only be validated here against rendered look-alikes. Accuracy measured on them says how well the
detector recovers the *rendered* events; it is not a measurement on real MLB broadcasts.

What a clip contains (each element optional / randomised per seed):
  lead-in shot (crowd or player close-up)  --hard cut-->  center-field view  --hard cut-->  tail close-up
The center-field view is a camera behind the pitcher looking toward home plate: backstop/crowd band at the top,
a dirt circle at home plate with batter + catcher, mowed infield grass, the mound at the bottom centre with the
pitcher seen from behind. In it the pitcher: stands (looks in, head nods) -> comes set (hands together) -> holds a
static set position (the pre-pitch pause) -> delivers (leg lift with zero initial velocity, stride, arm cock, arm
acceleration to release at maximum hand speed) -> follows through. A ball travels to the catcher after release.
Nuisance: sensor noise, sub-pixel camera jitter, batter bat waggle, a static score bug overlay, H.264 compression
at different CRFs, frame rates (24 .. 60 fps) and resolutions.

Ground truth (seconds from clip start): scene cuts, center-field segment, set start, delivery onset (the instant the
set pose starts to change), release, plate arrival, and the score bug box.

Run the validation (writes a `video_alignment_validations` row and an MLflow run when --store is given):
    python tests/video_synth.py --n-dev 30 --n-eval 60 --store
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import numpy as np
from scipy.interpolate import PchipInterpolator

GENERATOR_VERSION = "synth-v1"
FPS_CHOICES = [24.0, 25.0, 30000 / 1001, 30.0, 50.0, 60000 / 1001, 60.0]
RES_CHOICES = [(480, 270), (640, 360), (854, 480), (960, 540)]
NEGATIVE_KINDS = ["crowd_only", "closeup_only", "outfield_wide", "high_home", "no_delivery"]


@dataclass
class SynthSpec:
    seed: int
    fps: float = 30.0
    width: int = 640
    height: int = 360
    kind: str = "pitch"  # pitch | crowd_only | closeup_only | outfield_wide | high_home | no_delivery
    lead_kind: str = "crowd"  # crowd | closeup | none
    lead_s: float = 1.0
    pre_set_s: float = 0.8  # standing on the rubber, looking in for the sign
    come_set_s: float = 0.5  # hands come together
    set_s: float = 1.2  # static set position (the pre-pitch pause)
    delivery_s: float = 1.0  # onset -> release
    follow_s: float = 0.9  # after release, before the tail cut / clip end
    tail_kind: str = "closeup"  # closeup | none
    tail_s: float = 0.8
    noise_sigma: float = 2.0
    jitter_px: float = 0.3
    batter_waggle: bool = True
    scoreboard: bool = True
    scoreboard_corner: str = "tl"  # tl | tr | bl | br
    crf: int = 23
    zoom: float = 1.0
    x_offset: float = 0.0
    jersey: tuple = (60, 40, 30)  # BGR
    pants: tuple = (200, 200, 200)

    @classmethod
    def random(cls, seed: int, kind: str = "pitch") -> "SynthSpec":
        r = np.random.default_rng(seed)
        w, h = RES_CHOICES[r.integers(len(RES_CHOICES))]
        jerseys = [(60, 40, 30), (150, 150, 150), (40, 40, 160), (200, 200, 205), (90, 60, 20)]
        return cls(
            seed=seed, kind=kind, fps=float(FPS_CHOICES[r.integers(len(FPS_CHOICES))]), width=int(w), height=int(h),
            lead_kind=str(r.choice(["crowd", "closeup", "none"], p=[0.45, 0.35, 0.2])),
            lead_s=float(r.uniform(0.6, 2.0)), pre_set_s=float(r.uniform(0.4, 1.6)),
            come_set_s=float(r.uniform(0.3, 0.7)), set_s=float(r.uniform(0.7, 2.5)),
            delivery_s=float(r.uniform(0.8, 1.25)), follow_s=float(r.uniform(0.7, 1.3)),
            tail_kind=str(r.choice(["closeup", "none"], p=[0.6, 0.4])), tail_s=float(r.uniform(0.5, 1.2)),
            noise_sigma=float(r.uniform(0.0, 5.0)), jitter_px=float(r.uniform(0.0, 0.8)),
            batter_waggle=bool(r.random() < 0.8), scoreboard=bool(r.random() < 0.7),
            scoreboard_corner=str(r.choice(["tl", "tr", "bl", "br"])), crf=int(r.integers(18, 31)),
            zoom=float(r.uniform(0.85, 1.2)), x_offset=float(r.uniform(-0.05, 0.05)),
            jersey=tuple(int(c) for c in jerseys[r.integers(len(jerseys))]),
            pants=tuple(int(c) for c in [(200, 200, 200), (150, 150, 155), (230, 230, 230)][r.integers(3)]),
        )


@dataclass
class GroundTruth:
    spec: dict
    duration: float
    n_frames: int
    scene_cuts: list = field(default_factory=list)  # times (s) of the first frame of each new shot
    has_pitcher_view: bool = True
    has_delivery: bool = True
    cf_start: float | None = None
    cf_end: float | None = None
    set_start: float | None = None
    onset: float | None = None
    release: float | None = None
    plate_arrival: float | None = None
    scoreboard_box: list | None = None  # normalised [x0, y0, x1, y1]


# ------------------------------------------------------------------ rendering helpers
def _noise_texture(r, h, w, base, amp, blur):
    n = r.normal(0, 1, (h, w, 1)).astype(np.float32)
    n = cv2.GaussianBlur(n, (0, 0), blur)[..., None] if blur else n
    n = n / (n.std() + 1e-6)
    return np.clip(np.array(base, np.float32)[None, None, :] + amp * n, 0, 255)


def _field_background(r, w, h, high_home=False, outfield=False):
    img = np.zeros((h, w, 3), np.float32)
    grass = _noise_texture(r, h, w, (40, 120, 45), 6, 2.0)
    stripes = ((np.arange(h)[:, None] // max(h // 12, 1)) % 2).astype(np.float32)
    img[:] = grass + stripes[..., None] * np.array([4, 14, 6], np.float32)
    dirt = _noise_texture(r, h, w, (70, 120, 170), 8, 1.5)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    if outfield:  # wide outfield view: grass, warning track band, wall
        track = (yy > 0.18 * h) & (yy < 0.26 * h)
        img[track] = dirt[track]
        img[yy <= 0.18 * h] = _noise_texture(r, h, w, (50, 70, 40), 10, 1.0)[yy <= 0.18 * h]
        return img
    if high_home:  # camera behind home plate: mound small in the centre, plate dirt at the bottom
        mound = ((xx - 0.5 * w) / (0.07 * w)) ** 2 + ((yy - 0.42 * h) / (0.05 * h)) ** 2 < 1
        plate = ((xx - 0.5 * w) / (0.30 * w)) ** 2 + ((yy - 1.02 * h) / (0.2 * h)) ** 2 < 1
        img[mound | plate] = dirt[mound | plate]
        img[yy <= 0.12 * h] = _noise_texture(r, h, w, (60, 60, 60), 20, 0.8)[yy <= 0.12 * h]
        return img
    top = yy <= 0.17 * h  # backstop / crowd band behind home plate
    img[top] = _noise_texture(r, h, w, (55, 50, 60), 22, 0.7)[top]
    plate = ((xx - 0.5 * w) / (0.17 * w)) ** 2 + ((yy - 0.26 * h) / (0.10 * h)) ** 2 < 1
    img[plate & ~top] = dirt[plate & ~top]
    mound = ((xx - 0.5 * w) / (0.21 * w)) ** 2 + ((yy - 0.93 * h) / (0.14 * h)) ** 2 < 1
    img[mound] = dirt[mound]
    cv2.rectangle(img, (int(0.47 * w), int(0.875 * h)), (int(0.53 * w), int(0.888 * h)), (235, 235, 235), -1)
    cv2.rectangle(img, (int(0.485 * w), int(0.255 * h)), (int(0.515 * w), int(0.27 * h)), (240, 240, 240), -1)
    return img


def _crowd(r, w, h):
    img = _noise_texture(r, h, w, (70, 60, 60), 15, 1.0)
    for _ in range(int(60 * w * h / (640 * 360))):
        c = tuple(float(x) for x in r.integers(20, 235, 3))
        cv2.ellipse(img, (int(r.integers(w)), int(r.integers(h))), (int(r.integers(6, 22)), int(r.integers(8, 30))),
                    0, 0, 360, c, -1)
    return img


def _closeup(r, w, h, t, jersey):
    img = _noise_texture(r, h, w, (90, 100, 110), 10, 3.0)
    cx = int(0.5 * w + 0.03 * w * np.sin(t * 1.3))
    cv2.ellipse(img, (cx, int(1.05 * h)), (int(0.33 * w), int(0.45 * h)), 0, 0, 360, jersey, -1)
    cv2.ellipse(img, (cx, int(0.45 * h)), (int(0.13 * w), int(0.27 * h)), 0, 0, 360, (120, 150, 200), -1)
    cv2.ellipse(img, (cx, int(0.22 * h)), (int(0.15 * w), int(0.09 * h)), 0, 180, 360, jersey, -1)
    cv2.circle(img, (cx - int(0.05 * w), int(0.42 * h)), int(0.012 * w), (30, 30, 30), -1)
    cv2.circle(img, (cx + int(0.05 * w), int(0.42 * h)), int(0.012 * w), (30, 30, 30), -1)
    return img


SHIFT = 4  # cv2 fixed-point sub-pixel drawing
_SC = 1 << SHIFT


def _pt(p):
    return int(round(p[0] * _SC)), int(round(p[1] * _SC))


def _line(img, a, b, color, th):
    cv2.line(img, _pt(a), _pt(b), color, max(int(th), 1), cv2.LINE_AA, SHIFT)


def _circle(img, c, rad, color):
    cv2.circle(img, _pt(c), int(round(rad * _SC)), color, -1, cv2.LINE_AA, SHIFT)


# Pose keyframes (units: figure height; origin hip centre; +y is down the image). Seen from behind the pitcher;
# "lead" leg strides toward the plate (up the image), "arm" is the throwing hand.
POSE_STAND = dict(lead_knee=(0.07, 0.25), lead_foot=(0.08, 0.5), back_knee=(-0.07, 0.25), back_foot=(-0.08, 0.5),
                  arm_elbow=(-0.17, -0.18), arm_hand=(-0.18, 0.0), glove_elbow=(0.17, -0.18), glove=(0.18, 0.0),
                  torso=(0.0, 0.0), head=(0.0, -0.47))
POSE_SET = dict(POSE_STAND, arm_elbow=(-0.13, -0.17), arm_hand=(-0.02, -0.22), glove_elbow=(0.13, -0.17),
                glove=(0.02, -0.22))


def _pose_keys(spec: SynthSpec):
    """(time_offset_from_onset_in_delivery_units, pose) keyframes; u = (t - onset)/delivery_s; release at u=1."""
    lift = dict(POSE_SET, lead_knee=(0.03, 0.03), lead_foot=(0.06, 0.26), arm_hand=(-0.01, -0.27),
                glove=(0.01, -0.27), arm_elbow=(-0.12, -0.2), glove_elbow=(0.12, -0.2), torso=(-0.01, -0.01))
    stride = dict(lift, lead_knee=(0.06, 0.16), lead_foot=(0.04, 0.36), torso=(0.0, 0.04), head=(0.0, -0.43),
                  arm_elbow=(-0.2, -0.05), arm_hand=(-0.26, 0.04), glove_elbow=(0.2, -0.28), glove=(0.26, -0.33))
    cocked = dict(stride, back_knee=(-0.06, 0.3), arm_elbow=(-0.25, -0.33), arm_hand=(-0.27, -0.55),
                  glove_elbow=(0.16, -0.2), glove=(0.14, -0.12), torso=(0.01, 0.06))
    release = dict(cocked, arm_elbow=(-0.13, -0.47), arm_hand=(-0.02, -0.7), head=(0.02, -0.4), torso=(0.02, 0.08))
    follow1 = dict(release, arm_elbow=(0.08, -0.2), arm_hand=(0.2, 0.05), back_knee=(-0.02, 0.16),
                   back_foot=(-0.03, 0.3), torso=(0.03, 0.12), head=(0.03, -0.35), glove=(0.1, -0.05))
    follow2 = dict(follow1, arm_hand=(0.24, 0.12), arm_elbow=(0.12, -0.12), back_foot=(0.0, 0.42),
                   back_knee=(0.0, 0.22), torso=(0.02, 0.08), head=(0.02, -0.4))
    return [(0.0, POSE_SET), (0.45, lift), (0.72, stride), (0.88, cocked), (1.0, release), (1.18, follow1),
            (1.55, follow2)]


class PitcherMotion:
    def __init__(self, spec: SynthSpec, cf_start: float, delivery: bool = True):
        self.spec = spec
        self.t_come = cf_start + spec.pre_set_s
        self.t_set = self.t_come + spec.come_set_s
        self.onset = self.t_set + spec.set_s
        self.release = self.onset + spec.delivery_s
        self.delivery = delivery
        keys = _pose_keys(spec)
        ts = np.array([self.onset - 1.0] + [self.onset + u * spec.delivery_s for u, _ in keys]
                      + [self.onset + 3.0 * spec.delivery_s])
        poses = [POSE_SET] + [p for _, p in keys] + [keys[-1][1]]
        self.interp = {k: PchipInterpolator(ts, np.array([p[k] for p in poses], float), axis=0) for k in POSE_SET}

    def pose(self, t):
        s = self.spec
        if t < self.t_come or (not self.delivery and t < self.t_come):
            p = dict(POSE_STAND)
            nod = 0.012 * np.sin(2 * np.pi * 1.5 * (t - self.t_come)) if t > self.t_come - 0.6 else 0.0
            p["head"] = (0.0, -0.47 + nod)
            return p
        if t < self.t_set:  # coming set: smooth move of the hands to the chest
            a = (t - self.t_come) / s.come_set_s
            a = a * a * (3 - 2 * a)
            return {k: tuple((1 - a) * np.array(POSE_STAND[k]) + a * np.array(POSE_SET[k])) for k in POSE_SET}
        if not self.delivery or t <= self.onset:
            return dict(POSE_SET)
        return {k: tuple(self.interp[k](t)) for k in POSE_SET}


def _draw_pitcher(img, pose, cx, cy, H, spec: SynthSpec):
    def P(k):
        x, y = pose[k]
        tx, ty = pose["torso"]
        if k in ("head", "arm_elbow", "arm_hand", "glove_elbow", "glove"):
            x, y = x + tx, y + ty
        return np.array([cx + x * H, cy + y * H])
    hip = np.array([cx + pose["torso"][0] * H, cy + pose["torso"][1] * H])
    sh_l, sh_r = hip + np.array([-0.12 * H, -0.36 * H]), hip + np.array([0.12 * H, -0.36 * H])
    th = 0.075 * H
    for knee, foot, hipx in (("back_knee", "back_foot", -0.06), ("lead_knee", "lead_foot", 0.06)):
        hp = hip + np.array([hipx * H, 0.02 * H])
        _line(img, hp, P(knee), spec.pants, th)
        _line(img, P(knee), P(foot), spec.pants, th * 0.85)
        _circle(img, P(foot) + np.array([0, 0.02 * H]), 0.035 * H, (25, 25, 25))
    pts = np.array([sh_l, sh_r, hip + np.array([0.09 * H, 0.03 * H]), hip + np.array([-0.09 * H, 0.03 * H])])
    cv2.fillConvexPoly(img, np.round(pts * _SC).astype(np.int32), spec.jersey, cv2.LINE_AA, SHIFT)
    _line(img, sh_l, P("arm_elbow"), spec.jersey, th * 0.7)
    _line(img, P("arm_elbow"), P("arm_hand"), (110, 140, 190), th * 0.55)
    _circle(img, P("arm_hand"), 0.025 * H, (120, 150, 200))
    _line(img, sh_r, P("glove_elbow"), spec.jersey, th * 0.7)
    _line(img, P("glove_elbow"), P("glove"), (110, 140, 190), th * 0.55)
    _circle(img, P("glove"), 0.045 * H, (30, 50, 80))
    _circle(img, P("head"), 0.065 * H, (35, 35, 35))  # cap from behind


def _draw_batter(img, w, h, t, waggle):
    bx, by = 0.44 * w, 0.24 * h
    H = 0.2 * h
    _line(img, (bx, by - 0.1 * H), (bx, by + 0.45 * H), (210, 210, 210), 0.12 * H)
    _circle(img, (bx, by - 0.2 * H), 0.08 * H, (40, 40, 40))
    ang = (0.6 + 0.25 * np.sin(2 * np.pi * 1.1 * t)) if waggle else 0.6
    hand = np.array([bx + 0.05 * H, by - 0.05 * H])
    _line(img, hand, hand + 0.5 * H * np.array([np.cos(np.pi / 2 + ang), -np.sin(np.pi / 2 + ang)]),
          (90, 150, 190), 0.05 * H)
    cx, cy = 0.5 * w, 0.29 * h  # catcher (crouched) + umpire
    _circle(img, (cx, cy), 0.07 * H * 1.4, (50, 50, 50))
    _circle(img, (cx + 0.02 * w, cy - 0.06 * h), 0.08 * H * 1.4, (30, 30, 30))


def _scoreboard(img, w, h, corner):
    bw, bh = int(0.24 * w), int(0.09 * h)
    x0 = int(0.03 * w) if corner in ("tl", "bl") else w - bw - int(0.03 * w)
    y0 = int(0.04 * h) if corner in ("tl", "tr") else h - bh - int(0.04 * h)
    cv2.rectangle(img, (x0, y0), (x0 + bw, y0 + bh), (25, 20, 20), -1)
    cv2.putText(img, "BOS 2  NYY 1", (x0 + 4, y0 + int(bh * 0.45)), cv2.FONT_HERSHEY_SIMPLEX, bh / 70, (240, 240, 240),
                1, cv2.LINE_AA)
    cv2.putText(img, "TOP 5  1-2  1 OUT", (x0 + 4, y0 + int(bh * 0.9)), cv2.FONT_HERSHEY_SIMPLEX, bh / 80,
                (200, 230, 255), 1, cv2.LINE_AA)
    return [x0 / w, y0 / h, (x0 + bw) / w, (y0 + bh) / h]


# ------------------------------------------------------------------ clip generation
def generate_clip(spec: SynthSpec, out_path: Path) -> GroundTruth:
    """Render the clip described by `spec` to an H.264 MP4 at out_path and return its ground truth."""
    r = np.random.default_rng(spec.seed + 1000)
    w, h, fps = spec.width, spec.height, spec.fps
    kind = spec.kind
    shots: list[tuple[str, float]] = []
    lead = spec.lead_kind if kind == "pitch" or kind in ("no_delivery", "high_home") else "none"
    if lead != "none":
        shots.append((lead, spec.lead_s))
    if kind in ("pitch", "no_delivery", "high_home"):
        main_len = spec.pre_set_s + spec.come_set_s + spec.set_s + spec.delivery_s + spec.follow_s
        if kind == "no_delivery":
            main_len = spec.pre_set_s + spec.come_set_s + spec.set_s + 1.5
        shots.append(("high_home" if kind == "high_home" else "cf", main_len))
    elif kind == "crowd_only":
        shots.append(("crowd", 4.0))
    elif kind == "closeup_only":
        shots.append(("closeup", 4.0))
    elif kind == "outfield_wide":
        shots.append(("outfield", 4.0))
    if kind == "pitch" and spec.tail_kind != "none":
        shots.append(("closeup" if spec.lead_kind != "closeup" else "crowd", spec.tail_s))
    bounds, t = [], 0.0
    for name, d in shots:
        bounds.append((name, t, t + d))
        t += d
    n_frames = int(round(t * fps))
    gt = GroundTruth(spec=asdict(spec), duration=n_frames / fps, n_frames=n_frames,
                     has_pitcher_view=kind in ("pitch", "no_delivery"), has_delivery=kind == "pitch")
    # a cut is the first frame whose timestamp falls in the next shot
    shot_of_frame = []
    for i in range(n_frames):
        tt = i / fps
        shot_of_frame.append(next((j for j, (_, a, b) in enumerate(bounds) if a <= tt < b), len(bounds) - 1))
    for i in range(1, n_frames):
        if shot_of_frame[i] != shot_of_frame[i - 1]:
            gt.scene_cuts.append(i / fps)
    cf = next(((a, b) for n, a, b in bounds if n == "cf"), None)
    motion = None
    if cf is not None:
        # the CF segment starts at the first frame inside it
        first = next(i for i in range(n_frames) if shot_of_frame[i] == [n for n, _, _ in bounds].index("cf"))
        last = max(i for i in range(n_frames) if shot_of_frame[i] == [n for n, _, _ in bounds].index("cf"))
        gt.cf_start, gt.cf_end = first / fps, (last + 1) / fps
        motion = PitcherMotion(spec, cf[0], delivery=(kind == "pitch"))
        gt.set_start = motion.t_set
        if kind == "pitch":
            gt.onset, gt.release = motion.onset, motion.release
            gt.plate_arrival = motion.release + 0.42
    hh = next(((a, b) for n, a, b in bounds if n == "high_home"), None)
    if hh is not None:
        motion = PitcherMotion(spec, hh[0], delivery=True)
    bgs = {"cf": _field_background(r, w, h), "high_home": _field_background(r, w, h, high_home=True),
           "outfield": _field_background(r, w, h, outfield=True), "crowd": _crowd(r, w, h)}
    cmd = ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}",
           "-r", f"{fps:.6f}", "-i", "-", "-c:v", "libx264", "-preset", "veryfast", "-crf", str(spec.crf),
           "-pix_fmt", "yuv420p", "-g", str(int(round(fps * 2))), str(out_path)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    jit = np.zeros(2)
    H = 0.42 * h * spec.zoom
    sb_box = None
    try:
        for i in range(n_frames):
            tt = i / fps
            name, a, b = bounds[shot_of_frame[i]]
            if name in ("cf", "high_home", "outfield"):
                img = bgs[name].copy()
                if name == "cf":
                    _draw_batter(img, w, h, tt, spec.batter_waggle)
                    cx, cy = (0.5 + spec.x_offset) * w, 0.865 * h - 0.5 * H
                    _draw_pitcher(img, motion.pose(tt), cx, cy, H, spec)
                    if gt.release is not None and gt.release < tt < gt.plate_arrival:
                        f = (tt - gt.release) / 0.42
                        bx = cx - 0.02 * H + f * (0.5 * w - cx)
                        by = (cy - 0.7 * H) * (1 - f) + 0.29 * h * f
                        _circle(img, (bx, by), max(0.006 * w * (1 - 0.6 * f), 1.0), (245, 245, 245))
                elif name == "high_home":
                    _draw_pitcher(img, motion.pose(tt), 0.5 * w, 0.42 * h - 0.06 * h, 0.13 * h, spec)
                else:
                    for k in range(3):
                        _circle(img, ((0.2 + 0.3 * k) * w + 30 * np.sin(tt * (1 + k)), 0.5 * h), 0.02 * h,
                                (200, 200, 200))
                jit = 0.85 * jit + r.normal(0, spec.jitter_px, 2) * 0.55
                M = np.float32([[1, 0, jit[0]], [0, 1, jit[1]]])
                img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
            elif name == "crowd":
                M = np.float32([[1, 0, 8 * (tt - a)], [0, 1, 0]])
                img = cv2.warpAffine(bgs["crowd"], M, (w, h), borderMode=cv2.BORDER_REFLECT)
            else:
                img = _closeup(r, w, h, tt, spec.jersey)
            if spec.noise_sigma > 0:
                img = img + r.normal(0, spec.noise_sigma, img.shape).astype(np.float32)
            img = np.clip(img, 0, 255).astype(np.uint8)
            if spec.scoreboard and kind == "pitch":
                sb_box = _scoreboard(img, w, h, spec.scoreboard_corner)
            proc.stdin.write(img.tobytes())
    finally:
        proc.stdin.close()
        proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed for seed {spec.seed}")
    gt.scoreboard_box = sb_box
    return gt


def make_set(out_dir: Path, seeds, negatives_per_kind: int = 0, neg_seed0: int = 900000) -> list[tuple[Path, GroundTruth]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    items = []
    for s in seeds:
        p = out_dir / f"pitch_{s}.mp4"
        items.append((p, generate_clip(SynthSpec.random(int(s)), p)))
    k = 0
    for kind in NEGATIVE_KINDS:
        for j in range(negatives_per_kind):
            sd = neg_seed0 + k
            k += 1
            p = out_dir / f"{kind}_{sd}.mp4"
            items.append((p, generate_clip(SynthSpec.random(sd, kind=kind), p)))
    return items


# ------------------------------------------------------------------ validation runner
def _run(n_dev: int, n_eval: int, neg_per_kind: int, store: bool, out_dir: Path, workers: int) -> dict:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from pitchiq.video.alignment import DETECTOR_VERSION, AlignmentParams
    from pitchiq.video.alignment_eval import evaluate_clips

    params = AlignmentParams()
    # Seed ranges are disjoint: development seeds were the only ones looked at while choosing the detector's
    # thresholds/margin; evaluation seeds are a fresh draw from the same generator.
    dev_seeds = list(range(1, 1 + n_dev))
    eval_seeds = list(range(10001, 10001 + n_eval))
    reports = {}
    for role, seeds, negs in (("development", dev_seeds, max(1, neg_per_kind // 2)),
                              ("evaluation", eval_seeds, neg_per_kind)):
        if not seeds:
            continue
        items = make_set(out_dir / role, seeds, negs, neg_seed0=900000 if role == "development" else 950000)
        rep = evaluate_clips(items, params, workers=workers)
        rep["role"] = role
        rep["seeds"] = {"pitch": [seeds[0], seeds[-1]], "negatives_per_kind": negs}
        reports[role] = rep
        print(json.dumps({"role": role, **rep["summary"]}, indent=2, default=str))
    if store:
        from pitchiq.db import session_scope
        from pitchiq.video.alignment_eval import store_validation
        with session_scope() as s:
            for role, rep in reports.items():
                row = store_validation(s, rep, role=role, generator=f"tests/video_synth.py:{GENERATOR_VERSION}",
                                       params=params)
                print(f"stored video_alignment_validations id={row.validation_id} role={role} "
                      f"detector={DETECTOR_VERSION} mlflow_run_id={row.mlflow_run_id}")
    return reports


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-dev", type=int, default=30)
    ap.add_argument("--n-eval", type=int, default=60)
    ap.add_argument("--neg-per-kind", type=int, default=2, help="negative clips per kind in the evaluation set")
    ap.add_argument("--out", type=Path, default=Path("data/video_synth"))
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--store", action="store_true", help="store results in Postgres + MLflow")
    a = ap.parse_args()
    _run(a.n_dev, a.n_eval, a.neg_per_kind, a.store, a.out, a.workers)
