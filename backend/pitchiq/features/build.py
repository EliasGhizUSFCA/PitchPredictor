"""Leakage-safe pre-pitch feature reconstruction.

Design: a single function `make_features(ctx, hist, game, pa, classes)` builds the feature dict for
one pitch. Its inputs are
  * `ctx`  – the pre-pitch situation of the target pitch (count, inning, outs, runners, score...),
  * `hist` – counters over games strictly before the current game,
  * `game` – running state of the current game, containing only pitches BEFORE the target,
  * `pa`   – running state of the current plate appearance, BEFORE the target.
Historical reconstruction (`build_feature_rows`) and the simulator both call it, so there is one code
path, and the label/post-pitch measurements of the target pitch are only read AFTER its features
have been produced (to update the running state for the next pitch).
"""
from __future__ import annotations

from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from datetime import date

import numpy as np
import pandas as pd

from ..pitch_types import result_category
from . import v2
from .schema import NONE_TOKEN

SMOOTH_GAME = 10.0     # pseudo-pitches toward the prior for in-game usage
SMOOTH_BVP = 15.0      # pseudo-pitches toward the prior for batter-vs-pitcher usage
SMOOTH_COND = 20.0     # pseudo-pitches toward the prior for count/transition priors
RECENT_GAMES = 3

# Columns of the target row that are allowed to be read before its features are produced.
PRE_PITCH_COLUMNS = [
    "game_pk", "game_date", "season", "batter", "stand", "balls", "strikes", "inning", "inning_topbot",
    "outs_when_up", "on_1b", "on_2b", "on_3b", "bat_score", "fld_score", "at_bat_number", "pitch_number",
    "n_thruorder_pitcher", "bat_order_slot", "days_rest",
    "catcher_id",  # v2: the catcher for the pitch is known before it is thrown
]
# Post-pitch measurements of a thrown pitch read only AFTER its features are frozen (to update state for later pitches).
POST_PITCH_LOCATION = ("plate_x", "plate_z", "zone")


def count_key(b: int, s: int) -> str:
    return f"{int(b)}-{int(s)}"


def _norm(counter: Counter, classes: list[str]) -> dict[str, float]:
    n = sum(counter[c] for c in classes)
    if n == 0:
        return {c: 1.0 / len(classes) for c in classes}
    return {c: counter[c] / n for c in classes}


def _smoothed(counter: Counter, prior: dict[str, float], k: float, classes: list[str]) -> dict[str, float]:
    n = sum(counter[c] for c in classes)
    return {c: (counter[c] + k * prior[c]) / (n + k) for c in classes}


@dataclass
class History:
    """Counters over games strictly before the current game."""
    overall: Counter = field(default_factory=Counter)
    by_count: dict = field(default_factory=lambda: defaultdict(Counter))
    by_prev: dict = field(default_factory=lambda: defaultdict(Counter))
    by_batter: dict = field(default_factory=lambda: defaultdict(Counter))
    recent_games: deque = field(default_factory=lambda: deque(maxlen=RECENT_GAMES))
    last_game_date: date | None = None
    # ---- v2 (features/v2.py)
    game_log: list = field(default_factory=list)
    velo_sum: Counter = field(default_factory=Counter)
    velo_n: Counter = field(default_factory=Counter)
    by_catcher: dict = field(default_factory=lambda: defaultdict(Counter))
    bvp_swings: dict = field(default_factory=lambda: defaultdict(Counter))
    bvp_whiffs: dict = field(default_factory=lambda: defaultdict(Counter))
    swings: Counter = field(default_factory=Counter)
    whiffs: Counter = field(default_factory=Counter)
    batter_priors: object = None  # batter_priors.BatterPriors (MLB-wide priors from seasons before the pitch)
    ewm_half_life_days: float = v2.EWM_HALF_LIFE_DAYS
    window_cache: dict = field(default_factory=dict)

    def overall_prior(self, classes):
        return _norm(self.overall, classes)

    def absorb_game(self, game: "GameState", game_date: date) -> None:
        self.overall.update(game.usage)
        for k, c in game.count_events:
            self.by_count[k][c] += 1
        for prev, c in game.transitions:
            self.by_prev[prev][c] += 1
        for b, cnt in game.by_batter.items():
            self.by_batter[b].update(cnt)
        self.recent_games.append(Counter(game.usage))
        self.last_game_date = game_date
        v2.absorb(self, game, game_date)


@dataclass
class GameState:
    """Running state of the current game (only pitches already thrown)."""
    types: list = field(default_factory=list)        # thrown pitch classes in order
    results: list = field(default_factory=list)
    speeds: list = field(default_factory=list)
    ab_numbers: list = field(default_factory=list)
    ff_speeds: list = field(default_factory=list)
    usage: Counter = field(default_factory=Counter)
    by_batter: dict = field(default_factory=lambda: defaultdict(Counter))
    count_events: list = field(default_factory=list)
    transitions: list = field(default_factory=list)
    # ---- v2
    speeds_measured: list = field(default_factory=list)  # measured velocity or None (never imputed)
    plate_x: list = field(default_factory=list)
    plate_z: list = field(default_factory=list)
    zones: list = field(default_factory=list)
    by_catcher: dict = field(default_factory=lambda: defaultdict(Counter))
    bvp_swings: dict = field(default_factory=lambda: defaultdict(Counter))
    bvp_whiffs: dict = field(default_factory=lambda: defaultdict(Counter))
    swings: Counter = field(default_factory=Counter)
    whiffs: Counter = field(default_factory=Counter)

    @property
    def n(self) -> int:
        return len(self.types)


@dataclass
class PAState:
    usage: Counter = field(default_factory=Counter)


def make_features(ctx: dict, hist: History, game: GameState, pa: PAState, classes: list[str]) -> dict:
    prior = hist.overall_prior(classes)
    balls, strikes = int(ctx["balls"]), int(ctx["strikes"])
    ck = count_key(balls, strikes)
    prev = lambda k: game.types[-k] if game.n >= k else NONE_TOKEN  # noqa: E731
    f: dict = {
        "stand": ctx["stand"],
        "bat_order_slot": ctx.get("bat_order_slot"),
        "balls": balls,
        "strikes": strikes,
        "count_str": ck,
        "inning": int(ctx["inning"]),
        "pitch_count_before": ctx.get("pitch_count_before", game.n),  # simulator may override
        "pa_pitch_number": int(ctx["pitch_number"]),
        "times_through_order": ctx.get("n_thruorder_pitcher"),
        "outs": int(ctx["outs_when_up"]),
        "on_1b": int(bool(ctx.get("on_1b"))),
        "on_2b": int(bool(ctx.get("on_2b"))),
        "on_3b": int(bool(ctx.get("on_3b"))),
        "score_diff": (ctx.get("fld_score") or 0) - (ctx.get("bat_score") or 0),
        "is_home": int(ctx["inning_topbot"] == "Top"),
        "prev1_type": prev(1),
        "prev2_type": prev(2),
        "prev3_type": prev(3),
        "prev1_same_pa": int(game.n > 0 and game.ab_numbers[-1] == ctx["at_bat_number"]),
        "prev1_result": game.results[-1] if game.n else NONE_TOKEN,
        "prev1_speed": game.speeds[-1] if game.n else None,
        "game_ff_velo_mean": float(np.mean(game.ff_speeds)) if game.ff_speeds else None,
        "season": int(ctx["season"]),
        "days_rest": ctx.get("days_rest"),
    }
    # In-game usage, smoothed toward pre-game career prior.
    for c, v in _smoothed(game.usage, prior, SMOOTH_GAME, classes).items():
        f[f"game_share_{c}"] = v
    recent = Counter()
    for g in hist.recent_games:
        recent.update(g)
    for c, v in _norm(recent, classes).items() if recent else prior.items():
        f[f"recent_share_{c}"] = v
    for c, v in _smoothed(hist.by_count[ck], prior, SMOOTH_COND, classes).items():
        f[f"count_prior_{c}"] = v
    for c, v in _smoothed(hist.by_prev[prev(1)], prior, SMOOTH_COND, classes).items():
        f[f"trans_prior_{c}"] = v
    bvp = hist.by_batter[ctx["batter"]] + game.by_batter[ctx["batter"]]
    f["bvp_n"] = sum(bvp[c] for c in classes)
    for c, v in _smoothed(bvp, prior, SMOOTH_BVP, classes).items():
        f[f"bvp_share_{c}"] = v
    for c in classes:
        f[f"pa_n_{c}"] = pa.usage[c]
    f.update(v2.features(ctx, hist, game, pa, classes, f))  # v2 additions; v1 keys/values unchanged
    return f


def update_after_pitch(game: GameState, pa: PAState, ctx: dict, label: str | None, speed, description,
                       location: tuple | None = None) -> None:
    """Called only AFTER the target pitch's features have been frozen. `location` = (plate_x, plate_z, zone)."""
    if label is None:  # automatic ball/strike (pitch clock): changes count, but no pitch thrown
        return
    ck = count_key(ctx["balls"], ctx["strikes"])
    prev = game.types[-1] if game.n else NONE_TOKEN
    game.count_events.append((ck, label))
    game.transitions.append((prev, label))
    game.types.append(label)
    result = result_category(description)
    game.results.append(result)
    measured = None if speed is None or pd.isna(speed) else float(speed)
    game.speeds.append(measured)
    game.ab_numbers.append(ctx["at_bat_number"])
    if label == "FF" and speed is not None and not pd.isna(speed):
        game.ff_speeds.append(float(speed))
    game.usage[label] += 1
    game.by_batter[ctx["batter"]][label] += 1
    pa.usage[label] += 1
    v2.update(game, ctx, label, measured, result, location)


def _location(row: dict) -> tuple:
    return tuple(row.get(k) for k in POST_PITCH_LOCATION)


def map_label(pitch_type, classes: list[str]) -> str | None:
    if pitch_type is None or (isinstance(pitch_type, float) and np.isnan(pitch_type)) or pitch_type == "":
        return None
    return pitch_type if pitch_type in classes else "OTHER"


def build_feature_rows(df: pd.DataFrame, classes: list[str], batter_priors=None) -> list[dict]:
    """df: one pitcher's pitches with PRE_PITCH_COLUMNS + pitch_type/release_speed/description (+ plate_x/plate_z/zone).

    Returns one dict per row: {pitch_uid, label, features} (label None for non-pitch rows). `batter_priors`
    (batter_priors.BatterPriors) supplies MLB-wide batter priors from seasons before each pitch's season.
    """
    df = df.sort_values(["game_date", "game_pk", "at_bat_number", "pitch_number"])
    hist = History(batter_priors=batter_priors)
    out = []
    for gpk, g in df.groupby("game_pk", sort=False):
        game = GameState()
        pa, pa_ab = PAState(), None
        game_date = g["game_date"].iloc[0]
        for row in g.to_dict("records"):
            if row["at_bat_number"] != pa_ab:
                pa, pa_ab = PAState(), row["at_bat_number"]
            ctx = {k: row.get(k) for k in PRE_PITCH_COLUMNS}
            feats = make_features(ctx, hist, game, pa, classes)
            label = map_label(row.get("pitch_type"), classes)
            out.append({"pitch_uid": row.get("pitch_uid"), "label": label, "features": feats})
            update_after_pitch(game, pa, ctx, label, row.get("release_speed"), row.get("description"),
                               _location(row))
        hist.absorb_game(game, game_date)
    return out


def history_as_of(df: pd.DataFrame, classes: list[str], before: date, batter_priors=None) -> History:
    """History counters from all games strictly before `before` (used by the simulator)."""
    df = df[df["game_date"] < before].sort_values(["game_date", "game_pk", "at_bat_number", "pitch_number"])
    hist = History(batter_priors=batter_priors)
    for _, g in df.groupby("game_pk", sort=False):
        game, pa, pa_ab = GameState(), PAState(), None
        for row in g.to_dict("records"):
            if row["at_bat_number"] != pa_ab:
                pa, pa_ab = PAState(), row["at_bat_number"]
            ctx = {k: row.get(k) for k in PRE_PITCH_COLUMNS}
            update_after_pitch(game, pa, ctx, map_label(row.get("pitch_type"), classes),
                               row.get("release_speed"), row.get("description"), _location(row))
        hist.absorb_game(game, g["game_date"].iloc[0])
    return hist
