"""Feature version v2: additions computed by `build.make_features` on top of the v1 dictionary.

All inputs come from `History` (games strictly before the current game, plus MLB-wide batter priors from seasons
before the current season), `GameState` / `PAState` (pitches already thrown in this game / plate appearance) and the
pre-pitch context. v1 keys and values are untouched, so a v2 row restricted to the v1 keys equals the v1 row.

Groups (for ablation):

* ``explicit_flags``: is_first_pitch_of_game, is_first_pitch_of_plate_appearance, has_previous_pitch,
  previous_pitch_velocity_missing, previous_pitch_location_missing. Absence of a previous pitch is stated explicitly
  instead of being implied by a missing value.
* ``prev_pitch_detail``: previous pitch velocity (a missing measurement is imputed with the pitch type's mean over
  prior games and flagged), velocity relative to that mean, location (plate_x, plate_z, zone, in-zone) of the previous
  pitch and location/result of the one before.
* ``catcher``: catcher identity (categorical) and smoothed usage with this catcher in prior games.
* ``recency_overall`` / ``recency_count`` / ``recency_transition`` / ``recency_batter``: usage shares over windows of
  prior games: previous start, last 3 starts, last 6 starts, season to date, previous 30 days, previous 60 days and an
  exponentially weighted window (half-life EWM_HALF_LIFE_DAYS days) - overall, in this count, after the previous pitch
  type and against this batter. Each is smoothed toward the coarser level (count/transition/batter toward the window's
  overall share, overall toward the career prior).
* ``batter_vs_pitcher``: smoothed swing rate and per-pitch-type whiff rate of this batter against this pitcher.
* ``batter_league`` / ``batter_platoon``: MLB-wide batter priors from prior seasons (see pitchiq.batter_priors).
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import date, timedelta

import numpy as np

from .. import batter_priors as bp

EWM_HALF_LIFE_DAYS = 30.0
K_REC_ALL = 10.0    # pseudo-pitches toward the career prior for a window's overall usage
K_REC_COND = 20.0   # toward the window's overall usage for count / transition windows
K_REC_BAT = 15.0    # toward the window's overall usage for batter windows
K_CATCHER = 30.0
K_BVP_SWING = 20.0
K_BVP_WHIFF = 10.0
WINDOWS = ["g1", "g3", "g6", "season", "d30", "d60", "ewm"]
WINDOW_DESC = {"g1": "the previous start", "g3": "the last 3 starts", "g6": "the last 6 starts",
               "season": "this season's earlier starts", "d30": "starts in the previous 30 days",
               "d60": "starts in the previous 60 days",
               "ewm": f"all prior starts, exponentially weighted (half-life {EWM_HALF_LIFE_DAYS:g} days)"}
SCOPES = {"all": ("recency_overall", "overall"), "count": ("recency_count", "in this ball-strike count"),
          "trans": ("recency_transition", "after the previous pitch type"),
          "bat": ("recency_batter", "against this batter")}
SWING_RESULTS = {"swinging_strike", "foul", "in_play"}
WHIFF_RESULTS = {"swinging_strike"}


@dataclass
class GameRecord:
    game_date: date
    season: int
    usage: Counter
    by_count: dict
    by_prev: dict
    by_batter: dict


def absorb(hist, game, game_date) -> None:
    """v2 history updates, called from History.absorb_game after the v1 counters."""
    hist.game_log.append(GameRecord(game_date, game_date.year, Counter(game.usage),
                                    {k: Counter(v) for k, v in _pairs(game.count_events).items()},
                                    {k: Counter(v) for k, v in _pairs(game.transitions).items()},
                                    {b: Counter(v) for b, v in game.by_batter.items()}))
    for t, spd in zip(game.types, game.speeds_measured):
        if spd is not None:
            hist.velo_sum[t] += spd
            hist.velo_n[t] += 1
    for cat, cnt in game.by_catcher.items():
        hist.by_catcher[cat].update(cnt)
    for b, cnt in game.bvp_swings.items():
        hist.bvp_swings[b].update(cnt)
    for b, cnt in game.bvp_whiffs.items():
        hist.bvp_whiffs[b].update(cnt)
    hist.swings.update(game.swings)
    hist.whiffs.update(game.whiffs)
    hist.window_cache.clear()


def _pairs(events) -> dict:
    out: dict = defaultdict(Counter)
    for k, c in events:
        out[k][c] += 1
    return out


def update(game, ctx: dict, label: str, speed, result: str, location) -> None:
    """v2 game-state updates for a thrown pitch (called only after its features were frozen)."""
    game.speeds_measured.append(speed)
    px, pz, zn = location if location is not None else (None, None, None)
    game.plate_x.append(_f(px))
    game.plate_z.append(_f(pz))
    game.zones.append(None if _f(zn) is None else int(zn))
    cat = _catcher(ctx.get("catcher_id"))
    game.by_catcher[cat][label] += 1
    b = ctx.get("batter")
    if result in SWING_RESULTS:
        game.swings[label] += 1
        game.bvp_swings[b][label] += 1
    if result in WHIFF_RESULTS:
        game.whiffs[label] += 1
        game.bvp_whiffs[b][label] += 1


def _f(v):
    if v is None:
        return None
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return None if np.isnan(v) else v


def _catcher(v) -> str:
    v = _f(v)
    return "NONE" if v is None else str(int(v))


def _windows(hist, game_date: date, classes: list[str], half_life: float) -> dict:
    key = (game_date, tuple(classes), half_life)
    if key in hist.window_cache:
        return hist.window_cache[key]
    log = hist.game_log
    sel: dict[str, list[tuple[GameRecord, float]]] = {w: [] for w in WINDOWS}
    n = len(log)
    for i, g in enumerate(log):
        back = n - i  # 1 = previous start
        age = (game_date - g.game_date).days
        if back <= 1:
            sel["g1"].append((g, 1.0))
        if back <= 3:
            sel["g3"].append((g, 1.0))
        if back <= 6:
            sel["g6"].append((g, 1.0))
        if g.season == game_date.year:
            sel["season"].append((g, 1.0))
        if age <= 30:
            sel["d30"].append((g, 1.0))
        if age <= 60:
            sel["d60"].append((g, 1.0))
        sel["ewm"].append((g, 0.5 ** (max(age, 0) / half_life)))
    out = {}
    for w, items in sel.items():
        agg = {"all": Counter(), "count": defaultdict(Counter), "trans": defaultdict(Counter),
               "bat": defaultdict(Counter)}
        for g, wt in items:
            _add(agg["all"], g.usage, wt)
            for k, c in g.by_count.items():
                _add(agg["count"][k], c, wt)
            for k, c in g.by_prev.items():
                _add(agg["trans"][k], c, wt)
            for k, c in g.by_batter.items():
                _add(agg["bat"][k], c, wt)
        out[w] = agg
    hist.window_cache[key] = out
    return out


def _add(dst: Counter, src: Counter, w: float) -> None:
    for k, v in src.items():
        dst[k] += v * w


def _share(cnt: Counter, prior: dict, k: float, classes: list[str]) -> dict:
    n = sum(cnt[c] for c in classes)
    return {c: (cnt[c] + k * prior[c]) / (n + k) for c in classes}


def features(ctx: dict, hist, game, pa, classes: list[str], f: dict) -> dict:
    """The v2 additions for one pitch. `f` is the v1 dictionary already built for it."""
    out: dict = {}
    has_prev = game.n > 0
    out["is_first_pitch_of_game"] = int(int(f["pitch_count_before"]) == 0)
    out["is_first_pitch_of_plate_appearance"] = int(int(ctx["pitch_number"]) == 1)
    out["has_previous_pitch"] = int(has_prev)
    # previous pitch detail; missing measurements are explicit flags, not implicit NaN
    if has_prev:
        t1 = game.types[-1]
        mean1 = hist.velo_sum[t1] / hist.velo_n[t1] if hist.velo_n[t1] else None
        measured = game.speeds_measured[-1]
        out["previous_pitch_velocity_missing"] = int(measured is None)
        out["prev1_velocity"] = measured if measured is not None else mean1
        out["prev1_velo_vs_type_mean"] = (measured - mean1) if (measured is not None and mean1 is not None) else None
        px, pz, zn = game.plate_x[-1], game.plate_z[-1], game.zones[-1]
        out["previous_pitch_location_missing"] = int(px is None or pz is None or zn is None)
        out["prev1_plate_x"], out["prev1_plate_z"] = px, pz
        out["prev1_zone"] = str(zn) if zn is not None else "UNK"
        out["prev1_in_zone"] = None if zn is None else int(zn <= 9)
    else:
        out.update(previous_pitch_velocity_missing=0, prev1_velocity=None, prev1_velo_vs_type_mean=None,
                   previous_pitch_location_missing=0, prev1_plate_x=None, prev1_plate_z=None,
                   prev1_zone="NONE", prev1_in_zone=None)
    if game.n >= 2:
        out["prev2_plate_x"], out["prev2_plate_z"] = game.plate_x[-2], game.plate_z[-2]
        out["prev2_result"] = game.results[-2]
    else:
        out.update(prev2_plate_x=None, prev2_plate_z=None, prev2_result="NONE")
    # catcher
    prior = hist.overall_prior(classes)
    cat = _catcher(ctx.get("catcher_id"))
    out["catcher_id"] = cat
    ccnt = hist.by_catcher[cat] if cat != "NONE" else Counter()
    out["cat_n"] = sum(ccnt[c] for c in classes)
    for c, v in _share(ccnt, prior, K_CATCHER, classes).items():
        out[f"cat_share_{c}"] = v
    # recency windows over prior games
    gd = ctx.get("game_date")
    half_life = float(getattr(hist, "ewm_half_life_days", EWM_HALF_LIFE_DAYS))
    wins = _windows(hist, gd, classes, half_life) if gd is not None else None
    ck, p1, b = f["count_str"], f["prev1_type"], ctx.get("batter")
    for w in WINDOWS:
        agg = wins[w] if wins is not None else None
        allc = agg["all"] if agg else Counter()
        out[f"rec_{w}_n"] = float(sum(allc[c] for c in classes))
        base = _share(allc, prior, K_REC_ALL, classes)
        for c in classes:
            out[f"rec_{w}_all_{c}"] = base[c]
        for scope, src, k in (("count", ck, K_REC_COND), ("trans", p1, K_REC_COND), ("bat", b, K_REC_BAT)):
            cnt = agg[scope].get(src, Counter()) if agg else Counter()
            for c, v in _share(cnt, base, k, classes).items():
                out[f"rec_{w}_{scope}_{c}"] = v
    # batter vs this pitcher outcomes (prior games + earlier in this game)
    sw = hist.bvp_swings[b] + game.bvp_swings[b]
    wh = hist.bvp_whiffs[b] + game.bvp_whiffs[b]
    seen = hist.by_batter[b] + game.by_batter[b]
    tot_sw, tot_seen = sum(hist.swings.values()), sum(hist.overall.values())
    p_swing = tot_sw / tot_seen if tot_seen else 0.5
    n_seen = sum(seen.values())
    out["bvp_swing_rate"] = (sum(sw.values()) + K_BVP_SWING * p_swing) / (n_seen + K_BVP_SWING)
    for c in classes:
        p_wh = (hist.whiffs[c] + 1.0) / (hist.swings[c] + 4.0)  # pitcher's prior whiff rate on c (lightly smoothed)
        out[f"bvp_whiff_{c}"] = (wh[c] + K_BVP_WHIFF * p_wh) / (sw[c] + K_BVP_WHIFF)
    # MLB-wide batter priors from seasons before this one
    pri = hist.batter_priors
    names = bp.feature_names(classes)
    if pri is None:
        out.update({n: (1 if n == "blb_missing" else None) for n in names})
    else:
        out.update(pri.lookup(b, ctx.get("season")))
    return out


# ------------------------------------------------------------------ schema
def schema(classes: list[str]):
    """[(name, kind, group, description)] of every v2 addition for these classes."""
    rows = [
        ("is_first_pitch_of_game", "numeric", "explicit_flags", "No pitch thrown by this pitcher earlier in the game"),
        ("is_first_pitch_of_plate_appearance", "numeric", "explicit_flags", "First pitch of the plate appearance"),
        ("has_previous_pitch", "numeric", "explicit_flags", "A previous thrown pitch exists this game"),
        ("previous_pitch_velocity_missing", "numeric", "explicit_flags",
         "Previous pitch exists but its velocity was not measured (imputed with the type's prior mean)"),
        ("previous_pitch_location_missing", "numeric", "explicit_flags",
         "Previous pitch exists but its plate location / zone was not measured"),
        ("prev1_velocity", "numeric", "prev_pitch_detail",
         "Velocity of the previous pitch (mph); unmeasured -> prior mean of that pitch type"),
        ("prev1_velo_vs_type_mean", "numeric", "prev_pitch_detail",
         "Previous pitch velocity minus the pitcher's prior mean for that type"),
        ("prev1_plate_x", "numeric", "prev_pitch_detail", "Horizontal plate location of the previous pitch (ft)"),
        ("prev1_plate_z", "numeric", "prev_pitch_detail", "Vertical plate location of the previous pitch (ft)"),
        ("prev1_zone", "categorical", "prev_pitch_detail", "Statcast zone of the previous pitch (NONE / UNK)"),
        ("prev1_in_zone", "numeric", "prev_pitch_detail", "Previous pitch was in the strike zone (zones 1-9)"),
        ("prev2_plate_x", "numeric", "prev_pitch_detail", "Horizontal plate location two pitches ago"),
        ("prev2_plate_z", "numeric", "prev_pitch_detail", "Vertical plate location two pitches ago"),
        ("prev2_result", "categorical", "prev_pitch_detail", "Outcome of the pitch two pitches ago"),
        ("catcher_id", "categorical", "catcher", "Catcher (MLBAM id) for this pitch"),
        ("cat_n", "numeric", "catcher", "Prior-games pitches thrown with this catcher"),
        ("bvp_swing_rate", "numeric", "batter_vs_pitcher", "Smoothed swing rate of this batter against this pitcher"),
    ]
    for c in classes:
        rows.append((f"cat_share_{c}", "numeric", "catcher", f"Smoothed prior-games share of {c} with this catcher"))
        rows.append((f"bvp_whiff_{c}", "numeric", "batter_vs_pitcher",
                     f"Smoothed whiff rate per swing of this batter against this pitcher's {c}"))
    for w in WINDOWS:
        rows.append((f"rec_{w}_n", "numeric", "recency_overall", f"Pitches in {WINDOW_DESC[w]} (weighted for ewm)"))
        for scope, (group, sdesc) in SCOPES.items():
            for c in classes:
                rows.append((f"rec_{w}_{scope}_{c}", "numeric", group,
                             f"Smoothed share of {c} {sdesc} over {WINDOW_DESC[w]}"))
    for tmpl, desc in bp.PER_CLASS_PRIORS:
        for c in bp.prior_codes(classes):
            rows.append((tmpl.format(c=c), "numeric", "batter_league", desc.format(c=c)))
    for n, desc in bp.DISCIPLINE_PRIORS + bp.FLAG_PRIORS:
        rows.append((n, "numeric", "batter_league", desc))
    for n, desc in bp.PLATOON_PRIORS:
        rows.append((n, "numeric", "batter_platoon", desc))
    return rows


GROUPS = ["explicit_flags", "prev_pitch_detail", "catcher", "recency_overall", "recency_count",
          "recency_transition", "recency_batter", "batter_vs_pitcher", "batter_league", "batter_platoon"]


def window_bounds(game_date: date) -> dict:
    """Human-readable window definitions for a game date (documentation / API)."""
    return {"d30": str(game_date - timedelta(days=30)), "d60": str(game_date - timedelta(days=60)),
            "season": game_date.year, "ewm_half_life_days": EWM_HALF_LIFE_DAYS}
