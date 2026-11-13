"""Feature schema (version-stamped) and leakage policy.

Every feature must be computable from information available immediately before the pitch.
`FORBIDDEN_SOURCE_COLUMNS` lists Statcast columns describing the CURRENT pitch or later events;
the leakage tests assert none of them can influence the current pitch's feature vector.
"""
from __future__ import annotations

from dataclasses import dataclass

FEATURE_VERSION = "v1"
NONE_TOKEN = "NONE"  # no previous pitch (first pitch of game / PA)


@dataclass(frozen=True)
class Feature:
    name: str
    kind: str  # numeric | categorical
    group: str
    description: str


# Per-class features are expanded at runtime for each model class (e.g. game_share_SL).
PER_CLASS_TEMPLATES = [
    ("game_share_{c}", "usage", "Smoothed share of {c} among pitches already thrown this game"),
    ("recent_share_{c}", "usage", "Share of {c} over the previous 3 games (strictly before today)"),
    ("count_prior_{c}", "count", "Prior-games share of {c} in this exact ball-strike count"),
    ("trans_prior_{c}", "sequence", "Prior-games share of {c} immediately after the previous pitch type"),
    ("bvp_share_{c}", "batter_history", "Smoothed share of {c} thrown to this batter before this pitch"),
    ("pa_n_{c}", "sequence", "Number of {c} already thrown in this plate appearance"),
]

BASE_FEATURES: list[Feature] = [
    Feature("stand", "categorical", "handedness", "Batter handedness (L/R)"),
    Feature("bat_order_slot", "numeric", "lineup", "Batter's batting-order position (1-9)"),
    Feature("balls", "numeric", "count", "Balls before the pitch"),
    Feature("strikes", "numeric", "count", "Strikes before the pitch"),
    Feature("count_str", "categorical", "count", "Ball-strike count, e.g. '1-2'"),
    Feature("inning", "numeric", "inning", "Inning"),
    Feature("pitch_count_before", "numeric", "pitch_count", "Pitches thrown by the pitcher earlier in this game"),
    Feature("pa_pitch_number", "numeric", "count", "Pitch number within the plate appearance"),
    Feature("times_through_order", "numeric", "tto", "Times through the batting order"),
    Feature("outs", "numeric", "situation", "Outs"),
    Feature("on_1b", "numeric", "situation", "Runner on first"),
    Feature("on_2b", "numeric", "situation", "Runner on second"),
    Feature("on_3b", "numeric", "situation", "Runner on third"),
    Feature("score_diff", "numeric", "situation", "Pitcher's team score minus batting team score"),
    Feature("is_home", "numeric", "context", "Pitcher's team is home"),
    Feature("prev1_type", "categorical", "sequence", "Previous pitch type this game"),
    Feature("prev2_type", "categorical", "sequence", "Pitch type two pitches ago"),
    Feature("prev3_type", "categorical", "sequence", "Pitch type three pitches ago"),
    Feature("prev1_same_pa", "numeric", "sequence", "Previous pitch was in the same plate appearance"),
    Feature("prev1_result", "categorical", "sequence", "Outcome of the previous pitch"),
    Feature("prev1_speed", "numeric", "sequence", "Velocity of the previous pitch (mph)"),
    Feature("game_ff_velo_mean", "numeric", "pitch_count", "Mean four-seam velocity so far this game"),
    Feature("bvp_n", "numeric", "batter_history", "Pitches previously thrown to this batter"),
    Feature("season", "numeric", "context", "Season"),
    Feature("days_rest", "numeric", "context", "Days since previous appearance"),
]

# Never allowed to be read for the target pitch (they describe the pitch itself or the future).
FORBIDDEN_SOURCE_COLUMNS = {
    "pitch_type", "pitch_name", "release_speed", "release_spin_rate", "spin_axis", "pfx_x", "pfx_z",
    "plate_x", "plate_z", "zone", "description", "events", "type", "des", "launch_speed", "launch_angle",
    "hit_distance_sc", "bb_type", "hc_x", "hc_y", "effective_speed", "vx0", "vy0", "vz0", "ax", "ay", "az",
    "release_extension", "release_pos_x", "release_pos_y", "release_pos_z", "post_away_score",
    "post_home_score", "post_bat_score", "post_fld_score", "delta_home_win_exp", "delta_run_exp",
    "estimated_ba_using_speedangle", "estimated_woba_using_speedangle", "woba_value", "bat_speed",
    "swing_length", "arm_angle",
}

# Stored for analytics but excluded from model inputs by default: a numeric season cannot generalize to
# seasons outside the training range (linear models extrapolate it). Decided from the a-priori argument and
# the validation ablation (removing the context group improved logreg validation log loss by 0.036).
MODEL_EXCLUDED_DEFAULT = {"season"}

FEATURE_GROUPS = ["handedness", "lineup", "count", "inning", "pitch_count", "sequence", "situation", "tto",
                  "batter_history", "usage", "context"]


def feature_list(classes: list[str]) -> list[Feature]:
    feats = list(BASE_FEATURES)
    for tmpl, group, desc in PER_CLASS_TEMPLATES:
        for c in classes:
            feats.append(Feature(tmpl.format(c=c), "numeric", group, desc.format(c=c)))
    return feats


def schema_dict(classes: list[str]) -> dict:
    return {"feature_version": FEATURE_VERSION, "classes": classes,
            "features": [f.__dict__ for f in feature_list(classes)]}
