"""Feature schema (version-stamped) and leakage policy.

Every feature must be computable from information available immediately before the pitch.
`FORBIDDEN_SOURCE_COLUMNS` lists Statcast columns describing the CURRENT pitch or later events;
the leakage tests assert none of them can influence the current pitch's feature vector.
"""
from __future__ import annotations

from dataclasses import dataclass

FEATURE_VERSION = "v2"  # current version built, trained and served by default
FEATURE_VERSIONS = ("v1", "v2")  # every version is rebuilt on ingestion so v1 models stay usable
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
# v2 models read the previous pitch's velocity from `prev1_velocity` + explicit flags; v1's `prev1_speed` (missing
# value = "no previous pitch") stays in the stored dictionary for v1 models but is not a v2 model input.
MODEL_EXCLUDED_BY_VERSION = {"v1": MODEL_EXCLUDED_DEFAULT, "v2": MODEL_EXCLUDED_DEFAULT | {"prev1_speed"}}

V1_GROUPS = ["handedness", "lineup", "count", "inning", "pitch_count", "sequence", "situation", "tto",
             "batter_history", "usage", "context"]


def _v2_groups() -> list[str]:
    from .v2 import GROUPS
    return list(GROUPS)


FEATURE_GROUPS = V1_GROUPS + _v2_groups()


def feature_list(classes: list[str], version: str = FEATURE_VERSION) -> list[Feature]:
    if version not in FEATURE_VERSIONS:
        raise ValueError(f"unknown feature version {version}; known {FEATURE_VERSIONS}")
    feats = list(BASE_FEATURES)
    for tmpl, group, desc in PER_CLASS_TEMPLATES:
        for c in classes:
            feats.append(Feature(tmpl.format(c=c), "numeric", group, desc.format(c=c)))
    if version == "v2":
        from .v2 import schema
        feats += [Feature(n, k, g, d) for n, k, g, d in schema(classes)]
    return feats


# v2 feature groups that are stored but not used as default model inputs. Chosen on the rolling development
# ablation (2024-2025 folds) only; see docs/np-features.md. Empty = every v2 group is a default input.
V2_MODEL_EXCLUDED_GROUPS: set[str] = set()


def default_model_features(classes: list[str], version: str = FEATURE_VERSION) -> list[str]:
    excl = MODEL_EXCLUDED_BY_VERSION.get(version, MODEL_EXCLUDED_DEFAULT)
    groups = V2_MODEL_EXCLUDED_GROUPS if version == "v2" else set()
    return [f.name for f in feature_list(classes, version) if f.name not in excl and f.group not in groups]


def schema_dict(classes: list[str], version: str = FEATURE_VERSION) -> dict:
    d = {"feature_version": version, "classes": classes,
         "features": [f.__dict__ for f in feature_list(classes, version)]}
    if version == "v2":
        from . import v2
        d["params"] = {"ewm_half_life_days": v2.EWM_HALF_LIFE_DAYS, "windows": v2.WINDOWS,
                       "smoothing": {"rec_all": v2.K_REC_ALL, "rec_cond": v2.K_REC_COND, "rec_bat": v2.K_REC_BAT,
                                     "catcher": v2.K_CATCHER, "bvp_swing": v2.K_BVP_SWING,
                                     "bvp_whiff": v2.K_BVP_WHIFF},
                       "model_excluded_default": sorted(MODEL_EXCLUDED_BY_VERSION["v2"])}
    return d
