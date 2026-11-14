"""Survivor-bias and age-vs-calendar-year analyses of the population aging curves (reported, never used for
selection).

* Survivor bias: within-pitcher yearly changes exist only for pitchers who qualified again. The curve is estimated
  (a) on survivors as observed and (b) with inverse-probability-of-continuation weights (logistic model on season-t
  age, standardized value, workload, role). IPW corrects selection on what is observed at t; selection on the change
  itself (a pitcher who lost velocity and therefore lost his job) is not identified without an exclusion
  restriction, so a returners diagnostic is reported as well: the annualized change of pitchers who missed t+1 but
  qualified again within 3 seasons versus survivors of the same age band.
* Age vs calendar year: (c) adds target-season effects (sum-to-zero deviation coding) to the change model. Seasons
  are identified across birth cohorts (in a given season pitchers of many ages change together), but in first
  differences a linear calendar trend and the mean yearly age change are the same constant (age = season - birth
  year), so only the deviations of seasons from their mean trend are separated from age; the coding attributes the
  average yearly change to age. (d) adds season-t workload and starter share. (e) a level model (MixedLM, pitcher
  random intercept and age slope) with era fixed effect vs with season fixed effects, and (f) the naive
  cross-sectional level-by-age profile, which mixes age with cohort and survivor composition.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from .methods import KNOTS, consecutive_pairs, fit_age_curve, rcs

AGES = list(range(24, 41))


def _curve_points(curve) -> dict:
    return {str(a): float(curve.delta(np.array([a]))[0]) for a in AGES}


def _cum(curve, a0: int, a1: int) -> float:
    return float(sum(curve.delta(np.array([a]))[0] for a in range(a0, a1)))


def returners(hist: pd.DataFrame, cutoff: int, max_gap: int = 3) -> dict:
    """Annualized change of non-survivors who returned (qualified again within max_gap seasons, by the cutoff) vs
    survivors' one-season change, by origin age band; same-era, non-2020 pairs only."""
    from .methods import crosses_era
    h = hist[hist["season"] <= cutoff]
    s = h.set_index(["pitcher_id", "season"])["value"]
    seasons = set(h["season"])
    out = {}
    rows = []
    for (pid, t), v in s.items():
        if t + 1 not in seasons or t + 1 > cutoff:
            continue
        if (pid, t + 1) in s.index:
            continue
        for g in range(2, max_gap + 1):
            if (pid, t + g) in s.index and t + g <= cutoff:
                if not crosses_era(t, t + g) and not (t <= 2020 <= t + g):
                    age = float(h.loc[(h["pitcher_id"] == pid) & (h["season"] == t), "age"].iloc[0])
                    rows.append({"age": age, "annual": (s[(pid, t + g)] - v) / g, "gap": g})
                break
    r = pd.DataFrame(rows)
    sv = consecutive_pairs(h, 1)
    for lab, lo, hi in (("<32", 0, 31.999), (">=32", 32, 99)):
        a = r[(r["age"] >= lo) & (r["age"] <= hi)] if len(r) else r
        b = sv[(sv["age"] >= lo) & (sv["age"] <= hi)]
        out[lab] = {"returners_n": int(len(a)),
                    "returners_mean_annual_change": float(a["annual"].mean()) if len(a) else None,
                    "survivors_n": int(len(b)), "survivors_mean_change": float(b["delta"].mean()) if len(b) else None}
    return out


def survival_summary(hist: pd.DataFrame, cutoff: int) -> dict:
    h = hist[hist["season"] + 1 <= cutoff]
    seasons = set(hist["season"])
    h = h[h["season"].map(lambda s: s + 1 in seasons) & ~h["season"].isin([2019, 2020])]
    nxt = set(zip(hist["pitcher_id"], hist["season"]))
    surv = np.array([(p, s + 1) in nxt for p, s in zip(h["pitcher_id"], h["season"])])
    out = {}
    for lab, lo, hi in (("<=26", 0, 26.999), ("27-30", 27, 30.999), ("31-33", 31, 33.999), ("34-36", 34, 36.999),
                        ("37+", 37, 99)):
        m = ((h["age"] >= lo) & (h["age"] <= hi)).values
        out[lab] = {"at_risk": int(m.sum()), "continued": int(surv[m].sum()),
                    "rate": float(surv[m].mean()) if m.any() else None}
    # does season-t value predict continuation? (standardized within season)
    g = h.groupby("season")["value"]
    z = ((h["value"] - g.transform("mean")) / g.transform("std")).values
    ok = np.isfinite(z)
    out["continuation_rate_by_value_tercile"] = {}
    if ok.sum() > 30:
        qs = np.quantile(z[ok], [1 / 3, 2 / 3])
        for lab, m in (("low", z <= qs[0]), ("mid", (z > qs[0]) & (z <= qs[1])), ("high", z > qs[1])):
            m = m & ok
            out["continuation_rate_by_value_tercile"][lab] = float(surv[m].mean()) if m.any() else None
    return out


def mixed_level_compare(hist: pd.DataFrame) -> dict:
    """Level MixedLM: era FE vs season FE (pitcher random intercept + age slope). Age profile 30 -> 36 / 40."""
    import statsmodels.api as sm
    d = hist.dropna(subset=["value", "age", "workload"])
    if len(d) < 200 or d["pitcher_id"].nunique() < 30:
        return {"skipped": "too few rows"}
    base = [np.ones(len(d)), rcs(d["age"].values), np.log(d["workload"].clip(lower=50).values),
            d["gs_share"].fillna(0).values]
    Z = np.column_stack([np.ones(len(d)), (d["age"].values - 30.0) / 10.0])
    out = {}
    seasons = sorted(d["season"].unique())
    for name in ("era_fe", "season_fe"):
        cols = list(base)
        if name == "era_fe":
            cols += [(d["season"].values > 2019).astype(float), (d["season"].values == 2020).astype(float)]
        else:
            cols += [(d["season"].values == s).astype(float) for s in seasons[1:]]
        X = np.column_stack(cols)
        keep = np.zeros(X.shape[1], bool)
        keep[0] = True
        for j in range(1, X.shape[1]):
            trial = keep.copy()
            trial[j] = True
            if X[:, j].std() > 0 and np.linalg.matrix_rank(X[:, trial]) == trial.sum():
                keep = trial
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = sm.MixedLM(d["value"].values, X[:, keep], groups=d["pitcher_id"].values, exog_re=Z).fit(
                    reml=True, method=["lbfgs"])
        except Exception as e:  # noqa: BLE001
            out[name] = {"error": f"{type(e).__name__}: {e}"[:200]}
            continue
        fe = np.asarray(res.fe_params)
        idx = np.where(keep)[0]
        coef = np.zeros(X.shape[1])
        coef[idx] = fe

        def prof(a):
            return float(np.r_[1.0, rcs(np.array([a]))[0]] @ coef[:1 + len(KNOTS) - 1])
        r = {"age_effect_30_to_36": prof(36) - prof(30), "age_effect_30_to_40": prof(40) - prof(30),
             "age_effect_34_to_37": prof(37) - prof(34)}
        if name == "season_fe":
            r["season_effects_vs_first"] = {str(s): float(coef[1 + len(KNOTS) - 1 + 2 + i])
                                            for i, s in enumerate(seasons[1:]) if keep[1 + len(KNOTS) - 1 + 2 + i]}
        else:
            r["hawkeye_effect"] = float(coef[-2]) if keep[-2] else None
        out[name] = r
    return out


def cross_section(hist: pd.DataFrame) -> dict:
    """Mean level by completed age (all qualified pitcher-seasons pooled), mixing age, cohort and survival."""
    a = hist["age"].astype(int)
    g = hist.groupby(a)["value"].agg(["mean", "count"])
    prof = {str(int(k)): {"mean": float(v["mean"]), "n": int(v["count"])} for k, v in g.iterrows() if 24 <= k <= 40}
    d30, d36 = prof.get("30"), prof.get("36")
    return {"by_age": prof, "diff_36_minus_30": None if not (d30 and d36) else d36["mean"] - d30["mean"]}


def analyze_target(hist: pd.DataFrame, cutoff: int, include_mixed: bool = True) -> dict:
    h = hist[hist["season"] <= cutoff]
    variants = {"survivors": fit_age_curve(h, cutoff), "ipw": fit_age_curve(h, cutoff, ipw=True),
                "survivors_season_fe": fit_age_curve(h, cutoff, season_fe=True),
                "ipw_season_fe_workload": fit_age_curve(h, cutoff, ipw=True, season_fe=True, workload=True)}
    curves = {}
    for k, c in variants.items():
        curves[k] = {"delta_by_age": _curve_points(c), "cum_30_to_36": _cum(c, 30, 36), "cum_34_to_37": _cum(c, 34, 37),
                     "cum_37_to_40": _cum(c, 37, 40), "n_pairs": c.n_pairs, "mean_delta": c.mean_delta,
                     "season_effects": c.season_effects}
    out = {"cutoff": int(cutoff), "curves": curves,
           "ipw_minus_survivors_cum_30_to_36": curves["ipw"]["cum_30_to_36"] - curves["survivors"]["cum_30_to_36"],
           "ipw_minus_survivors_cum_37_to_40": curves["ipw"]["cum_37_to_40"] - curves["survivors"]["cum_37_to_40"],
           "season_fe_minus_survivors_cum_30_to_36": (curves["survivors_season_fe"]["cum_30_to_36"]
                                                      - curves["survivors"]["cum_30_to_36"]),
           "survival": survival_summary(h, cutoff), "returners": returners(h, cutoff),
           "cross_section": cross_section(h)}
    if include_mixed:
        out["mixed_level"] = mixed_level_compare(h)
    return out
