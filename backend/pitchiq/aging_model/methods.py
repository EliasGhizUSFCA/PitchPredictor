"""Forecast methods for pitcher-season characteristics (season t+h from data <= t only).

Every method takes an `OriginData` (one target, one historical cutoff; `hist` holds only qualified rows with
season <= cutoff) and returns one row per (pitcher qualified in the cutoff season, horizon) with a point forecast and,
where the method has its own predictive distribution, quantiles / a Gaussian sd. Methods without their own
distribution get empirical (rolling-origin conformal) intervals in pitchiq.aging_model.backtest.

Methods
* no_change       y(t+h) = y(t).
* marcel          weighted recent seasons (5/4/3 x sample size) regressed to the season-t population mean with
                  k "pseudo-pitches"; k chosen on pseudo-forecasts that end on or before the cutoff.
* age_curve       population delta method: y(t) + sum of the population mean yearly change at each age, estimated
                  from consecutive-season pairs of survivors (restricted cubic spline in age).
* age_curve_ipw   same, with survivors weighted by the inverse of their estimated continuation probability
                  (logistic model on age, standardized value, workload and role in season t): survivor-bias correction.
* mixedlm         statsmodels MixedLM on levels: restricted cubic spline in age + Hawk-Eye era + 2020 short-season
                  + log workload + starter share fixed effects, random intercept and random age slope per pitcher.
* lgbm_quantile   LightGBM quantile regression (10/50/90) of y(t+h) - y(t) on the pitcher's history at t.
* eb_kalman       empirical-Bayes local-level model: the pitcher's true level follows the (IPW) population age curve
                  plus a random walk; observations add season noise and sampling noise that shrinks with pitches.
                  Variance components from a variogram of within-pitcher changes; Kalman filter over the pitcher's
                  seasons; Gaussian predictive distribution (shrinks to the population when the pitcher has little
                  history).

No method forces a decline and none infers injuries. Hawk-Eye (2020+) vs TrackMan (2015-2019) is handled as described
per method; the 60-game 2020 season is excluded from pair-based estimates.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..aging.families import TRACKMAN_LAST_SEASON

from .panel import Target

KNOTS = (23.5, 27.0, 30.5, 36.0)  # ~5/35/65/95 % of qualifying pitcher-season ages (Harrell's placement)
QUANTILES = (0.1, 0.5, 0.9)
Z90 = 1.2815515655446004
MARCEL_WEIGHTS = (5.0, 4.0, 3.0)
MARCEL_K_GRID = (0.0, 50.0, 150.0, 400.0, 1000.0, 3000.0)
LGBM_PARAMS = dict(n_estimators=300, learning_rate=0.05, num_leaves=15, min_child_samples=40, subsample=0.8,
                   subsample_freq=1, colsample_bytree=0.9, reg_lambda=1.0, verbose=-1, n_jobs=1)
METHODS = ["no_change", "marcel", "age_curve", "age_curve_ipw", "mixedlm", "lgbm_quantile", "eb_kalman"]
NATIVE_DISTRIBUTION = {"lgbm_quantile", "eb_kalman"}


def rcs(x, knots=KNOTS) -> np.ndarray:
    """Restricted (natural) cubic spline basis: linear beyond the outer knots. Columns: x, then len(knots)-2."""
    x = np.asarray(x, float)
    t = knots
    k = len(t)
    out = [x]

    def cube(u):
        return np.maximum(u, 0.0) ** 3

    for j in range(k - 2):
        out.append((cube(x - t[j]) - cube(x - t[k - 2]) * (t[k - 1] - t[j]) / (t[k - 1] - t[k - 2])
                    + cube(x - t[k - 1]) * (t[k - 2] - t[j]) / (t[k - 1] - t[k - 2])) / (t[k - 1] - t[0]) ** 2)
    return np.column_stack(out)


def is_hawkeye(season) -> np.ndarray:
    return np.asarray(season) > TRACKMAN_LAST_SEASON


def crosses_era(s_from, s_to) -> np.ndarray:
    return (np.asarray(s_from) <= TRACKMAN_LAST_SEASON) & (np.asarray(s_to) > TRACKMAN_LAST_SEASON)


@dataclass
class OriginData:
    target: Target
    cutoff: int
    hist: pd.DataFrame  # qualified rows, season <= cutoff
    horizons: tuple[int, ...] = (1, 2, 3)
    pitchers: set[int] | None = None  # restrict forecast rows to these pitchers (fits still use everyone)
    cache: dict = field(default_factory=dict)

    @property
    def origin(self) -> pd.DataFrame:
        o = self.hist[self.hist["season"] == self.cutoff]
        if self.pitchers is not None:
            o = o[o["pitcher_id"].isin(self.pitchers)]
        return o

    def wide(self, col: str) -> pd.DataFrame:
        key = ("wide", col)
        if key not in self.cache:
            self.cache[key] = self.hist.pivot_table(index="pitcher_id", columns="season", values=col, aggfunc="first")
        return self.cache[key]

    def league_mean(self) -> pd.Series:
        """n-weighted mean of qualified values per season (seasons <= cutoff)."""
        if "league" not in self.cache:
            h = self.hist
            w = h["n"].clip(lower=1)
            self.cache["league"] = (h["value"] * w).groupby(h["season"]).sum() / w.groupby(h["season"]).sum()
        return self.cache["league"]


def _frame(o: pd.DataFrame, h: int, point, **extra) -> pd.DataFrame:
    d = pd.DataFrame({"pitcher_id": o["pitcher_id"].values, "h": h, "origin_season": o["season"].values,
                      "target_season": o["season"].values + h, "origin_value": o["value"].values,
                      "origin_age": o["age"].values, "origin_n": o["n"].values, "point": np.asarray(point, float)})
    for k, v in extra.items():
        d[k] = np.asarray(v, float) if not np.isscalar(v) else v
    return d


def _empty() -> pd.DataFrame:
    return pd.DataFrame(columns=["pitcher_id", "h", "origin_season", "target_season", "origin_value", "origin_age",
                                 "origin_n", "point"])


# --------------------------------------------------------------------------- baselines
def no_change(od: OriginData) -> pd.DataFrame:
    o = od.origin
    if o.empty:
        return _empty()
    return pd.concat([_frame(o, h, o["value"].values) for h in od.horizons], ignore_index=True)


def _marcel_point(V: pd.DataFrame, N: pd.DataFrame, league: pd.Series, t: int, k: float) -> pd.Series:
    s1 = pd.Series(0.0, index=V.index)
    s0 = pd.Series(0.0, index=V.index)
    for lag, w in enumerate(MARCEL_WEIGHTS):
        s = t - lag
        if s not in V.columns:
            continue
        v, n = V[s], N[s]
        m = v.notna() & n.notna()
        s1 += (w * n * v).where(m, 0.0)
        s0 += (w * n).where(m, 0.0)
    return (s1 + k * league.get(t, np.nan)) / (s0 + k)


def marcel_k(od: OriginData, max_origin: int | None = None) -> float:
    """Shrinkage k chosen by h=1 pseudo-forecast MAE over origins t+1 <= min(cutoff, max_origin+1)."""
    key = ("marcel_k", max_origin)
    if key in od.cache:
        return od.cache[key]
    V, N, L = od.wide("value"), od.wide("n"), od.league_mean()
    last = od.cutoff if max_origin is None else min(od.cutoff, int(max_origin) + 1)
    errs = {k: [] for k in MARCEL_K_GRID}
    for t in [s for s in V.columns if s + 1 <= last and s + 1 in V.columns]:
        if crosses_era(t, t + 1) or t + 1 == 2020 or t == 2020:
            continue
        have = V[t].notna() & V[t + 1].notna()
        if have.sum() < 20:
            continue
        for k in MARCEL_K_GRID:
            p = _marcel_point(V, N, L, t, k)[have]
            errs[k].append(np.abs(p - V.loc[have, t + 1]).values)
    scores = {k: float(np.mean(np.concatenate(v))) for k, v in errs.items() if v}
    best = min(scores, key=scores.get) if scores else 400.0
    od.cache[key] = best
    od.cache[("marcel_k_scores", max_origin)] = scores
    return best


def marcel(od: OriginData, k: float | None = None) -> pd.DataFrame:
    o = od.origin
    if o.empty:
        return _empty()
    k = marcel_k(od, od.cache.get("tuning_max_origin")) if k is None else k
    p = _marcel_point(od.wide("value"), od.wide("n"), od.league_mean(), od.cutoff, k)
    pts = p.reindex(o["pitcher_id"]).values
    return pd.concat([_frame(o, h, pts, marcel_k=k) for h in od.horizons], ignore_index=True)


# --------------------------------------------------------------------------- population age curve (delta method)
def consecutive_pairs(hist: pd.DataFrame, k: int = 1, clean: bool = True) -> pd.DataFrame:
    """(pitcher, t, t+k) where both seasons qualified. clean: same measurement era and no 2020 season."""
    a = hist[["pitcher_id", "season", "value", "n", "age", "workload", "gs_share"]]
    b = hist[["pitcher_id", "season", "value", "n"]].copy()
    b["season"] = b["season"] - k
    d = a.merge(b, on=["pitcher_id", "season"], suffixes=("", "_to"))
    d["season_to"] = d["season"] + k
    d["delta"] = d["value_to"] - d["value"]
    if clean:
        d = d[~crosses_era(d["season"], d["season_to"]) & ~((d["season"] <= 2020) & (d["season_to"] >= 2020))]
    return d.reset_index(drop=True)


def _winsor(x: pd.Series, q=0.01) -> pd.Series:
    if len(x) < 20:
        return x
    lo, hi = x.quantile(q), x.quantile(1 - q)
    return x.clip(lo, hi)


@dataclass
class AgeCurve:
    coef: np.ndarray | None
    age_min: float
    age_max: float
    n_pairs: int
    mean_delta: float
    resid_sd: float
    weighted: bool
    season_effects: dict | None = None

    def delta(self, age) -> np.ndarray:
        a = np.clip(np.asarray(age, float), self.age_min, self.age_max)
        if self.coef is None:
            return np.full(a.shape, self.mean_delta)
        return np.column_stack([np.ones(a.size), rcs(a.ravel())]) @ self.coef

    def cumulative(self, age, h: int) -> np.ndarray:
        age = np.asarray(age, float)
        return sum(self.delta(age + j) for j in range(h))

    def to_dict(self) -> dict:
        return {"coef": None if self.coef is None else [float(c) for c in self.coef], "knots": list(KNOTS),
                "age_range": [self.age_min, self.age_max], "n_pairs": self.n_pairs, "mean_delta": self.mean_delta,
                "resid_sd": self.resid_sd, "weighted": self.weighted, "season_effects": self.season_effects}


def survival_weights(hist: pd.DataFrame, cutoff: int) -> pd.Series:
    """Inverse probability of continuation for each qualified (pitcher, t) with t+1 <= cutoff.

    Continuation = the unit qualifies again in t+1. Logistic model on season-t information only: spline in age,
    value standardized within season, log workload, starter share. t = 2019 / 2020 (opt-outs, 60 games) are not
    used to fit. Returns weights indexed by (pitcher_id, season), normalised to mean 1 over survivors, trimmed at
    the 99th percentile."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    h = hist[hist["season"] + 1 <= cutoff].copy()
    seasons = set(hist["season"].unique())
    h = h[h["season"].map(lambda s: s + 1 in seasons)]
    if h.empty:
        return pd.Series(dtype=float)
    nxt = set(zip(hist["pitcher_id"], hist["season"]))
    h["survived"] = [(p, s + 1) in nxt for p, s in zip(h["pitcher_id"], h["season"])]
    g = h.groupby("season")["value"]
    h["z"] = ((h["value"] - g.transform("mean")) / g.transform("std").replace(0, np.nan)).fillna(0.0)
    X = np.column_stack([rcs(h["age"].values), h["z"].values, np.log(h["workload"].clip(lower=50).values),
                         h["gs_share"].fillna(0.0).values])
    fit = h[~h["season"].isin([2019, 2020])]
    if fit["survived"].nunique() < 2 or len(fit) < 50:
        return pd.Series(1.0, index=pd.MultiIndex.from_frame(h.loc[h["survived"], ["pitcher_id", "season"]]))
    Xf = X[~h["season"].isin([2019, 2020]).values]
    sc = StandardScaler().fit(Xf)
    lr = LogisticRegression(C=1.0, max_iter=1000).fit(sc.transform(Xf), fit["survived"].values)
    p = np.clip(lr.predict_proba(sc.transform(X))[:, 1], 0.05, 1.0)
    s = h["survived"].values
    w = pd.Series(1.0 / p[s], index=pd.MultiIndex.from_frame(h.loc[s, ["pitcher_id", "season"]]))
    w = w.clip(upper=w.quantile(0.99))
    return w / w.mean()


def fit_age_curve(hist: pd.DataFrame, cutoff: int, ipw: bool = False, season_fe: bool = False,
                  workload: bool = False) -> AgeCurve:
    pr = consecutive_pairs(hist[hist["season"] <= cutoff])
    if len(pr) < 30:
        md = float(pr["delta"].mean()) if len(pr) else 0.0
        return AgeCurve(None, 20.0, 45.0, int(len(pr)), md, float(pr["delta"].std()) if len(pr) > 1 else np.nan, ipw)
    y = _winsor(pr["delta"]).values
    cols = [np.ones(len(pr)), rcs(pr["age"].values)]
    names = ["const"] + [f"rcs{i}" for i in range(len(KNOTS) - 1)]
    seasons = []
    if workload:
        cols.append(np.log(pr["workload"].clip(lower=50).values))
        names.append("log_workload_t")
        cols.append(pr["gs_share"].fillna(0).values)
        names.append("gs_share_t")
    if season_fe:
        # Sum-to-zero (deviation) coding of the target season: the mean yearly change is attributed to age, the
        # seasonal deviations to calendar time. The linear part of a calendar trend is not separable from the mean
        # age change in first differences (age = season - birth year); this coding is the identifying assumption.
        seasons = sorted(pr["season_to"].unique())
        if len(seasons) > 1:
            ref = seasons[-1]
            for s in seasons[:-1]:
                cols.append(np.where(pr["season_to"] == s, 1.0, np.where(pr["season_to"] == ref, -1.0, 0.0)))
                names.append(f"season_{s}")
    X = np.column_stack(cols)
    w = np.ones(len(pr))
    if ipw:
        sw = survival_weights(hist, cutoff)
        idx = pd.MultiIndex.from_frame(pr[["pitcher_id", "season"]])
        w = sw.reindex(idx).fillna(1.0).values
    sw_ = np.sqrt(w)
    coef, *_ = np.linalg.lstsq(X * sw_[:, None], y * sw_, rcond=None)
    resid = y - X @ coef
    se = None
    if season_fe and len(seasons) > 1:
        vals = {f"season_{s}": float(coef[names.index(f"season_{s}")]) for s in seasons[:-1]}
        vals[f"season_{seasons[-1]}"] = -float(sum(vals.values()))
        se = vals
    extra = {n: float(c) for n, c in zip(names, coef) if n in ("log_workload_t", "gs_share_t")}
    curve = AgeCurve(coef[:len(KNOTS)], float(pr["age"].quantile(0.002)), float(pr["age"].max()), int(len(pr)),
                     float(np.average(y, weights=w)), float(np.sqrt(np.average(resid ** 2, weights=w))), ipw, se)
    if extra:
        curve.season_effects = {**(curve.season_effects or {}), **{f"coef_{k}": v for k, v in extra.items()}}
        # workload terms shift the level of the yearly change; fold their mean contribution into the constant so
        # delta(age) is the change at the average workload / role of the pairs.
        X_w = X[:, names.index("log_workload_t")] * extra["log_workload_t"] + \
            X[:, names.index("gs_share_t")] * extra["gs_share_t"]
        curve.coef = curve.coef.copy()
        curve.coef[0] += float(np.average(X_w, weights=w))
    return curve


def _curve(od: OriginData, ipw: bool) -> AgeCurve:
    key = ("curve", ipw)
    if key not in od.cache:
        od.cache[key] = fit_age_curve(od.hist, od.cutoff, ipw=ipw)
    return od.cache[key]


def age_curve(od: OriginData, ipw: bool = False) -> pd.DataFrame:
    o = od.origin
    if o.empty:
        return _empty()
    c = _curve(od, ipw)
    return pd.concat([_frame(o, h, o["value"].values + c.cumulative(o["age"].values, h)) for h in od.horizons],
                     ignore_index=True)


# --------------------------------------------------------------------------- mixed-effects model
def _mixed_design(d: pd.DataFrame, season=None, age=None) -> np.ndarray:
    season = d["season"].values if season is None else np.asarray(season)
    age = d["age"].values if age is None else np.asarray(age, float)
    return np.column_stack([np.ones(len(d)), rcs(age), is_hawkeye(season).astype(float),
                            (season == 2020).astype(float), np.log(d["workload"].clip(lower=50).values),
                            d["gs_share"].fillna(0.0).values])


MIXED_FE_NAMES = ["const", "age", "age_rcs1", "age_rcs2", "hawkeye", "short_2020", "log_workload", "gs_share"]


def fit_mixed(hist: pd.DataFrame):
    import statsmodels.api as sm
    d = hist.dropna(subset=["value", "age", "workload"])
    X = _mixed_design(d)
    # keep columns that add rank (e.g. at cutoff 2020 the Hawk-Eye and 2020 indicators are the same column)
    keep = np.zeros(X.shape[1], bool)
    keep[0] = True
    for j in range(1, X.shape[1]):
        trial = keep.copy()
        trial[j] = True
        if X[:, j].std() > 0 and np.linalg.matrix_rank(X[:, trial]) == trial.sum():
            keep = trial
    Z = np.column_stack([np.ones(len(d)), (d["age"].values - 30.0) / 10.0])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = sm.MixedLM(d["value"].values, X[:, keep], groups=d["pitcher_id"].values, exog_re=Z)
        res = m.fit(reml=True, method=["lbfgs", "powell"])
    return res, keep


def mixedlm(od: OriginData) -> pd.DataFrame:
    o = od.origin
    if o.empty:
        return _empty()
    if "mixed" not in od.cache:
        od.cache["mixed"] = fit_mixed(od.hist)
    res, keep = od.cache["mixed"]
    re = res.random_effects
    b = np.array([np.asarray(re.get(p, np.zeros(2)), float)[:2] for p in o["pitcher_id"]])
    fe = np.asarray(res.fe_params, float)
    out = []
    for h in od.horizons:
        age = o["age"].values + h
        X = _mixed_design(o, season=o["season"].values + h, age=age)[:, keep]
        pt = X @ fe + b[:, 0] + b[:, 1] * (age - 30.0) / 10.0
        out.append(_frame(o, h, pt))
    return pd.concat(out, ignore_index=True)


# --------------------------------------------------------------------------- LightGBM quantile regression
LGBM_FEATURES = ["y_t", "d_prev", "d_mean3", "gap_prev", "log_n_t", "log_workload_t", "gs_share_t", "age_t", "h",
                 "league_t", "rel_league_t", "hawkeye_t", "crosses_era", "target_2020", "origin_2020", "n_seasons"]


def _lgbm_rows(hist: pd.DataFrame, league: pd.Series) -> pd.DataFrame:
    """History features for every qualified (pitcher, t), using seasons <= t only."""
    h = hist.sort_values(["pitcher_id", "season"]).copy()
    g = h.groupby("pitcher_id")
    prev_v = g["value"].shift(1)
    prev_s = g["season"].shift(1)
    h["d_prev"] = (h["value"] - prev_v).where(h["season"] - prev_s <= 3)
    h["gap_prev"] = (h["season"] - prev_s).fillna(0)
    roll = g["value"].transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
    h["d_mean3"] = h["value"] - roll
    h["n_seasons"] = g.cumcount() + 1
    h["y_t"] = h["value"]
    h["log_n_t"] = np.log1p(h["n"])
    h["log_workload_t"] = np.log(h["workload"].clip(lower=50))
    h["gs_share_t"] = h["gs_share"].fillna(0.0)
    h["age_t"] = h["age"]
    h["league_t"] = h["season"].map(league)
    h["rel_league_t"] = h["value"] - h["league_t"]
    h["hawkeye_t"] = is_hawkeye(h["season"]).astype(float)
    h["origin_2020"] = (h["season"] == 2020).astype(float)
    return h


def lgbm_quantile(od: OriginData) -> pd.DataFrame:
    import lightgbm as lgb
    o = od.origin
    if o.empty:
        return _empty()
    rows = _lgbm_rows(od.hist, od.league_mean())
    key = rows.set_index(["pitcher_id", "season"])["value"]
    train = []
    for h in od.horizons:
        r = rows[rows["season"] + h <= od.cutoff].copy()
        r["h"] = float(h)
        tgt = key.reindex(pd.MultiIndex.from_arrays([r["pitcher_id"], r["season"] + h])).values
        r["y"] = tgt - r["value"].values
        r["crosses_era"] = crosses_era(r["season"], r["season"] + h).astype(float)
        r["target_2020"] = ((r["season"] + h) == 2020).astype(float)
        train.append(r[np.isfinite(r["y"])])
    train = pd.concat(train, ignore_index=True)
    if len(train) < 200:
        return _empty()
    models = od.cache.get("lgbm")
    if models is None:
        models = {}
        for q in QUANTILES:
            m = lgb.LGBMRegressor(objective="quantile", alpha=q, random_state=17, **LGBM_PARAMS)
            m.fit(train[LGBM_FEATURES], train["y"])
            models[q] = m
        od.cache["lgbm"] = models
        od.cache["lgbm_n_train"] = int(len(train))
    orow = rows[rows["season"] == od.cutoff]
    orow = orow.set_index("pitcher_id").reindex(o["pitcher_id"]).reset_index()
    out = []
    for h in od.horizons:
        X = orow.copy()
        X["h"] = float(h)
        X["crosses_era"] = crosses_era(X["season"], X["season"] + h).astype(float)
        X["target_2020"] = ((X["season"] + h) == 2020).astype(float)
        qs = np.sort(np.column_stack([models[q].predict(X[LGBM_FEATURES]) for q in QUANTILES]), axis=1)
        base = o["value"].values
        out.append(_frame(o, h, base + qs[:, 1], q10=base + qs[:, 0], q50=base + qs[:, 1], q90=base + qs[:, 2]))
    return pd.concat(out, ignore_index=True)


# --------------------------------------------------------------------------- empirical-Bayes local-level model
@dataclass
class EBParams:
    season_var: float  # a: season-to-season noise of an observed season around the true level
    sampling_coef: float  # b: sampling variance = b / n
    drift_var: float  # q: yearly random-walk variance of the true level
    prior_var: float  # tau^2: between-pitcher variance of true levels at entry
    era_shift: float  # mean Hawk-Eye vs TrackMan level shift (0 if not estimable at the cutoff)
    era_var: float
    n_pairs: int

    def to_dict(self) -> dict:
        return {k: float(v) for k, v in self.__dict__.items()}


def fit_eb(hist: pd.DataFrame, cutoff: int, curve: AgeCurve) -> EBParams:
    """Variogram: E[(r_{t,t+k})^2] = 2a + b(1/n_t + 1/n_{t+k}) + k q, r = within-pitcher change minus the population
    curve's expected change. Non-negative least squares on winsorized squared residuals of same-era, non-2020 pairs
    with lags 1-3."""
    from scipy.optimize import nnls
    rows, ys = [], []
    for k in (1, 2, 3):
        pr = consecutive_pairs(hist[hist["season"] <= cutoff], k=k)
        if pr.empty:
            continue
        r = pr["delta"].values - curve.cumulative(pr["age"].values, k)
        r = r - np.nanmean(r)
        rows.append(np.column_stack([np.full(len(pr), 2.0), 1.0 / pr["n"].values + 1.0 / pr["n_to"].values,
                                     np.full(len(pr), float(k))]))
        ys.append(r ** 2)
    if not rows:
        v = float(hist["value"].var()) if len(hist) > 1 else 1.0
        return EBParams(v / 4, 0.0, v / 4, v, 0.0, 0.0, 0)
    A, y = np.vstack(rows), np.concatenate(ys)
    y = np.minimum(y, np.quantile(y, 0.99))
    (a, b, q), _ = nnls(A, y)
    cur = hist[hist["season"] == hist["season"].max()]
    noise = a + b * float(np.mean(1.0 / cur["n"].clip(lower=1)))
    tau2 = max(float(cur["value"].var()) - noise, 0.05 * float(cur["value"].var()))
    # Era shift (difference-in-differences of 2-year changes): 2019 -> 2021 vs same-era 2-year changes.
    shift, sv = 0.0, 0.0
    if cutoff >= 2021:
        allp = consecutive_pairs(hist[hist["season"] <= cutoff], k=2, clean=False)
        xr = allp[(allp["season"] == 2019) & (allp["season_to"] == 2021)]
        same = allp[~crosses_era(allp["season"], allp["season_to"]) & ~((allp["season"] <= 2020) &
                                                                          (allp["season_to"] >= 2020))]
        if len(xr) >= 20 and len(same) >= 20:
            rx = xr["delta"].values - curve.cumulative(xr["age"].values, 2)
            rs = same["delta"].values - curve.cumulative(same["age"].values, 2)
            shift = float(np.mean(rx) - np.mean(rs))
            sv = max(float(np.var(rx) - np.var(rs)), 0.0)
    return EBParams(float(a), float(b), float(q), float(tau2), shift, sv, int(len(y)))


def kalman_level(seasons, values, ns, ages, league: pd.Series, curve: AgeCurve, p: EBParams):
    """Filtered level and variance at the last observed season of one pitcher."""
    x, P, last, last_age = None, None, None, None
    for s, v, n, a in zip(seasons, values, ns, ages):
        R = p.season_var + p.sampling_coef / max(float(n), 1.0)
        if x is None:
            x, P = float(league.get(s, v)), p.prior_var
        else:
            gap = int(s - last)
            x = x + float(curve.cumulative(np.array([last_age]), gap)[0])
            P = P + gap * p.drift_var
            if crosses_era(last, s):
                x += p.era_shift
                P += p.era_var
        K = P / (P + R)
        x, P = x + K * (v - x), (1 - K) * P
        last, last_age = s, a
    return x, P


def eb_kalman(od: OriginData) -> pd.DataFrame:
    o = od.origin
    if o.empty:
        return _empty()
    curve = _curve(od, True)
    if "eb" not in od.cache:
        od.cache["eb"] = fit_eb(od.hist, od.cutoff, curve)
    p = od.cache["eb"]
    league = od.league_mean()
    hist = od.hist[od.hist["pitcher_id"].isin(set(o["pitcher_id"]))].sort_values("season")
    state = {}
    for pid, g in hist.groupby("pitcher_id"):
        state[pid] = kalman_level(g["season"].values, g["value"].values, g["n"].values, g["age"].values, league,
                                  curve, p)
    xs = np.array([state[pid][0] for pid in o["pitcher_id"]])
    Ps = np.array([state[pid][1] for pid in o["pitcher_id"]])
    out = []
    for h in od.horizons:
        mean = xs + curve.cumulative(o["age"].values, h)
        if od.cutoff <= TRACKMAN_LAST_SEASON < od.cutoff + h:
            mean = mean + p.era_shift
        var = Ps + h * p.drift_var + p.season_var + p.sampling_coef / o["n"].clip(lower=1).values
        sd = np.sqrt(var)
        out.append(_frame(o, h, mean, q10=mean - Z90 * sd, q50=mean, q90=mean + Z90 * sd, sd=sd))
    return pd.concat(out, ignore_index=True)


FUNCS = {"no_change": no_change, "marcel": marcel, "age_curve": lambda od: age_curve(od, False),
         "age_curve_ipw": lambda od: age_curve(od, True), "mixedlm": mixedlm, "lgbm_quantile": lgbm_quantile,
         "eb_kalman": eb_kalman}


def run_method(name: str, od: OriginData) -> pd.DataFrame:
    df = FUNCS[name](od)
    df["method"] = name
    return df


def fitted_state(od: OriginData) -> dict:
    """JSON-able parameters fitted at this origin (stored with the model artifacts)."""
    out: dict = {}
    for ipw in (False, True):
        if ("curve", ipw) in od.cache:
            out["age_curve_ipw" if ipw else "age_curve"] = od.cache[("curve", ipw)].to_dict()
    if "eb" in od.cache:
        out["eb_kalman"] = od.cache["eb"].to_dict()
    for key, v in od.cache.items():
        if isinstance(key, tuple) and key[0] == "marcel_k":
            out["marcel"] = {"k": v, "tuned_through_origin": key[1],
                             "k_scores_mae": od.cache.get(("marcel_k_scores", key[1]))}
    if "mixed" in od.cache:
        res, keep = od.cache["mixed"]
        names = [n for n, k in zip(MIXED_FE_NAMES, keep) if k]
        out["mixedlm"] = {"fe_params": dict(zip(names, map(float, np.asarray(res.fe_params)))),
                          "cov_re": np.asarray(res.cov_re, float).tolist(), "scale": float(res.scale),
                          "n_obs": int(res.nobs), "n_groups": int(len(res.random_effects)),
                          "re_terms": ["intercept", "(age-30)/10"]}
    if "lgbm" in od.cache:
        out["lgbm_quantile"] = {"params": LGBM_PARAMS, "features": LGBM_FEATURES, "quantiles": list(QUANTILES),
                                "n_train": od.cache.get("lgbm_n_train")}
    return out
