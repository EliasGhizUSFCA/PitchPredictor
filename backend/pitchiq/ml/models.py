"""Model families. All expose: fit, predict_proba (n x K, rows sum to 1), explain (optional).

Every model receives ONLY the pre-pitch feature frame produced by `features.build.make_features`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import warnings
from ..features.schema import default_model_features, feature_list
warnings.filterwarnings("ignore", message=".*eval_set.*")
from .metrics import log_loss


def _normalize(P: np.ndarray) -> np.ndarray:
    P = np.clip(P, 1e-12, None)
    return P / P.sum(axis=1, keepdims=True)


class TemperatureCalibrator:
    """Single-parameter temperature scaling of log-probabilities, fit on validation data only."""

    def __init__(self):
        self.T = 1.0

    def fit(self, P: np.ndarray, y_idx: np.ndarray) -> "TemperatureCalibrator":
        L = np.log(np.clip(P, 1e-12, 1))
        res = minimize_scalar(lambda t: log_loss(self._apply(L, t), y_idx), bounds=(0.25, 4.0), method="bounded")
        self.T = float(res.x)
        return self

    @staticmethod
    def _apply(L: np.ndarray, T: float) -> np.ndarray:
        Z = L / T
        Z -= Z.max(axis=1, keepdims=True)
        E = np.exp(Z)
        return E / E.sum(axis=1, keepdims=True)

    def transform(self, P: np.ndarray) -> np.ndarray:
        return self._apply(np.log(np.clip(P, 1e-12, 1)), self.T)


class BaseModel:
    family = "base"

    def __init__(self, classes: list[str], features: list[str] | None = None, **params):
        self.classes = list(classes)
        all_feats = feature_list(self.classes)  # current version (a superset of v1: kinds cover v1 models)
        self.features = features or default_model_features(self.classes)
        kinds = {f.name: f.kind for f in all_feats}
        self.categorical = [f for f in self.features if kinds.get(f) == "categorical"]
        self.numeric = [f for f in self.features if kinds.get(f) != "categorical"]
        self.params = params
        self.calibrator: TemperatureCalibrator | None = None
        self.decay_half_life_days: float | None = None  # set by protocol.make_model; weights via ml/decay.py

    def y_index(self, y: pd.Series) -> np.ndarray:
        m = {c: i for i, c in enumerate(self.classes)}
        return y.map(m).to_numpy()

    def _raw_proba(self, X: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError

    def predict_proba(self, X: pd.DataFrame, calibrated: bool = True) -> np.ndarray:
        P = _normalize(self._raw_proba(X))
        if calibrated and self.calibrator is not None:
            P = self.calibrator.transform(P)
        return P

    def calibrate(self, X_val: pd.DataFrame, y_val: pd.Series) -> dict:
        P = self.predict_proba(X_val, calibrated=False)
        y = self.y_index(y_val)
        before = log_loss(P, y)
        self.calibrator = TemperatureCalibrator().fit(P, y)
        after = log_loss(self.calibrator.transform(P), y)
        return {"method": "temperature_scaling", "fit_split": "validation", "temperature": self.calibrator.T,
                "val_log_loss_before": before, "val_log_loss_after": after}

    def explain(self, X: pd.DataFrame) -> list[dict] | None:
        return None

    def config(self) -> dict:
        cfg = {"family": self.family, "features": self.features, "params": self.params}
        if getattr(self, "decay_half_life_days", None) is not None:
            cfg["decay_half_life_days"] = self.decay_half_life_days
        return cfg


class MajorityClassModel(BaseModel):
    family = "majority"

    def fit(self, X, y, X_val=None, y_val=None, sample_weight=None):
        w = np.ones(len(y)) if sample_weight is None else np.asarray(sample_weight, dtype=float)
        counts = pd.Series(w, index=y.index).groupby(y.to_numpy()).sum()
        self.prior = np.array([counts.get(c, 0) for c in self.classes], dtype=float) + 1.0
        self.prior /= self.prior.sum()
        return self

    def _raw_proba(self, X):
        return np.tile(self.prior, (len(X), 1))


class FrequencyBaseline(BaseModel):
    """Hierarchically smoothed historical frequencies.

    P(c | key_L) = (n_L(c) + alpha * P(c | key_{L-1})) / (n_L + alpha), backing off through levels
    such as () -> (stand) -> (stand, count) -> (stand, count, prev1_type).
    """
    family = "frequency"
    DEFAULT_LEVELS = [["stand"], ["stand", "count_str"], ["stand", "count_str", "prev1_type"]]

    def __init__(self, classes, features=None, levels=None, alpha: float = 20.0):
        levels = levels or self.DEFAULT_LEVELS
        feats = sorted({f for lvl in levels for f in lvl})
        super().__init__(classes, features=feats, levels=levels, alpha=alpha)
        self.levels, self.alpha = levels, alpha

    def fit(self, X, y, X_val=None, y_val=None, sample_weight=None):
        k = len(self.classes)
        yi = self.y_index(y)
        w = np.ones(len(yi)) if sample_weight is None else np.asarray(sample_weight, dtype=float)
        self.root = (np.bincount(yi, weights=w, minlength=k) + 1.0) / (w.sum() + k)
        self.tables = []
        for lvl in self.levels:
            keys = X[lvl].astype(str).agg("|".join, axis=1).to_numpy()
            table: dict[str, np.ndarray] = {}
            for key, c, wi in zip(keys, yi, w):
                table.setdefault(key, np.zeros(k))[c] += wi
            self.tables.append(table)
        return self

    def _raw_proba(self, X):
        out = np.zeros((len(X), len(self.classes)))
        keys_by_level = [X[lvl].astype(str).agg("|".join, axis=1).to_numpy() for lvl in self.levels]
        for i in range(len(X)):
            p = self.root
            for L, table in enumerate(self.tables):
                cnt = table.get(keys_by_level[L][i])
                if cnt is not None:
                    p = (cnt + self.alpha * p) / (cnt.sum() + self.alpha)
            out[i] = p
        return out

    def support(self, row: pd.DataFrame) -> list[dict]:
        """Historical sample size behind each backoff level for one row."""
        res = []
        for lvl, table in zip(self.levels, self.tables):
            key = "|".join(row[lvl].astype(str).iloc[0])
            cnt = table.get(key)
            res.append({"level": lvl, "key": key, "n": int(cnt.sum()) if cnt is not None else 0})
        return res


class LogRegModel(BaseModel):
    family = "logreg"

    def __init__(self, classes, features=None, C: float = 0.3):
        super().__init__(classes, features=features, C=C)
        self.C = C

    def _pipe(self):
        pre = ColumnTransformer([
            ("cat", OneHotEncoder(handle_unknown="ignore"), self.categorical),
            ("num", Pipeline([("imp", SimpleImputer(strategy="median", add_indicator=True)),
                              ("sc", StandardScaler())]), self.numeric),
        ])
        return Pipeline([("pre", pre), ("clf", LogisticRegression(C=self.C, max_iter=3000))])

    def _X(self, X):
        X = X[self.features].copy()
        for c in self.categorical:
            X[c] = X[c].astype(str)
        for c in self.numeric:
            X[c] = pd.to_numeric(X[c], errors="coerce").astype(float)
        return X

    def fit(self, X, y, X_val=None, y_val=None, sample_weight=None):
        kw = {} if sample_weight is None else {"clf__sample_weight": np.asarray(sample_weight, dtype=float)}
        self.pipe = self._pipe().fit(self._X(X), y, **kw)
        order = list(self.pipe.classes_)
        self._order = [order.index(c) for c in self.classes]
        return self

    def _raw_proba(self, X):
        return self.pipe.predict_proba(self._X(X))[:, self._order]

    def explain(self, X):
        """Exact linear attributions in log-odds space: coef_k * standardized_x, summed per source feature."""
        pre, clf = self.pipe.named_steps["pre"], self.pipe.named_steps["clf"]
        Z = pre.transform(self._X(X))
        Z = Z.toarray() if hasattr(Z, "toarray") else Z
        names = pre.get_feature_names_out()
        src = []
        for n in names:
            kind, rest = n.split("__", 1)
            if kind == "cat":
                src.append(next(c for c in self.categorical if rest.startswith(c + "_")))
            else:
                src.append(rest.replace("missingindicator_", ""))
        order = list(clf.classes_)
        out = []
        for i in range(len(Z)):
            per_class = {}
            for c in self.classes:
                contrib = clf.coef_[order.index(c)] * Z[i]
                agg: dict[str, float] = {}
                for s, v in zip(src, contrib):
                    agg[s] = agg.get(s, 0.0) + float(v)
                per_class[c] = {"base": float(clf.intercept_[order.index(c)]), "contributions": agg}
            out.append({"space": "log_odds (multinomial logit)", "per_class": per_class})
        return out


class LightGBMModel(BaseModel):
    family = "lightgbm"
    DEFAULTS = dict(n_estimators=2000, learning_rate=0.02, num_leaves=15, min_child_samples=40,
                    subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=1.0)

    def __init__(self, classes, features=None, **params):
        p = {**self.DEFAULTS, **params}
        super().__init__(classes, features=features, **p)

    def _X(self, X):
        X = X[self.features].copy()
        for c in self.categorical:
            v = X[c].astype(str)  # categories unseen in training (e.g. a new catcher) -> missing
            X[c] = pd.Categorical(v.where(v.isin(self.categories_[c])), categories=self.categories_[c])
        for c in self.numeric:
            X[c] = pd.to_numeric(X[c], errors="coerce").astype(float)
        return X

    def fit(self, X, y, X_val=None, y_val=None, sample_weight=None):
        import lightgbm as lgb
        self.categories_ = {c: sorted(X[c].astype(str).unique().tolist()) for c in self.categorical}
        p = dict(self.params)
        p.setdefault("random_state", 0)
        self.model = lgb.LGBMClassifier(objective="multiclass", verbose=-1, **p)
        fit_kw = {}
        if X_val is not None:
            fit_kw = dict(eval_set=[(self._X(X_val), self.y_index(y_val))], eval_metric="multi_logloss",
                          callbacks=[lgb.early_stopping(100, verbose=False)])
        if sample_weight is not None:
            fit_kw["sample_weight"] = np.asarray(sample_weight, dtype=float)
        self.model.fit(self._X(X), self.y_index(y), **fit_kw)
        self.best_iteration_ = int(self.model.best_iteration_ or p["n_estimators"])
        return self

    def _raw_proba(self, X):
        return self.model.predict_proba(self._X(X), num_iteration=self.best_iteration_)

    def explain(self, X):
        """TreeSHAP (LightGBM native pred_contrib) in raw multiclass margin space."""
        C = self.model.predict(self._X(X), pred_contrib=True, num_iteration=self.best_iteration_)
        C = np.asarray(C)
        k, f = len(self.classes), len(self.features)
        C = C.reshape(len(X), k, f + 1)
        out = []
        for i in range(len(X)):
            per_class = {c: {"base": float(C[i, j, -1]),
                             "contributions": {self.features[m]: float(C[i, j, m]) for m in range(f)}}
                         for j, c in enumerate(self.classes)}
            out.append({"space": "raw margin (softmax logit), TreeSHAP", "per_class": per_class})
        return out

    def config(self):
        return {**super().config(), "best_iteration": getattr(self, "best_iteration_", None)}


FAMILIES = {"majority": MajorityClassModel, "frequency": FrequencyBaseline, "logreg": LogRegModel,
            "lightgbm": LightGBMModel}
