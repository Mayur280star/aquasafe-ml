"""Custom preprocessing steps. Every one is fitted on training folds only.

Column layout contract: the nine base readings are always the first nine
columns, in schema order. Steps that add columns (missing indicators, domain
flags, interactions) append them on the right, so later steps can still find
the base readings by position.
"""

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import IsolationForest
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer
from sklearn.linear_model import BayesianRidge

from . import schema

N_BASE = len(schema.FEATURES)
LOWER = np.array([f.physical_min for f in schema.FEATURES])
UPPER = np.array([f.physical_max for f in schema.FEATURES])


def _array(X):
    return np.asarray(X, dtype=float).copy()


class PhysicalBounds(BaseEstimator, TransformerMixin):
    """Readings that cannot physically exist become missing values."""

    def fit(self, X, y=None):
        self.n_features_in_ = np.asarray(X).shape[1]
        return self

    def transform(self, X):
        X = _array(X)
        base = X[:, :N_BASE]
        base[(base < LOWER) | (base > UPPER)] = np.nan
        return X


class Imputer(BaseEstimator, TransformerMixin):
    """Missing value strategies behind one interface.

    strategy
        median, mean, knn, iterative : one method for every column
        hybrid : model-based (iterative) imputation for the informative, heavily
                 missing pH and Sulfate; median for the rest
        none   : leave NaN in place, for models that handle it natively
    add_indicator
        append a 0/1 column for each feature that had gaps in training
    """

    MODEL_BASED = ("ph", "Sulfate")

    def __init__(self, strategy="median", add_indicator=False, random_state=0):
        self.strategy = strategy
        self.add_indicator = add_indicator
        self.random_state = random_state

    def _make(self, strategy):
        if strategy in ("median", "mean"):
            return SimpleImputer(strategy=strategy, keep_empty_features=True)
        if strategy == "knn":
            return KNNImputer(n_neighbors=10, weights="distance", keep_empty_features=True)
        if strategy == "iterative":
            return IterativeImputer(estimator=BayesianRidge(), max_iter=15,
                                    random_state=self.random_state,
                                    keep_empty_features=True)
        raise ValueError(f"unknown imputation strategy: {strategy}")

    def fit(self, X, y=None):
        X = _array(X)
        self.n_features_in_ = X.shape[1]
        self.indicator_cols_ = np.where(np.isnan(X[:, :N_BASE]).any(axis=0))[0]

        if self.strategy == "none":
            return self
        if self.strategy == "hybrid":
            names = schema.FEATURE_NAMES
            self.model_cols_ = [names.index(n) for n in self.MODEL_BASED]
            self.simple_cols_ = [i for i in range(N_BASE) if i not in self.model_cols_]
            self.simple_ = self._make("median").fit(X[:, self.simple_cols_])
            filled = X.copy()
            filled[:, self.simple_cols_] = self.simple_.transform(X[:, self.simple_cols_])
            self.model_ = self._make("iterative").fit(filled[:, :N_BASE])
        else:
            self.model_ = self._make(self.strategy).fit(X[:, :N_BASE])
        return self

    def transform(self, X):
        X = _array(X)
        mask = np.isnan(X[:, :N_BASE])
        if self.strategy == "hybrid":
            X[:, self.simple_cols_] = self.simple_.transform(X[:, self.simple_cols_])
            X[:, :N_BASE] = self.model_.transform(X[:, :N_BASE])
        elif self.strategy != "none":
            X[:, :N_BASE] = self.model_.transform(X[:, :N_BASE])
        if self.add_indicator and len(self.indicator_cols_):
            X = np.hstack([X, mask[:, self.indicator_cols_].astype(float)])
        return X


class Winsorizer(BaseEstimator, TransformerMixin):
    """Clip each base reading to bounds learned from the training fold.

    method 'quantile' clips at the (q, 1 - q) quantiles; 'iqr' clips at the
    Tukey fences Q1 - k*IQR and Q3 + k*IQR.
    """

    def __init__(self, method="quantile", q=0.01, k=1.5):
        self.method = method
        self.q = q
        self.k = k

    def fit(self, X, y=None):
        base = _array(X)[:, :N_BASE]
        if self.method == "quantile":
            self.low_ = np.nanquantile(base, self.q, axis=0)
            self.high_ = np.nanquantile(base, 1 - self.q, axis=0)
        elif self.method == "iqr":
            q1, q3 = np.nanquantile(base, [0.25, 0.75], axis=0)
            self.low_, self.high_ = q1 - self.k * (q3 - q1), q3 + self.k * (q3 - q1)
        else:
            raise ValueError(f"unknown winsorizing method: {self.method}")
        return self

    def transform(self, X):
        X = _array(X)
        X[:, :N_BASE] = np.clip(X[:, :N_BASE], self.low_, self.high_)
        return X


class DomainFeatures(BaseEstimator, TransformerMixin):
    """Guideline exceedance flags, their count, and the TDS / conductivity ratio.

    Runs on unscaled readings. In real water, TDS is about 0.5 to 0.7 times the
    conductivity, so the ratio is a physical consistency check.
    """

    def fit(self, X, y=None):
        self.n_features_in_ = np.asarray(X).shape[1]
        return self

    def transform(self, X):
        X = _array(X)
        flags = []
        for i, f in enumerate(schema.FEATURES):
            column = X[:, i]
            out = np.zeros_like(column)
            if f.guideline_min is not None:
                out = np.maximum(out, column < f.guideline_min)
            if f.guideline_max is not None:
                out = np.maximum(out, column > f.guideline_max)
            flags.append(np.where(np.isnan(column), 0.0, out))
        flags = np.column_stack(flags)
        solids = X[:, schema.FEATURE_NAMES.index("Solids")]
        conductivity = X[:, schema.FEATURE_NAMES.index("Conductivity")]
        ratio = np.log1p(solids) - np.log1p(conductivity)
        return np.hstack([X, flags.sum(axis=1, keepdims=True), ratio[:, None]])


class InteractionFeatures(BaseEstimator, TransformerMixin):
    """Products z_i * z_j for the pairs whose correlation differs most by class.

    This is the one supervised feature step. It looks at y only through the
    training fold passed to fit, so the pairs it picks never see validation or
    test rows. Expects standardised base readings.
    """

    def __init__(self, k=5):
        self.k = k

    def fit(self, X, y):
        base = _array(X)[:, :N_BASE]
        y = np.asarray(y)
        scores = []
        for i in range(N_BASE):
            for j in range(i + 1, N_BASE):
                diff = [self._corr(base[y == c, i], base[y == c, j]) for c in (0, 1)]
                scores.append((abs(diff[1] - diff[0]), i, j))
        scores.sort(reverse=True)
        self.pairs_ = [(i, j) for _, i, j in scores[:self.k]]
        self.pair_scores_ = [s for s, _, _ in scores[:self.k]]
        return self

    @staticmethod
    def _corr(a, b):
        keep = ~(np.isnan(a) | np.isnan(b))
        if keep.sum() < 3:
            return 0.0
        return float(np.corrcoef(a[keep], b[keep])[0, 1])

    def transform(self, X):
        X = _array(X)
        if not self.k:
            return X
        products = np.column_stack([X[:, i] * X[:, j] for i, j in self.pairs_])
        return np.hstack([X, products])

    def pair_names(self):
        names = schema.FEATURE_NAMES
        return [f"{names[i]} x {names[j]}" for i, j in self.pairs_]


class DeviationFeatures(BaseEstimator, TransformerMixin):
    """|z| for each base reading, the distance from a typical value."""

    def fit(self, X, y=None):
        self.n_features_in_ = np.asarray(X).shape[1]
        return self

    def transform(self, X):
        X = _array(X)
        return np.hstack([X, np.abs(X[:, :N_BASE])])


class Passthrough(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return _array(X)


def remove_outliers(X, y, contamination=0.03, random_state=0):
    """Drop Isolation Forest outliers from a training fold.

    Wrapped in imblearn's FunctionSampler, so it only ever removes training rows;
    validation and test rows pass through untouched.
    """
    X = _array(X)
    filled = np.where(np.isnan(X), np.nanmedian(X, axis=0), X)
    forest = IsolationForest(contamination=contamination,
                             random_state=random_state).fit(filled)
    keep = forest.predict(filled) == 1
    return X[keep], np.asarray(y)[keep]
