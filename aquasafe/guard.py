"""Input checks that run before the model sees a sample.

A model is only trustworthy on inputs that look like its training data. The
guard separates three cases:

  error    the reading cannot be real (non-numeric, outside physical bounds), or
           too many readings are missing to say anything. No prediction is made.
  warning  the reading is real but outside the range seen in training, or the
           combination of readings is unusual (Isolation Forest). A prediction is
           made and flagged as less reliable.
  info     a reading is missing and will be imputed.
"""

import math

import numpy as np
from sklearn.ensemble import IsolationForest

from . import config, schema

MAX_MISSING = 4
KEY_FEATURES = ("ph", "Sulfate")   # carry most of the signal (see the EDA)


class TrainingProfile:
    """What the training data looked like, kept inside the model artifact."""

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        self.median_ = np.nanmedian(X, axis=0)
        self.low_ = np.nanquantile(X, 0.005, axis=0)
        self.high_ = np.nanquantile(X, 0.995, axis=0)
        self.min_ = np.nanmin(X, axis=0)
        self.max_ = np.nanmax(X, axis=0)
        self.mean_ = np.nanmean(X, axis=0)
        self.std_ = np.nanstd(X, axis=0)
        Z = self._standardise(X)
        self.forest_ = IsolationForest(n_estimators=300, random_state=config.RANDOM_STATE).fit(Z)
        self.train_scores_ = np.sort(self.forest_.score_samples(Z))
        self.unusual_cutoff_ = float(np.quantile(self.train_scores_, 0.01))
        return self

    def _standardise(self, X):
        X = np.where(np.isnan(X), self.median_, X)
        return (X - self.mean_) / self.std_

    def typicality(self, X):
        """Percentile of each row's anomaly score among training rows (low = unusual)."""
        scores = self.forest_.score_samples(self._standardise(np.asarray(X, dtype=float)))
        pct = np.searchsorted(self.train_scores_, scores) / len(self.train_scores_)
        return scores, pct

    def as_dict(self):
        return {name: {"median": float(self.median_[i]), "p0_5": float(self.low_[i]),
                       "p99_5": float(self.high_[i]), "min": float(self.min_[i]),
                       "max": float(self.max_[i])}
                for i, name in enumerate(schema.FEATURE_NAMES)}


def _fmt(x):
    """Readable number for messages: 49,069 rather than 4.907e+04."""
    return f"{x:,.0f}" if abs(x) >= 1000 else f"{x:.4g}"


def _number(value):
    if value is None:
        return None
    if isinstance(value, str):
        value = value.strip()
        if value == "" or value.lower() in {"na", "nan", "null", "none"}:
            return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"not a number: {value!r}")
    if math.isnan(number):
        return None
    if math.isinf(number):
        raise ValueError("infinite value")
    return number


def check(record, profile):
    """Clean one record. Returns (values in schema order, issues, ok)."""
    issues, values = [], []
    for i, f in enumerate(schema.FEATURES):
        raw = record.get(f.name)
        try:
            v = _number(raw)
        except ValueError as exc:
            issues.append({"feature": f.name, "level": "error", "message": str(exc)})
            values.append(None)
            continue

        if v is None:
            level = "warning" if f.name in KEY_FEATURES else "info"
            issues.append({"feature": f.name, "level": level,
                           "message": f"{f.label} is missing and will be estimated from the "
                                      "other readings" + (", which weakens the prediction"
                                                           if level == "warning" else "")})
        elif not f.physical_min <= v <= f.physical_max:
            issues.append({"feature": f.name, "level": "error",
                           "message": f"{f.label} = {_fmt(v)} is outside the physically possible "
                                      f"range {f.physical_min:g} to {f.physical_max:g}"})
        elif not profile.low_[i] <= v <= profile.high_[i]:
            issues.append({"feature": f.name, "level": "warning",
                           "message": f"{f.label} = {_fmt(v)} is outside the range seen in training "
                                      f"({_fmt(profile.low_[i])} to {_fmt(profile.high_[i])}); "
                                      "the prediction is an extrapolation"})
        values.append(v)

    missing = sum(v is None for v in values)
    if missing > MAX_MISSING:
        issues.append({"feature": None, "level": "error",
                       "message": f"{missing} of 9 readings are missing; at least "
                                  f"{9 - MAX_MISSING} are needed"})

    ok = not any(i["level"] == "error" for i in issues)
    return values, issues, ok
