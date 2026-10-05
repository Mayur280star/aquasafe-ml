"""Thresholds, held-out evaluation, robustness checks and the leakage demonstration."""

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE, RandomOverSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
                             brier_score_loss, confusion_matrix, f1_score, log_loss,
                             matthews_corrcoef, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split

from . import config, schema

GRID = np.round(np.arange(0.05, 0.951, 0.005), 3)


# ----------------------------------------------------------------- thresholds

def threshold_table(y, p):
    rows = []
    y = np.asarray(y)
    for t in GRID:
        pred = (p >= t).astype(int)
        above, below = p >= t, p <= t
        rows.append({
            "threshold": t,
            "accuracy": accuracy_score(y, pred),
            "precision": precision_score(y, pred, zero_division=0),
            "recall": recall_score(y, pred, zero_division=0),
            "f1": f1_score(y, pred, zero_division=0),
            "share_above": above.mean(),
            "precision_above": y[above].mean() if above.sum() else np.nan,
            "n_above": int(above.sum()),
            "npv_below": 1 - y[below].mean() if below.sum() else np.nan,
            "n_below": int(below.sum()),
        })
    return pd.DataFrame(rows)


def choose_thresholds(y, p, min_support=25):
    """Operating points picked on out-of-fold predictions, never on the test set.

    decision     the single cut that maximises accuracy (the binary label)
    potable      the lowest cut whose 'potable' calls reach the target precision
    not_potable  the highest cut whose 'not potable' calls reach the target NPV
    Between the last two the service answers 'uncertain, test in a laboratory'.
    """
    table = threshold_table(y, p)
    # A seven-step rolling mean stops the cut from landing on a single lucky step.
    smooth = table["accuracy"].rolling(7, center=True, min_periods=1).mean()
    decision = float(table.loc[smooth.idxmax(), "threshold"])

    ok = table[(table["precision_above"] >= config.TARGET_PRECISION_POTABLE) &
               (table["n_above"] >= min_support) & (table["threshold"] >= decision)]
    potable = float(ok["threshold"].min()) if len(ok) else float(
        table[table["n_above"] >= min_support]["threshold"].max())

    ok = table[(table["npv_below"] >= config.TARGET_PRECISION_UNSAFE) &
               (table["n_below"] >= min_support) & (table["threshold"] <= decision)]
    not_potable = float(ok["threshold"].max()) if len(ok) else float(
        table[table["n_below"] >= min_support]["threshold"].min())

    return {"decision": decision, "potable": max(potable, decision),
            "not_potable": min(not_potable, decision)}, table


# ------------------------------------------------------------------- metrics

def metrics(y, p, threshold):
    y = np.asarray(y)
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "accuracy": accuracy_score(y, pred),
        "balanced_accuracy": balanced_accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "specificity": tn / (tn + fp) if tn + fp else np.nan,
        "f1": f1_score(y, pred, zero_division=0),
        "mcc": matthews_corrcoef(y, pred),
        "roc_auc": roc_auc_score(y, p),
        "pr_auc": average_precision_score(y, p),
        "brier": brier_score_loss(y, p),
        "log_loss": log_loss(y, np.clip(p, 1e-6, 1 - 1e-6)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def three_way(y, p, thresholds):
    """How the potable / uncertain / not-potable verdicts perform."""
    y = np.asarray(y)
    hi, lo = p >= thresholds["potable"], p <= thresholds["not_potable"]
    decided = hi | lo
    correct = (hi & (y == 1)) | (lo & (y == 0))
    return {
        "coverage": float(decided.mean()),
        "n_potable_calls": int(hi.sum()),
        "n_not_potable_calls": int(lo.sum()),
        "n_uncertain": int((~decided).sum()),
        "precision_of_potable_calls": float(y[hi].mean()) if hi.any() else np.nan,
        "precision_of_not_potable_calls": float(1 - y[lo].mean()) if lo.any() else np.nan,
        "accuracy_on_decided": float(correct[decided].mean()) if decided.any() else np.nan,
    }


def bootstrap(y, p, threshold, n=2000, seed=config.RANDOM_STATE):
    """Percentile 95% intervals for every metric, by resampling the test rows."""
    rng = np.random.default_rng(seed)
    y, p = np.asarray(y), np.asarray(p)
    draws = []
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) < 2:
            continue
        draws.append(metrics(y[idx], p[idx], threshold))
    frame = pd.DataFrame(draws)
    return {c: (float(frame[c].quantile(0.025)), float(frame[c].quantile(0.975)))
            for c in ["accuracy", "balanced_accuracy", "precision", "recall", "specificity",
                      "f1", "mcc", "roc_auc", "pr_auc", "brier"]}


def selective_curve(y, p):
    """Accuracy when the model answers only its most confident share of samples."""
    y = np.asarray(y)
    confidence = np.abs(p - 0.5)
    order = np.argsort(-confidence)
    pred = (p >= 0.5).astype(int)
    rows = []
    for share in np.linspace(0.1, 1.0, 19):
        k = max(1, int(round(share * len(y))))
        idx = order[:k]
        rows.append({"coverage": share, "accuracy": float((pred[idx] == y[idx]).mean())})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- robustness

def noise_robustness(model, X, y, levels=(0, 0.01, 0.02, 0.05, 0.1, 0.2), repeats=5,
                     threshold=0.5):
    """Multiplicative Gaussian sensor noise on every reading."""
    rng = np.random.default_rng(config.RANDOM_STATE)
    rows = []
    for s in levels:
        aucs, accs = [], []
        for _ in range(repeats if s else 1):
            noisy = X * (1 + rng.normal(0, s, X.shape)) if s else X
            p = model.predict_proba(noisy)[:, 1]
            aucs.append(roc_auc_score(y, p))
            accs.append(accuracy_score(y, (p >= threshold).astype(int)))
        rows.append({"noise_sd": s, "roc_auc": np.mean(aucs), "accuracy": np.mean(accs)})
    return pd.DataFrame(rows)


def missing_robustness(model, X, y, threshold=0.5):
    """Score with one reading (or pH and Sulfate together) removed from every sample."""
    def score(frame):
        p = model.predict_proba(frame)[:, 1]
        return roc_auc_score(y, p), accuracy_score(y, (p >= threshold).astype(int))

    rows = [{"removed": "nothing", **dict(zip(["roc_auc", "accuracy"], score(X)))}]
    for f in schema.FEATURE_NAMES:
        rows.append({"removed": f, **dict(zip(["roc_auc", "accuracy"],
                                              score(X.assign(**{f: np.nan}))))})
    rows.append({"removed": "ph + Sulfate", **dict(zip(["roc_auc", "accuracy"], score(
        X.assign(ph=np.nan, Sulfate=np.nan))))})
    return pd.DataFrame(rows)


def importance(model, X, y):
    result = permutation_importance(model, X, y, scoring="roc_auc", n_repeats=30,
                                    random_state=config.RANDOM_STATE, n_jobs=-1)
    return pd.DataFrame({"feature": schema.FEATURE_NAMES,
                         "importance": result.importances_mean,
                         "std": result.importances_std}).sort_values("importance",
                                                                     ascending=False)


# ------------------------------------------------------------ leakage demo

def leakage_demo(frame, honest_cv_accuracy, honest_test_accuracy):
    """Reproduce the usual mistakes to show where 85-95% claims come from.

    Every leaky protocol trains the same Random Forest. Only the protocol changes.
    """
    X = frame[schema.FEATURE_NAMES]
    y = frame[config.TARGET]
    seed = config.RANDOM_STATE

    def rf():
        return RandomForestClassifier(n_estimators=500, random_state=seed, n_jobs=-1)

    def fit_score(Xa, ya, Xb, yb):
        return accuracy_score(yb, rf().fit(Xa, ya).predict(Xb))

    rows = [
        {"protocol": "Honest: tuned model, cross-validation on training split",
         "accuracy": honest_cv_accuracy, "leaky": False},
        {"protocol": "Honest: tuned model, untouched test split",
         "accuracy": honest_test_accuracy, "leaky": False},
    ]

    # 1. Fill gaps with the mean of the row's own class, over the whole dataset.
    filled = X.copy()
    for c in X.columns:
        filled[c] = X[c].fillna(X.groupby(y)[c].transform("mean"))
    Xa, Xb, ya, yb = train_test_split(filled, y, test_size=0.2, stratify=y, random_state=seed)
    rows.append({"protocol": "Class-wise mean imputation before the split",
                 "accuracy": fit_score(Xa, ya, Xb, yb), "leaky": True})

    median = SimpleImputer(strategy="median").fit_transform(X)

    # 2. Oversample the minority class by duplication, then split.
    Xo, yo = RandomOverSampler(random_state=seed).fit_resample(median, y)
    Xa, Xb, ya, yb = train_test_split(Xo, yo, test_size=0.2, stratify=yo, random_state=seed)
    rows.append({"protocol": "Random oversampling before the split",
                 "accuracy": fit_score(Xa, ya, Xb, yb), "leaky": True})

    # 3. SMOTE, then split.
    Xs, ys = SMOTE(random_state=seed).fit_resample(median, y)
    Xa, Xb, ya, yb = train_test_split(Xs, ys, test_size=0.2, stratify=ys, random_state=seed)
    rows.append({"protocol": "SMOTE before the split",
                 "accuracy": fit_score(Xa, ya, Xb, yb), "leaky": True})

    # 4. Try many random splits and report the luckiest one.
    best = 0.0
    for s in range(40):
        Xa, Xb, ya, yb = train_test_split(median, y, test_size=0.2, stratify=y, random_state=s)
        best = max(best, accuracy_score(yb, RandomForestClassifier(
            n_estimators=300, random_state=s, n_jobs=-1).fit(Xa, ya).predict(Xb)))
    rows.append({"protocol": "Best of 40 random splits (test set used for selection)",
                 "accuracy": best, "leaky": True})

    rows.append({"protocol": "Majority-class baseline (always 'not potable')",
                 "accuracy": 1 - y.mean(), "leaky": False})
    return pd.DataFrame(rows)
