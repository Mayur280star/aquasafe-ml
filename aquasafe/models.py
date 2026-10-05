"""The candidate classifiers, each with a sensible starting configuration."""

import numpy as np
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import (ExtraTreesClassifier, HistGradientBoostingClassifier,
                              RandomForestClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

from . import config

SEED = config.RANDOM_STATE
POSITIVE_RATE = 0.39   # used to turn "balanced" into scale_pos_weight for XGBoost

NAMES = {
    "logreg": "Logistic Regression",
    "knn": "K-Nearest Neighbours",
    "gnb": "Gaussian Naive Bayes",
    "qda": "Quadratic Discriminant Analysis",
    "svm": "Support Vector Machine (RBF)",
    "mlp": "Neural Network (MLP)",
    "rf": "Random Forest",
    "et": "Extra Trees",
    "hgb": "Histogram Gradient Boosting",
    "xgb": "XGBoost",
    "lgbm": "LightGBM",
    "catboost": "CatBoost",
}

# Models that accept NaN inputs, so they can skip imputation entirely.
NATIVE_NAN = {"hgb", "xgb", "lgbm", "catboost"}
# Models whose output is unaffected by monotone rescaling of each feature.
SCALE_FREE = {"rf", "et", "hgb", "xgb", "lgbm", "catboost"}


def make(name, params=None, class_weight=False):
    """A fresh, unfitted estimator. class_weight=True applies inverse-frequency weights."""
    params = dict(params or {})
    weight = "balanced" if class_weight else None

    if name == "logreg":
        return LogisticRegression(max_iter=5000, class_weight=weight,
                                  **{"C": 1.0, **params})
    if name == "knn":
        return KNeighborsClassifier(**{"n_neighbors": 25, "weights": "distance", **params})
    if name == "gnb":
        return GaussianNB(**params)
    if name == "qda":
        return QuadraticDiscriminantAnalysis(**{"reg_param": 0.05, **params})
    if name == "svm":
        return SVC(kernel="rbf", class_weight=weight, random_state=SEED,
                   **{"C": 1.0, "gamma": "scale", **params})
    if name == "mlp":
        return MLPClassifier(max_iter=2000, early_stopping=True, random_state=SEED,
                             **{"hidden_layer_sizes": (64, 32), "alpha": 1e-2, **params})
    if name == "rf":
        return RandomForestClassifier(n_jobs=1, random_state=SEED, class_weight=weight,
                                      **{"n_estimators": 500, "min_samples_leaf": 2, **params})
    if name == "et":
        return ExtraTreesClassifier(n_jobs=1, random_state=SEED, class_weight=weight,
                                    **{"n_estimators": 800, "min_samples_leaf": 2, **params})
    if name == "hgb":
        return HistGradientBoostingClassifier(random_state=SEED, class_weight=weight,
                                              **{"learning_rate": 0.05, "max_iter": 300,
                                                 "max_leaf_nodes": 15, **params})
    if name == "xgb":
        spw = (1 - POSITIVE_RATE) / POSITIVE_RATE if class_weight else 1.0
        return XGBClassifier(random_state=SEED, n_jobs=1, scale_pos_weight=spw,
                             eval_metric="logloss", tree_method="hist",
                             **{"n_estimators": 400, "learning_rate": 0.03, "max_depth": 5,
                                "subsample": 0.8, "colsample_bytree": 0.8, **params})
    if name == "lgbm":
        return LGBMClassifier(random_state=SEED, n_jobs=1, verbose=-1, class_weight=weight,
                              **{"n_estimators": 400, "learning_rate": 0.03, "num_leaves": 15,
                                 "subsample": 0.8, "subsample_freq": 1,
                                 "colsample_bytree": 0.8, **params})
    if name == "catboost":
        return CatBoostClassifier(random_seed=SEED, verbose=0, thread_count=1,
                                  allow_writing_files=False,
                                  auto_class_weights="Balanced" if class_weight else None,
                                  **{"iterations": 600, "learning_rate": 0.03, "depth": 6,
                                     **params})
    raise ValueError(f"unknown model: {name}")


def supports_class_weight(name):
    return name not in {"knn", "gnb", "qda", "mlp"}


def scores_of(estimator, X):
    """Probability of the positive class, or a decision score if there is none."""
    if hasattr(estimator, "predict_proba"):
        return estimator.predict_proba(X)[:, 1]
    return np.asarray(estimator.decision_function(X))
