"""Staged model selection on the training split, by repeated stratified CV.

Each preprocessing decision is made in its own stage, against fixed reference
models, and carried forward: imputation, then outliers, scaling, features and
class balancing. Then every classifier is compared under the chosen
preprocessing, the strongest are tuned with Optuna, and ensembles of the tuned
models are tried.

Decisions use a one-standard-error rule. An option replaces the stage default
only if its paired fold-by-fold ROC AUC gain over the default is larger than
the standard error of that gain. Otherwise the simpler default is kept, because
a smaller gain is indistinguishable from fold noise.
"""

import json
import time
import warnings

import numpy as np
import optuna
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import StackingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (RepeatedStratifiedKFold, StratifiedKFold,
                                     cross_validate)

from . import config, models, recipe as R

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

SCORING = ["roc_auc", "accuracy", "balanced_accuracy", "precision", "recall", "f1"]


def splitter(repeats=config.CV_REPEATS, seed=config.RANDOM_STATE):
    return RepeatedStratifiedKFold(n_splits=config.CV_FOLDS, n_repeats=repeats,
                                   random_state=seed)


def cv_estimator(estimator, X, y, repeats=config.CV_REPEATS):
    """Cross-validate any estimator. Returns summary metrics and the AUC per fold."""
    started = time.time()
    out = cross_validate(estimator, X, y, cv=splitter(repeats), scoring=SCORING,
                         n_jobs=-1, error_score="raise")
    summary = {}
    for m in SCORING:
        summary[m] = float(np.mean(out[f"test_{m}"]))
        summary[f"{m}_sd"] = float(np.std(out[f"test_{m}"]))
    summary["seconds"] = round(time.time() - started, 1)
    return summary, out["test_roc_auc"]


def cv_recipe(recipe, X, y, repeats=config.CV_REPEATS):
    return cv_estimator(R.build(recipe), X, y, repeats)


# ------------------------------------------------------------ staged decisions

def run_stage(name, options, references, base, X, y, log, decide_on=None):
    """Score every option under every reference model, then apply the 1-SE rule.

    options    list of (label, recipe overrides); the first one is the default
    references list of recipe overrides naming the reference models
    base       the recipe settings decided by earlier stages
    decide_on  reference models that vote on the decision (default: all). Others
               are scored for illustration only.
    """
    log(f"\n  STAGE: {name}")
    rows, folds = [], {}
    for label, overrides in options:
        for ref in references:
            r = {**base, **ref, **overrides}
            summary, auc = cv_recipe(r, X, y)
            folds[(label, ref["model"])] = auc
            rows.append({"stage": name, "option": label, "reference": ref["model"],
                         **summary})
            log(f"    {label:<26} {ref['model']:<7} auc={summary['roc_auc']:.4f}"
                f" acc={summary['accuracy']:.4f} rec={summary['recall']:.3f}"
                f" ({summary['seconds']}s)")

    table = pd.DataFrame(rows)
    default = options[0][0]
    ref_names = decide_on or [ref["model"] for ref in references]

    decision = []
    for label, _ in options:
        gains = np.mean([folds[(label, m)] - folds[(default, m)] for m in ref_names], axis=0)
        se = gains.std(ddof=1) / np.sqrt(len(gains)) if gains.any() else 0.0
        decision.append({"option": label, "mean_auc": float(np.mean(
            [folds[(label, m)].mean() for m in ref_names])),
            "gain_vs_default": float(gains.mean()), "gain_se": float(se)})
    decision = pd.DataFrame(decision).sort_values("mean_auc", ascending=False)

    best = decision.iloc[0]
    significant = best["gain_vs_default"] > best["gain_se"]
    chosen = best["option"] if significant else default
    reason = (f"{best['option']} beats {default} by {best['gain_vs_default']:+.4f} AUC "
              f"(SE {best['gain_se']:.4f}), " +
              ("adopted" if significant else f"within noise, keeping {default}"))
    if best["option"] == default:
        reason = f"{default} is best outright"
    log(f"    -> {chosen}: {reason}")

    table.to_csv(config.TABLE_DIR / f"stage_{name.lower().replace(' ', '_')}.csv", index=False)
    return chosen, dict(options)[chosen], table, decision, reason


def staged_selection(X, y, log=print):
    refs = [{"model": "svm"}, {"model": "rf"}]
    base = {}
    decisions = []

    stages = [
        ("Imputation", [
            ("median", {"imputer": "median"}),
            ("mean", {"imputer": "mean"}),
            ("median + indicators", {"imputer": "median", "indicator": True}),
            ("knn", {"imputer": "knn"}),
            ("iterative (MICE)", {"imputer": "iterative"}),
            ("hybrid", {"imputer": "hybrid"}),
            ("hybrid + indicators", {"imputer": "hybrid", "indicator": True}),
        ], refs),
        ("Outliers", [
            ("keep", {"outliers": "none"}),
            ("winsorize 1%", {"outliers": "winsor_1"}),
            ("winsorize 0.5%", {"outliers": "winsor_05"}),
            ("clip IQR fences", {"outliers": "iqr"}),
            ("drop (Isolation Forest)", {"outliers": "iforest"}),
        ], refs),
        ("Scaling", [
            ("standard", {"scaler": "standard"}),
            ("min-max [0, 1]", {"scaler": "minmax"}),
            ("robust", {"scaler": "robust"}),
            ("quantile-normal", {"scaler": "quantile"}),
        ], [{"model": "svm"}, {"model": "knn"}]),
        ("Features", [
            ("base readings", {}),
            ("+ domain flags", {"domain": True}),
            ("+ |z| deviations", {"deviation": True}),
            ("+ 3 interactions", {"interactions": 3}),
            ("+ 5 interactions", {"interactions": 5}),
            ("+ 8 interactions", {"interactions": 8}),
            ("+ 12 interactions", {"interactions": 12}),
            ("+ 5 interactions + |z|", {"interactions": 5, "deviation": True}),
        ], [{"model": "svm"}, {"model": "rf"}, {"model": "logreg"}], ["svm", "rf"]),
        ("Balancing", [
            ("none", {"balance": "none"}),
            ("class weights", {"balance": "class_weight"}),
            ("SMOTE", {"balance": "smote"}),
            ("SMOTE-Tomek", {"balance": "smote_tomek"}),
        ], refs),
    ]
    stages = [(s + (None,))[:4] for s in stages]

    tables = []
    for name, options, references, decide_on in stages:
        label, overrides, table, decision, reason = run_stage(
            name, options, references, base, X, y, log, decide_on)
        base = {**base, **overrides}
        tables.append(table)
        decisions.append({"stage": name, "chosen": label, "reason": reason,
                          "ranking": decision.to_dict("records")})

    pd.concat(tables).to_csv(config.TABLE_DIR / "stages_all.csv", index=False)
    return base, decisions


# ------------------------------------------------------------------ model zoo

def model_zoo(base, X, y, log=print):
    log("\n  STAGE: Model comparison")
    rows = []
    variants = [(name, {}) for name in models.NAMES]
    variants += [(name, {"imputer": "none"}) for name in ("hgb", "xgb", "lgbm", "catboost")]
    for name, extra in variants:
        r = {**base, "model": name, **extra}
        if r["imputer"] == "none" and r["balance"] in ("smote", "smote_tomek"):
            r["balance"] = "class_weight"
        summary, _ = cv_recipe(r, X, y)
        label = models.NAMES[name] + (" (native NaN)" if extra else "")
        rows.append({"model_key": name, "model": label, "recipe": json.dumps(r), **summary})
        log(f"    {label:<42} auc={summary['roc_auc']:.4f}±{summary['roc_auc_sd']:.3f}"
            f" acc={summary['accuracy']:.4f} prec={summary['precision']:.3f}"
            f" ({summary['seconds']}s)")
    table = pd.DataFrame(rows).sort_values("roc_auc", ascending=False).reset_index(drop=True)
    table.to_csv(config.TABLE_DIR / "model_comparison.csv", index=False)
    return table


# --------------------------------------------------------------------- tuning

def search_space(trial, name):
    f, i, c = trial.suggest_float, trial.suggest_int, trial.suggest_categorical
    if name == "svm":
        return {"C": f("C", 0.1, 30, log=True), "gamma": f("gamma", 0.005, 1.0, log=True)}
    if name == "logreg":
        return {"C": f("C", 1e-3, 10, log=True)}
    if name == "knn":
        return {"n_neighbors": i("n_neighbors", 5, 80), "weights": c("weights", ["uniform", "distance"]),
                "p": c("p", [1, 2])}
    if name == "qda":
        return {"reg_param": f("reg_param", 0.0, 0.9)}
    if name == "mlp":
        width = i("width", 16, 128, log=True)
        return {"hidden_layer_sizes": (width, max(4, width // 2)), "alpha": f("alpha", 1e-5, 1.0, log=True),
                "learning_rate_init": f("lr", 1e-4, 1e-2, log=True)}
    if name == "rf":
        return {"n_estimators": i("n_estimators", 200, 800, step=100),
                "max_features": c("max_features", ["sqrt", "log2", 0.5, 0.8]),
                "min_samples_leaf": i("min_samples_leaf", 1, 10),
                "max_depth": c("max_depth", [None, 8, 12, 20])}
    if name == "et":
        return {"n_estimators": i("n_estimators", 300, 1200, step=100),
                "max_features": c("max_features", ["sqrt", "log2", 0.5, 0.8, 1.0]),
                "min_samples_leaf": i("min_samples_leaf", 1, 10),
                "max_depth": c("max_depth", [None, 10, 16, 24])}
    if name == "hgb":
        return {"learning_rate": f("learning_rate", 0.01, 0.2, log=True),
                "max_iter": i("max_iter", 100, 800, step=50),
                "max_leaf_nodes": i("max_leaf_nodes", 4, 48),
                "min_samples_leaf": i("min_samples_leaf", 5, 80),
                "l2_regularization": f("l2_regularization", 1e-4, 10, log=True)}
    if name == "xgb":
        return {"n_estimators": i("n_estimators", 150, 1200, step=50),
                "learning_rate": f("learning_rate", 0.005, 0.2, log=True),
                "max_depth": i("max_depth", 2, 9),
                "min_child_weight": f("min_child_weight", 0.5, 20, log=True),
                "subsample": f("subsample", 0.5, 1.0),
                "colsample_bytree": f("colsample_bytree", 0.4, 1.0),
                "gamma": f("gamma", 1e-3, 5, log=True),
                "reg_lambda": f("reg_lambda", 1e-3, 20, log=True)}
    if name == "lgbm":
        return {"n_estimators": i("n_estimators", 150, 1200, step=50),
                "learning_rate": f("learning_rate", 0.005, 0.2, log=True),
                "num_leaves": i("num_leaves", 4, 64),
                "min_child_samples": i("min_child_samples", 5, 100),
                "subsample": f("subsample", 0.5, 1.0),
                "colsample_bytree": f("colsample_bytree", 0.4, 1.0),
                "reg_lambda": f("reg_lambda", 1e-3, 20, log=True)}
    if name == "catboost":
        return {"iterations": i("iterations", 200, 1200, step=100),
                "learning_rate": f("learning_rate", 0.01, 0.2, log=True),
                "depth": i("depth", 3, 8),
                "l2_leaf_reg": f("l2_leaf_reg", 0.5, 30, log=True)}
    if name == "gnb":
        return {"var_smoothing": f("var_smoothing", 1e-10, 1e-2, log=True)}
    raise ValueError(name)


def tune(name, base, X, y, n_trials, timeout, log=print):
    """Optuna TPE search over the model's parameters and its feature settings."""

    def objective(trial):
        imputers = ["median", "knn", "iterative", "hybrid"]
        if name in models.NATIVE_NAN:
            imputers.append("none")
        r = {**base, "model": name,
             "imputer": trial.suggest_categorical("imputer", imputers),
             "indicator": trial.suggest_categorical("indicator", [False, True]),
             "interactions": trial.suggest_categorical("interactions", [0, 3, 5, 8, 12]),
             "deviation": trial.suggest_categorical("deviation", [False, True]),
             "params": search_space(trial, name)}
        if name not in models.SCALE_FREE:
            r["scaler"] = trial.suggest_categorical("scaler", ["robust", "standard", "quantile"])
        summary, _ = cv_recipe(r, X, y, repeats=config.TUNE_REPEATS)
        trial.set_user_attr("recipe", r)
        return summary["roc_auc"]

    study = optuna.create_study(direction="maximize", study_name=name,
                                sampler=optuna.samplers.TPESampler(seed=config.RANDOM_STATE))
    start = {"imputer": base["imputer"], "indicator": base["indicator"],
             "interactions": base["interactions"], "deviation": base["deviation"]}
    if name not in models.SCALE_FREE:
        start["scaler"] = base["scaler"]
    study.enqueue_trial(start, skip_if_exists=True)
    started = time.time()
    study.optimize(objective, n_trials=n_trials, timeout=timeout)
    best = study.best_trial.user_attrs["recipe"]
    log(f"    {models.NAMES[name]:<34} {len(study.trials):>3} trials"
        f"  best tuning auc={study.best_value:.4f}  ({time.time() - started:.0f}s)")
    history = study.trials_dataframe(attrs=("number", "value", "params"))
    history.insert(0, "model", name)
    return best, study.best_value, history


# ------------------------------------------------------------------ ensembles

def probabilistic(recipe):
    """A pipeline that has predict_proba. Margin models are Platt-scaled in-fold."""
    pipe = R.build(recipe)
    if hasattr(pipe.steps[-1][1], "predict_proba"):
        return pipe
    return CalibratedClassifierCV(pipe, method="sigmoid", cv=3)


def voting(recipes):
    return VotingClassifier([(f"m{i}_{r['model']}", probabilistic(r))
                             for i, r in enumerate(recipes)], voting="soft", n_jobs=1)


def stacking(recipes):
    return StackingClassifier(
        [(f"m{i}_{r['model']}", R.build(r)) for i, r in enumerate(recipes)],
        final_estimator=LogisticRegression(C=1.0, max_iter=2000),
        cv=StratifiedKFold(5, shuffle=True, random_state=config.RANDOM_STATE), n_jobs=1)


def build_candidate(spec):
    """spec = {'kind': 'single'|'voting'|'stacking', 'recipes': [...]} -> estimator."""
    if spec["kind"] == "single":
        return R.build(spec["recipes"][0])
    if spec["kind"] == "voting":
        return voting(spec["recipes"])
    return stacking(spec["recipes"])


def final_estimator(spec):
    """The deployable estimator for a candidate, and how its probabilities are calibrated.

    A stacking ensemble's logistic-regression meta-learner is fitted on out-of-fold
    base-model scores, which is Platt scaling already, so it is used as is. Single
    models and voting ensembles get sigmoid (Platt) calibration on out-of-fold
    scores. Isotonic calibration is not used: as a step function it can output
    exactly 0 or 1, which overstates certainty for a model this weak.
    ensemble=False keeps one fitted model in the artifact instead of five.
    """
    if spec["kind"] == "stacking":
        return build_candidate(spec), "logistic meta-learner (stacked generalisation)"
    return CalibratedClassifierCV(build_candidate(spec), method="sigmoid", ensemble=False,
                                  cv=StratifiedKFold(5, shuffle=True,
                                                     random_state=config.RANDOM_STATE)), \
        "sigmoid (Platt) scaling"
