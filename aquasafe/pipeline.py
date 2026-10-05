"""Train AquaSafe-ML end to end and write the model artifact, tables and figures.

    python -m aquasafe.pipeline              full run (about an hour on 8 cores)
    python -m aquasafe.pipeline --quick      small tuning budget, for a smoke test
    python -m aquasafe.pipeline --reuse      reuse saved selection and tuning results

The test split is touched once, after the final model and its thresholds are
fixed. Every choice before that is made by cross-validation on the training split.
"""

import argparse
import json
import platform
import sys
import time
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import StratifiedKFold, cross_val_predict, learning_curve

from . import __version__, config, data, eda, evaluation as V, experiments as E, figures
from . import guard, models, recipe as R

SELECTION_CACHE = config.ARTIFACT_DIR / "selection.json"
TUNING_CACHE = config.ARTIFACT_DIR / "tuning.json"
ENSEMBLE_CACHE = config.ARTIFACT_DIR / "ensembles.json"


class Log:
    def __init__(self, path):
        self.handle = open(path, "w")

    def __call__(self, text=""):
        print(text, flush=True)
        self.handle.write(text + "\n")
        self.handle.flush()


def banner(log, text):
    log("\n" + "=" * 78 + f"\n{text}\n" + "=" * 78)


def jsonable(obj):
    if isinstance(obj, dict):
        return {k: jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        return None if np.isnan(obj) else round(float(obj), 5)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def describe_spec(spec):
    if spec["kind"] == "single":
        return R.describe(spec["recipes"][0])
    names = " + ".join(models.NAMES[r["model"]] for r in spec["recipes"])
    return f"{spec['kind'].capitalize()} ensemble of {names}"


def write_examples(X_test, y_test, p_test, thresholds):
    """Pick real held-out samples for the web UI's 'load example' menu."""
    t = pd.DataFrame({"y": y_test.values, "p": p_test}, index=X_test.index)
    complete = X_test.notna().all(axis=1)
    picks = [
        ("Held-out sample, lab result: potable", t[complete & (t.y == 1)]["p"].idxmax()),
        ("Held-out sample, lab result: not potable", t[complete & (t.y == 0)]["p"].idxmin()),
        ("Held-out sample, borderline", (t[complete]["p"] - thresholds["decision"]).abs().idxmin()),
        ("Held-out sample with a missing pH reading", t[X_test["ph"].isna()].index[0]),
    ]
    out = [{"name": name, "true_label": config.LABELS[int(y_test[i])],
            "values": {k: (None if pd.isna(v) else round(float(v), 4))
                       for k, v in X_test.loc[i].items()}} for name, i in picks]
    (config.ARTIFACT_DIR / "examples.json").write_text(json.dumps(out, indent=1))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--reuse", action="store_true")
    parser.add_argument("--trials", type=int, default=80)
    parser.add_argument("--timeout", type=int, default=900, help="seconds per tuned model")
    parser.add_argument("--tune-top", type=int, default=5)
    args = parser.parse_args(argv)
    if args.quick:
        args.trials, args.timeout, args.tune_top = 10, 90, 3

    config.ensure_dirs()
    log = Log(config.REPORT_DIR / "run_log.txt")
    started = time.time()

    # ----------------------------------------------------------------- data
    banner(log, "1. PROBLEM AND DATA")
    frame = data.load()
    (X, y), (X_test, y_test) = data.split(frame)
    baseline = float(1 - y_test.mean())
    log(f"  Binary classification: 9 continuous readings -> Potability (1 = potable)")
    log(f"  {len(frame)} rows, potable share {frame[config.TARGET].mean():.3f}")
    log(f"  Stratified split: train {len(X)} rows, test {len(X_test)} rows "
        f"(potable {y.mean():.3f} / {y_test.mean():.3f})")
    log(f"  Majority-class baseline accuracy on test: {baseline:.4f}")

    banner(log, "2-3. EXPLORATORY ANALYSIS, MISSING VALUES AND OUTLIERS (training split)")
    eda_tables, eda_summary = eda.run(X, y, log)

    # ----------------------------------------------------- staged selection
    banner(log, "4-7. PREPROCESSING SELECTION (repeated stratified 5-fold CV)")
    if args.reuse and SELECTION_CACHE.exists():
        cached = json.loads(SELECTION_CACHE.read_text())
        base, decisions = cached["base"], cached["decisions"]
        zoo = pd.read_csv(config.TABLE_DIR / "model_comparison.csv")
        log(f"  reusing {SELECTION_CACHE.name}: {R.describe(base)}")
    else:
        base, decisions = E.staged_selection(X, y, log)
        banner(log, "8. MODEL COMPARISON")
        zoo = E.model_zoo(base, X, y, log)
        SELECTION_CACHE.write_text(json.dumps({"base": base, "decisions": decisions},
                                              indent=1, default=float))
    base = {k: v for k, v in R.complete(base).items() if k not in ("model", "params")}
    log(f"\n  Shared preprocessing: {R.describe({**base, 'model': 'svm'})}")

    # --------------------------------------------------------------- tuning
    banner(log, "9. HYPER-PARAMETER TUNING (Optuna TPE)")
    top = list(dict.fromkeys(zoo.sort_values("roc_auc", ascending=False)["model_key"]))
    top = top[:args.tune_top]
    if args.reuse and TUNING_CACHE.exists():
        tuned = json.loads(TUNING_CACHE.read_text())
        history = pd.read_csv(config.TABLE_DIR / "tuning_history.csv")
        log("  reusing " + TUNING_CACHE.name)
    else:
        log(f"  tuning: {', '.join(top)}  ({args.trials} trials / {args.timeout}s each)")
        tuned, histories = [], []
        for name in top:
            best, tuning_auc, hist = E.tune(name, base, X, y, args.trials, args.timeout, log)
            summary, folds = E.cv_recipe(best, X, y)
            # Same folds as the zoo, so the default and tuned scores are paired.
            default = json.loads(zoo[zoo["model_key"] == name]
                                 .sort_values("roc_auc").iloc[-1]["recipe"])
            d_summary, d_folds = E.cv_recipe(default, X, y)
            if d_summary["roc_auc"] > summary["roc_auc"]:
                log(f"      tuned auc={summary['roc_auc']:.4f} does not beat the default "
                    f"{d_summary['roc_auc']:.4f}; keeping the default configuration")
                best, summary, folds = default, d_summary, d_folds
            tuned.append({"model_key": name, "model": models.NAMES[name], "recipe": best,
                          "tuning_auc": tuning_auc, "folds": folds.tolist(), **summary})
            histories.append(hist)
            log(f"      re-scored on 5x3 CV: auc={summary['roc_auc']:.4f} "
                f"acc={summary['accuracy']:.4f}")
        history = pd.concat(histories)
        history.to_csv(config.TABLE_DIR / "tuning_history.csv", index=False)
        TUNING_CACHE.write_text(json.dumps(jsonable(tuned), indent=1))
    tuned = sorted(tuned, key=lambda t: t["roc_auc"], reverse=True)
    pd.DataFrame([{k: v for k, v in t.items() if k not in ("folds",)} | {
        "recipe": R.describe(t["recipe"])} for t in tuned]).to_csv(
        config.TABLE_DIR / "tuned_models.csv", index=False)

    # ------------------------------------------------------------ ensembles
    banner(log, "10. ENSEMBLES OF THE TUNED MODELS")
    candidates = [{"name": t["model"] + " (tuned)", "spec": {"kind": "single",
                   "recipes": [t["recipe"]]}, "roc_auc": t["roc_auc"],
                   "accuracy": t["accuracy"], "folds": np.array(t["folds"]),
                   "summary": t} for t in tuned]
    members = [t["recipe"] for t in tuned[:3]]
    cached = {}
    if args.reuse and ENSEMBLE_CACHE.exists():
        cached = json.loads(ENSEMBLE_CACHE.read_text())
    for kind in ("voting", "stacking"):
        spec = {"kind": kind, "recipes": members}
        if kind in cached and cached[kind]["spec"] == jsonable(spec):
            summary, folds = cached[kind]["summary"], np.array(cached[kind]["folds"])
            log(f"  reusing cached {kind} scores")
        else:
            summary, folds = E.cv_estimator(E.build_candidate(spec), X, y)
            cached[kind] = {"spec": jsonable(spec), "summary": summary, "folds": folds.tolist()}
            ENSEMBLE_CACHE.write_text(json.dumps(jsonable(cached), indent=1))
        candidates.append({"name": describe_spec(spec), "spec": spec,
                           "roc_auc": summary["roc_auc"], "accuracy": summary["accuracy"],
                           "folds": folds, "summary": summary})
        log(f"  {kind:<9} auc={summary['roc_auc']:.4f}±{summary['roc_auc_sd']:.3f} "
            f"acc={summary['accuracy']:.4f} ({summary['seconds']}s)")

    ranking = pd.DataFrame([{"candidate": c["name"], **{k: v for k, v in c["summary"].items()
                             if k not in ("recipe", "folds", "model", "model_key")}}
                            for c in candidates]).sort_values("roc_auc", ascending=False)
    ranking.to_csv(config.TABLE_DIR / "final_candidates.csv", index=False)

    best = max(candidates, key=lambda c: c["roc_auc"])
    best_single = max((c for c in candidates if c["spec"]["kind"] == "single"),
                      key=lambda c: c["roc_auc"])
    gain = best["folds"] - best_single["folds"]
    se = gain.std(ddof=1) / np.sqrt(len(gain))
    if best is not best_single and gain.mean() <= se:
        log(f"  {best['name']} leads by {gain.mean():+.4f} AUC (SE {se:.4f}): within noise,"
            " so the simpler single model is kept")
        final = best_single
    else:
        final = best
    log(f"\n  FINAL: {final['name']}  (CV auc={final['roc_auc']:.4f}, acc={final['accuracy']:.4f})")

    # ------------------------------------------------ calibration, thresholds
    banner(log, "11. CALIBRATION AND OPERATING THRESHOLDS (out-of-fold, training split)")
    oof_cv = StratifiedKFold(5, shuffle=True, random_state=config.RANDOM_STATE + 1)
    estimator, method = E.final_estimator(final["spec"])
    oof_p = cross_val_predict(estimator, X, y, cv=oof_cv, method="predict_proba",
                              n_jobs=-1)[:, 1]
    m = V.metrics(y, oof_p, 0.5)
    log(f"  calibration: {method}")
    log(f"  OOF brier={m['brier']:.4f} log_loss={m['log_loss']:.4f} auc={m['roc_auc']:.4f}")
    thresholds, threshold_table = V.choose_thresholds(y, oof_p)
    oof_metrics = V.metrics(y, oof_p, thresholds["decision"])
    oof_three = V.three_way(y, oof_p, thresholds)
    log(f"  thresholds : {thresholds}")
    log(f"  OOF at decision cut: acc={oof_metrics['accuracy']:.4f} "
        f"prec={oof_metrics['precision']:.4f} rec={oof_metrics['recall']:.4f}")
    log(f"  OOF three-way: {jsonable(oof_three)}")

    model = estimator.fit(X, y)

    # ------------------------------------------------------------ test set
    banner(log, "12. HELD-OUT TEST SET (scored once)")
    p_test = model.predict_proba(X_test)[:, 1]
    test = V.metrics(y_test, p_test, thresholds["decision"])
    test_default = V.metrics(y_test, p_test, 0.5)
    ci = V.bootstrap(y_test, p_test, thresholds["decision"])
    three = V.three_way(y_test, p_test, thresholds)
    for k in ("accuracy", "precision", "recall", "specificity", "f1", "balanced_accuracy",
              "mcc", "roc_auc", "pr_auc", "brier"):
        log(f"  {k:<18} {test[k]:.4f}   95% CI {ci[k][0]:.4f} - {ci[k][1]:.4f}")
    log(f"  baseline accuracy  {baseline:.4f}  (gain {test['accuracy'] - baseline:+.4f})")
    log(f"  at cut 0.5         acc={test_default['accuracy']:.4f} "
        f"prec={test_default['precision']:.4f} rec={test_default['recall']:.4f}")
    log(f"  three-way verdicts {jsonable(three)}")
    write_examples(X_test, y_test, p_test, thresholds)

    # -------------------------------------------------------- robustness
    banner(log, "13. ROBUSTNESS, IMPORTANCE AND LEARNING CURVE")
    noise = V.noise_robustness(model, X_test, y_test, threshold=thresholds["decision"])
    missing = V.missing_robustness(model, X_test, y_test, threshold=thresholds["decision"])
    imp = V.importance(model, X_test, y_test)
    log("  noise:\n" + noise.round(4).to_string(index=False))
    log("  missing reading at prediction time:\n" + missing.round(4).to_string(index=False))
    log("  permutation importance:\n" + imp.round(4).to_string(index=False))

    # The learning curve is a data-sufficiency diagnostic, so it uses the best single
    # tuned model: about 18x fewer fits than the stack, with the same conclusion.
    lc_recipe = tuned[0]["recipe"]
    log(f"  learning curve on: {R.describe(lc_recipe)}")
    sizes, lc_train, lc_valid = learning_curve(
        R.build(lc_recipe), X, y, cv=StratifiedKFold(
            5, shuffle=True, random_state=config.RANDOM_STATE),
        train_sizes=np.linspace(0.15, 1.0, 7), scoring="roc_auc", n_jobs=-1)

    banner(log, "14. LEAKAGE DEMONSTRATION")
    leak = V.leakage_demo(frame, final["accuracy"], test["accuracy"])
    log(leak.round(4).to_string(index=False))

    # ---------------------------------------------------------- artifact
    banner(log, "15. ARTIFACT")
    profile = guard.TrainingProfile().fit(X.values)
    card = {
        "name": "AquaSafe-ML potability classifier",
        "version": __version__,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "description": describe_spec(final["spec"]) + f", calibrated by {method}",
        "spec": final["spec"],
        "calibration": method,
        "thresholds": thresholds,
        "data": {"source": "https://www.kaggle.com/datasets/adityakadiwal/water-potability",
                 "rows": len(frame), "train_rows": len(X), "test_rows": len(X_test),
                 "positive_rate": float(frame[config.TARGET].mean())},
        "metrics": {"cv": {k: final["summary"][k] for k in E.SCORING},
                    "oof_at_decision": oof_metrics, "oof_three_way": oof_three,
                    "test_at_decision": test, "test_at_0_5": test_default,
                    "test_95ci": ci, "test_three_way": three,
                    "majority_baseline": baseline},
        "preprocessing_decisions": [{k: d[k] for k in ("stage", "chosen", "reason")}
                                    for d in decisions],
        "intended_use": "Screening aid that ranks samples for laboratory testing. "
                        "Not a replacement for certified potability testing.",
        "limitations": [
            "Trained on one public dataset of 3,276 samples whose labels do not follow "
            "WHO or IS 10500 limits; it reproduces those labels, not a regulation.",
            "Physicochemical readings only: no microbiological contamination.",
            "Accuracy is modest (about 0.68); the three-way verdict withholds a call on "
            "uncertain samples instead of guessing.",
        ],
        "environment": {"python": platform.python_version(), "sklearn": sklearn.__version__},
    }
    bundle = {"model": model, "profile": profile, "thresholds": thresholds,
              "version": __version__, "trained_at": card["trained_at"],
              "description": card["description"], "metrics": jsonable(card["metrics"]),
              "spec": final["spec"]}
    joblib.dump(bundle, config.MODEL_PATH, compress=3)
    config.MODEL_CARD_PATH.write_text(json.dumps(jsonable(card), indent=2))
    log(f"  wrote {config.MODEL_PATH.relative_to(config.ROOT)} "
        f"({config.MODEL_PATH.stat().st_size / 1e6:.1f} MB)")
    log(f"  wrote {config.MODEL_CARD_PATH.relative_to(config.ROOT)}")

    # ----------------------------------------------------------- figures
    stage_tables = pd.read_csv(config.TABLE_DIR / "stages_all.csv")
    figures.stages(stage_tables, decisions)
    figures.model_comparison(zoo[~zoo["model"].str.contains("native")], tuned, baseline)
    figures.tuning_history(history)
    figures.thresholds(threshold_table, thresholds)
    figures.test_curves(y_test, p_test, test, thresholds["decision"])
    figures.calibration(y_test, p_test, thresholds)
    figures.importance(imp)
    figures.robustness(noise, missing)
    figures.learning_curve(sizes, lc_train, lc_valid)
    figures.leakage(leak)
    curve = V.selective_curve(y_test, p_test)
    figures.selective(curve, three)
    figures.save_tables(
        threshold_table=threshold_table, noise_robustness=noise, missing_robustness=missing,
        permutation_importance=imp, leakage_demo=leak, selective_accuracy=curve,
        learning_curve=pd.DataFrame({"train_size": sizes,
                                     "train_auc": lc_train.mean(axis=1),
                                     "valid_auc": lc_valid.mean(axis=1)}),
        test_predictions=pd.DataFrame({"index": X_test.index, "y": y_test.values,
                                       "p": p_test}))

    banner(log, "DONE")
    log(f"  model          : {card['description']}")
    log(f"  test accuracy  : {test['accuracy']:.4f} (baseline {baseline:.4f})")
    log(f"  test precision : {test['precision']:.4f}")
    log(f"  test ROC AUC   : {test['roc_auc']:.4f}")
    log(f"  elapsed        : {(time.time() - started) / 60:.1f} min")
    return card


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
