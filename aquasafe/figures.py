"""Figures for model selection and evaluation. EDA figures live in eda.py."""

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import precision_recall_curve, roc_curve

from . import config, schema
from . import plots as P
from .plots import plt

LABEL = {f.name: f.label for f in schema.FEATURES}


def stages(stage_tables, decisions):
    names = [d["stage"] for d in decisions]
    fig, axes = plt.subplots(len(names), 1, figsize=(8.6, 2.1 * len(names) + 0.8))
    ref_colours = [P.BLUE, P.ORANGE, P.AQUA]
    for ax, name, d in zip(axes, names, decisions):
        t = stage_tables[stage_tables["stage"] == name]
        options = list(dict.fromkeys(t["option"]))
        refs = list(dict.fromkeys(t["reference"]))
        for k, ref in enumerate(refs):
            sub = t[t["reference"] == ref].set_index("option").loc[options]
            ypos = np.arange(len(options)) + (k - (len(refs) - 1) / 2) * 0.22
            ax.errorbar(sub["roc_auc"], ypos, xerr=sub["roc_auc_sd"] / np.sqrt(15),
                        fmt="o", ms=5, color=ref_colours[k], elinewidth=1, capsize=0,
                        label=ref.upper() if ref != "logreg" else "LogReg")
        chosen = options.index(d["chosen"])
        ax.axhspan(chosen - 0.45, chosen + 0.45, color=P.GRID, alpha=0.6, zorder=0)
        ax.set_yticks(range(len(options)), options)
        ax.invert_yaxis()
        ax.set_title(f"{name}: chose '{d['chosen']}'", fontsize=10)
        ax.grid(axis="y", visible=False)
        ax.legend(loc="lower right", ncol=len(refs), fontsize=7.5)
    axes[-1].set_xlabel("Cross-validated ROC AUC (mean ± standard error)")
    fig.suptitle("Preprocessing decisions, one stage at a time (shaded = chosen)",
                 x=0.01, ha="left", fontweight="bold", fontsize=12)
    fig.tight_layout()
    return P.save(fig, "08_preprocessing_stages.png")


def model_comparison(zoo, tuned, baseline):
    zoo = zoo.sort_values("roc_auc")
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 0.36 * len(zoo) + 1.6), sharey=True)
    y = np.arange(len(zoo))
    for ax, metric, ref, title in (
            (axes[0], "roc_auc", 0.5, "ROC AUC"),
            (axes[1], "accuracy", baseline, "Accuracy at the default 0.5 cut")):
        ax.hlines(y, ref, zoo[metric], color=P.AXIS, linewidth=1)
        ax.scatter(zoo[metric], y, color=P.BLUE, s=34, zorder=3, label="default settings")
        tuned_keys = {r["model_key"]: r for r in tuned}
        for i, key in enumerate(zoo["model_key"]):
            if key in tuned_keys and "(native" not in zoo["model"].iloc[i]:
                ax.scatter(tuned_keys[key][metric], i, color=P.ORANGE, s=34, zorder=4,
                           marker="D", label="after tuning" if i == 0 else None)
        ax.axvline(ref, color=P.GREY, linewidth=1)
        ax.set_title(title)
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(y, zoo["model"])
    axes[1].text(baseline, len(zoo) - 0.4, " majority baseline", fontsize=7.5, color=P.INK_2)
    handles = [plt.Line2D([], [], marker="o", color=P.BLUE, linestyle="", label="Default settings"),
               plt.Line2D([], [], marker="D", color=P.ORANGE, linestyle="", label="Tuned (Optuna)")]
    axes[1].legend(handles=handles, loc="lower right")
    fig.suptitle("Every classifier under the chosen preprocessing (5-fold x 3 repeats)",
                 x=0.01, ha="left", fontweight="bold", fontsize=12)
    fig.tight_layout()
    return P.save(fig, "09_model_comparison.png")


def tuning_history(history):
    fig, ax = plt.subplots(figsize=(8.6, 3.8))
    colours = [P.BLUE, P.ORANGE, P.AQUA, "#eda100", "#e87ba4"]
    for colour, (name, h) in zip(colours, history.groupby("model", sort=False)):
        h = h.sort_values("number")
        best = h["value"].cummax()
        ax.plot(h["number"], best, color=colour, label=name.upper())
        ax.scatter(h["number"], h["value"], color=colour, s=6, alpha=0.35, linewidths=0)
    ax.set_xlabel("Optuna trial")
    ax.set_ylabel("CV ROC AUC")
    ax.set_ylim(max(0.6, history["value"].quantile(0.05) - 0.01), None)
    ax.legend(ncol=5, loc="lower right")
    ax.set_title("Hyper-parameter search: best score so far (line) and every trial (dots)")
    return P.save(fig, "10_tuning_history.png")


def thresholds(table, chosen):
    fig, ax = plt.subplots(figsize=(8.6, 3.9))
    ax.plot(table["threshold"], table["accuracy"], color=P.BLUE, label="Accuracy")
    ax.plot(table["threshold"], table["precision_above"], color=P.ORANGE,
            label="Precision of 'potable' calls (p ≥ cut)")
    ax.plot(table["threshold"], table["npv_below"], color=P.AQUA,
            label="Precision of 'not potable' calls (p ≤ cut)")
    for key, text, align in (("not_potable", "not-potable cut", "right"),
                             ("decision", "decision cut", "left"),
                             ("potable", "potable cut", "left")):
        ax.axvline(chosen[key], color=P.GREY, linewidth=1)
        pad = " " if align == "left" else ""
        ax.text(chosen[key], 1.01, f"{pad}{text}{' ' if not pad else ''}\n{pad}{chosen[key]:.3f} ",
                fontsize=7.5, color=P.INK_2, va="bottom", ha=align,
                transform=ax.get_xaxis_transform())
    ax.set_xlim(0.1, 0.9)
    ax.set_ylim(0.3, 1.0)
    ax.set_xlabel("Probability cut-off")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3, fontsize=8)
    ax.set_title("Operating points chosen on out-of-fold training predictions", pad=34)
    return P.save(fig, "11_threshold_selection.png")


def test_curves(y, p, m, threshold):
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4))
    cm = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])
    from matplotlib.colors import LinearSegmentedColormap
    axes[0].imshow(cm, cmap=LinearSegmentedColormap.from_list("seq", P.SEQUENTIAL[:5]))
    for i in range(2):
        for j in range(2):
            axes[0].text(j, i, f"{cm[i, j]}\n{cm[i, j] / cm.sum():.1%}", ha="center",
                         va="center", color=P.INK, fontsize=10)
    axes[0].set_xticks([0, 1], ["Not potable", "Potable"])
    axes[0].set_yticks([0, 1], ["Not potable", "Potable"])
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("Actual")
    axes[0].grid(False)
    axes[0].set_title(f"Confusion matrix (cut {threshold:.3f})")

    fpr, tpr, _ = roc_curve(y, p)
    axes[1].plot(fpr, tpr, color=P.BLUE)
    axes[1].plot([0, 1], [0, 1], color=P.GREY, linewidth=1)
    axes[1].set_xlabel("False positive rate")
    axes[1].set_ylabel("True positive rate")
    axes[1].set_title(f"ROC curve (AUC {m['roc_auc']:.3f})")

    prec, rec, _ = precision_recall_curve(y, p)
    axes[2].plot(rec, prec, color=P.ORANGE)
    axes[2].axhline(np.mean(y), color=P.GREY, linewidth=1)
    axes[2].text(0.99, np.mean(y) + 0.01, "chance", ha="right", fontsize=8, color=P.INK_2)
    axes[2].set_xlabel("Recall")
    axes[2].set_ylabel("Precision")
    axes[2].set_ylim(0, 1.02)
    axes[2].set_title(f"Precision-recall (AP {m['pr_auc']:.3f})")
    fig.suptitle("Held-out test set (656 samples, scored once)", x=0.01, ha="left",
                 fontweight="bold", fontsize=12)
    fig.tight_layout()
    return P.save(fig, "12_test_results.png")


def calibration(y, p, chosen):
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.9))
    frac, mean = calibration_curve(y, p, n_bins=10, strategy="quantile")
    axes[0].plot([0, 1], [0, 1], color=P.GREY, linewidth=1)
    axes[0].plot(mean, frac, marker="o", color=P.BLUE, ms=5)
    axes[0].set_xlabel("Predicted probability of 'potable'")
    axes[0].set_ylabel("Observed share potable")
    axes[0].set_title("Reliability: predicted vs observed")

    bins = np.linspace(0, 1, 41)
    for cls in (0, 1):
        axes[1].hist(p[np.asarray(y) == cls], bins=bins, color=P.CLASS_COLOUR[cls], alpha=0.65,
                     label=config.LABELS[cls])
    for key in ("not_potable", "decision", "potable"):
        axes[1].axvline(chosen[key], color=P.GREY, linewidth=1)
    axes[1].set_xlabel("Predicted probability of 'potable'")
    axes[1].set_ylabel("Test samples")
    axes[1].legend()
    axes[1].set_title("Scores by true class, with the three cut-offs")
    fig.tight_layout()
    return P.save(fig, "13_calibration.png")


def importance(table):
    t = table.sort_values("importance")
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    ax.barh([LABEL[f] for f in t["feature"]], t["importance"], xerr=t["std"],
            color=P.BLUE, height=0.6, error_kw={"elinewidth": 1, "ecolor": P.GREY})
    ax.axvline(0, color=P.AXIS, linewidth=1)
    ax.set_xlabel("Drop in test ROC AUC when the feature is shuffled")
    ax.grid(axis="y", visible=False)
    ax.set_title("Permutation importance on the test set")
    return P.save(fig, "14_importance.png")


def robustness(noise, missing):
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 3.9))
    axes[0].plot(noise["noise_sd"] * 100, noise["roc_auc"], marker="o", color=P.BLUE,
                 label="ROC AUC", ms=5)
    axes[0].plot(noise["noise_sd"] * 100, noise["accuracy"], marker="o", color=P.ORANGE,
                 label="Accuracy", ms=5)
    axes[0].set_xlabel("Sensor noise (% of each reading, 1 s.d.)")
    axes[0].set_title("Degradation under measurement noise")
    axes[0].legend()

    m = missing.copy()
    full = m.iloc[0]["roc_auc"]
    m = m.iloc[1:].sort_values("roc_auc")
    names = [LABEL.get(r, "pH + Sulfate") for r in m["removed"]]
    axes[1].barh(names, full - m["roc_auc"], color=P.BLUE, height=0.6)
    axes[1].set_xlabel("Loss in test ROC AUC when the reading is unavailable")
    axes[1].set_title("A sensor goes missing at prediction time")
    axes[1].grid(axis="y", visible=False)
    axes[1].invert_yaxis()
    fig.tight_layout()
    return P.save(fig, "15_robustness.png")


def learning_curve(sizes, train, valid):
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    for scores, colour, label in ((train, P.ORANGE, "Training folds"),
                                  (valid, P.BLUE, "Validation folds")):
        mean, sd = scores.mean(axis=1), scores.std(axis=1)
        ax.plot(sizes, mean, marker="o", color=colour, label=label, ms=5)
        ax.fill_between(sizes, mean - sd, mean + sd, color=colour, alpha=0.15, linewidth=0)
    ax.set_xlabel("Training samples")
    ax.set_ylabel("ROC AUC")
    ax.legend()
    ax.set_title("Learning curve: validation score still rising, so more data would help")
    return P.save(fig, "16_learning_curve.png")


def leakage(table):
    t = table.iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 3.6))
    colours = [P.ORANGE if leaky else P.BLUE for leaky in t["leaky"]]
    ax.barh(t["protocol"], t["accuracy"] * 100, color=colours, height=0.6)
    for i, v in enumerate(t["accuracy"]):
        ax.text(v * 100 + 0.5, i, f"{v:.1%}", va="center", fontsize=8.5, color=P.INK_2)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Reported accuracy (%)")
    ax.grid(axis="y", visible=False)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=P.BLUE, label="Valid protocol"),
                       Patch(color=P.ORANGE, label="Leaky protocol")], loc="lower right")
    ax.set_title("Where 80-95% accuracy claims on this dataset come from")
    return P.save(fig, "17_leakage.png")


def selective(curve, three):
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    ax.plot(curve["coverage"] * 100, curve["accuracy"] * 100, marker="o", color=P.BLUE, ms=4)
    ax.scatter([three["coverage"] * 100], [three["accuracy_on_decided"] * 100], s=70,
               color=P.ORANGE, zorder=4, edgecolors=P.SURFACE, linewidths=1.5)
    ax.annotate(f"deployed three-way rule\n{three['coverage']:.0%} decided, "
                f"{three['accuracy_on_decided']:.1%} correct",
                (three["coverage"] * 100, three["accuracy_on_decided"] * 100),
                xytext=(12, 12), textcoords="offset points", fontsize=8, color=P.INK_2)
    ax.set_xlabel("Share of test samples the model answers (most confident first, %)")
    ax.set_ylabel("Accuracy on answered samples (%)")
    ax.set_title("Answering only confident cases buys accuracy")
    return P.save(fig, "18_selective_accuracy.png")


def save_tables(**tables):
    for name, t in tables.items():
        (t if isinstance(t, pd.DataFrame) else pd.DataFrame(t)).to_csv(
            config.TABLE_DIR / f"{name}.csv", index=False)
