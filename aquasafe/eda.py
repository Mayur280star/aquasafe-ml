"""Exploratory data analysis: statistical tests, tables and figures.

Runs on the training split only, so nothing the analysis concludes (and nothing
the later stages decide because of it) has looked at the test set.
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.feature_selection import mutual_info_classif
from statsmodels.stats.outliers_influence import variance_inflation_factor

from . import config, schema
from . import plots as P
from .plots import plt

FEATURES = schema.FEATURE_NAMES
LABEL = {f.name: f.label for f in schema.FEATURES}
SHORT = {"ph": "pH", "Hardness": "Hardness", "Solids": "TDS", "Chloramines": "Chloramines",
         "Sulfate": "Sulfate", "Conductivity": "Conductivity", "Organic_carbon": "Org. carbon",
         "Trihalomethanes": "THM", "Turbidity": "Turbidity"}


# --------------------------------------------------------------------- tables

def overview(X):
    rows = []
    for c in X:
        s = X[c]
        present = s.dropna()
        rows.append({
            "feature": c,
            "dtype": str(s.dtype),
            "unique_values": s.nunique(),
            "variable_type": "continuous" if s.nunique() > 20 else "nominal",
            "missing": int(s.isna().sum()),
            "missing_pct": round(100 * s.isna().mean(), 2),
            "min": present.min(), "median": present.median(), "max": present.max(),
            "mean": present.mean(), "std": present.std(),
            "skew": stats.skew(present), "kurtosis": stats.kurtosis(present),
            "normality_p": stats.normaltest(present).pvalue,
        })
    return pd.DataFrame(rows).round(4)


def missingness(X, y):
    """Is each gap MCAR? Test gaps against the target and against the other readings."""
    rows = []
    for c in X.columns[X.isna().any()]:
        gap = X[c].isna()
        table = pd.crosstab(gap, y)
        chi2_p = stats.chi2_contingency(table).pvalue
        others = []
        for o in X.columns.drop(c):
            a, b = X.loc[gap, o].dropna(), X.loc[~gap, o].dropna()
            others.append(stats.mannwhitneyu(a, b).pvalue)
        min_p = min(others) * len(others)   # Bonferroni
        rows.append({
            "feature": c,
            "missing": int(gap.sum()),
            "missing_pct": round(100 * gap.mean(), 2),
            "missing_rate_not_potable": round(gap[y == 0].mean(), 4),
            "missing_rate_potable": round(gap[y == 1].mean(), 4),
            "chi2_p_vs_target": round(chi2_p, 4),
            "min_bonferroni_p_vs_other_features": round(min(min_p, 1.0), 4),
            "verdict": "consistent with MCAR" if chi2_p > 0.05 and min_p > 0.05
                       else "not MCAR",
        })
    co = X.isna().astype(int)
    both = {f"{a} & {b}": int((co[a] & co[b]).sum())
            for i, a in enumerate(co.columns) for b in co.columns[i + 1:]
            if co[a].any() and co[b].any()}
    return pd.DataFrame(rows), both


def class_tests(X, y):
    """Do the classes differ in location, shape or spread?"""
    filled = X.fillna(X.median())
    mi = mutual_info_classif(filled, y, random_state=config.RANDOM_STATE, n_neighbors=5)
    rows = []
    for c, m in zip(X, mi):
        a, b = X.loc[y == 0, c].dropna(), X.loc[y == 1, c].dropna()
        rows.append({
            "feature": c,
            "mean_not_potable": a.mean(), "mean_potable": b.mean(),
            "sd_not_potable": a.std(), "sd_potable": b.std(),
            "sd_ratio": b.std() / a.std(),
            "welch_t_p": stats.ttest_ind(a, b, equal_var=False).pvalue,
            "ks_p": stats.ks_2samp(a, b).pvalue,
            "levene_p": stats.levene(a, b).pvalue,
            "mutual_information": m,
        })
    return pd.DataFrame(rows).round(5)


def class_correlation_difference(X, y):
    c0, c1 = X[y == 0].corr(), X[y == 1].corr()
    diff = c1 - c0
    rows = [{"pair": f"{a} x {b}", "corr_not_potable": c0.loc[a, b],
             "corr_potable": c1.loc[a, b], "difference": diff.loc[a, b]}
            for i, a in enumerate(FEATURES) for b in FEATURES[i + 1:]]
    table = pd.DataFrame(rows).round(4)
    return diff, table.reindex(table["difference"].abs().sort_values(ascending=False).index)


def vif(X):
    filled = X.fillna(X.median())
    filled = (filled - filled.mean()) / filled.std()
    values = filled.assign(const=1.0).values
    return pd.DataFrame({
        "feature": FEATURES,
        "vif": [variance_inflation_factor(values, i) for i in range(len(FEATURES))],
    }).round(4)


def outliers(X, y):
    rows = []
    for f in schema.FEATURES:
        s = X[f.name]
        q1, q3 = s.quantile([0.25, 0.75])
        iqr = q3 - q1
        is_iqr = (s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)
        z = (s - s.mean()) / s.std()
        physical = (s < f.physical_min) | (s > f.physical_max)
        rows.append({
            "feature": f.name,
            "iqr_outliers": int(is_iqr.sum()),
            "iqr_pct": round(100 * is_iqr.mean(), 2),
            "z3_outliers": int((z.abs() > 3).sum()),
            "physically_impossible": int(physical.sum()),
            "iqr_outlier_potable_rate": round(y[is_iqr].mean(), 4) if is_iqr.any() else np.nan,
        })
    table = pd.DataFrame(rows)

    filled = X.fillna(X.median())
    forest = IsolationForest(contamination="auto", random_state=config.RANDOM_STATE)
    flagged = forest.fit_predict(filled) == -1
    summary = {
        "rows": len(X),
        "rows_with_any_iqr_outlier": int(sum(
            ((X[c] < X[c].quantile(.25) - 1.5 * (X[c].quantile(.75) - X[c].quantile(.25))) |
             (X[c] > X[c].quantile(.75) + 1.5 * (X[c].quantile(.75) - X[c].quantile(.25))))
            for c in FEATURES).gt(0).sum()),
        "isolation_forest_flagged": int(flagged.sum()),
        "potable_rate_overall": round(float(y.mean()), 4),
        "potable_rate_iforest_outliers": round(float(y[flagged].mean()), 4),
    }
    return table, summary


# -------------------------------------------------------------------- figures

def fig_class_balance(y):
    counts = y.value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(6.4, 2.2))
    for i, (cls, n) in enumerate(counts.items()):
        ax.barh(i, n, color=P.CLASS_COLOUR[cls], height=0.55)
        ax.text(n + 20, i, f"{n:,}  ({n / len(y):.1%})", va="center", color=P.INK_2, fontsize=9)
    ax.set_yticks([0, 1], [config.LABELS[0], config.LABELS[1]])
    ax.set_xlim(0, counts.max() * 1.28)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Training samples")
    ax.set_title(f"Class balance: always guessing 'Not potable' scores {counts.max() / len(y):.1%}")
    return P.save(fig, "01_class_balance.png")


def fig_missing(X, y):
    cols = [c for c in FEATURES if X[c].isna().any()]
    fig, ax = plt.subplots(figsize=(6.4, 2.6))
    width = 0.36
    pos = np.arange(len(cols))
    for k, cls in enumerate((0, 1)):
        rates = [100 * X.loc[y == cls, c].isna().mean() for c in cols]
        ax.barh(pos + (k - 0.5) * (width + 0.03), rates, height=width,
                color=P.CLASS_COLOUR[cls], label=config.LABELS[cls])
        for p, r in zip(pos, rates):
            ax.text(r + 0.3, p + (k - 0.5) * (width + 0.03), f"{r:.1f}%", va="center",
                    fontsize=8, color=P.INK_2)
    ax.set_yticks(pos, [LABEL[c] for c in cols])
    ax.invert_yaxis()
    ax.set_xlabel("Rows missing (%)")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right")
    ax.set_title("Missing values are spread evenly across both classes")
    return P.save(fig, "02_missing_values.png")


def fig_distributions(X, y):
    fig, axes = plt.subplots(3, 3, figsize=(10, 7.4))
    for ax, f in zip(axes.ravel(), schema.FEATURES):
        for cls in (0, 1):
            values = X.loc[y == cls, f.name].dropna()
            kde = stats.gaussian_kde(values)
            grid = np.linspace(X[f.name].min(), X[f.name].max(), 300)
            ax.plot(grid, kde(grid), color=P.CLASS_COLOUR[cls], linewidth=1.8,
                    label=config.LABELS[cls])
        ax.set_title(f"{f.label}" + (f" ({f.unit})" if f.unit else ""), fontsize=9.5)
        ax.set_yticks([])
        ax.grid(axis="y", visible=False)
    axes[0, 0].legend(loc="upper left")
    fig.suptitle("Class-conditional densities: same centre, different spread",
                 x=0.01, ha="left", fontweight="bold", fontsize=12)
    fig.tight_layout()
    return P.save(fig, "03_distributions.png")


def fig_boxplots(X, y):
    fig, axes = plt.subplots(1, 9, figsize=(12, 3.6))
    for ax, f in zip(axes, schema.FEATURES):
        data = [X.loc[y == cls, f.name].dropna() for cls in (0, 1)]
        box = ax.boxplot(data, widths=0.55, patch_artist=True,
                         flierprops={"marker": "o", "markersize": 2.5, "alpha": 0.5,
                                     "markeredgewidth": 0},
                         medianprops={"color": P.SURFACE, "linewidth": 1.5},
                         whiskerprops={"color": P.GREY}, capprops={"color": P.GREY})
        for patch, flier, cls in zip(box["boxes"], box["fliers"], (0, 1)):
            patch.set_facecolor(P.CLASS_COLOUR[cls])
            patch.set_linewidth(0)
            flier.set_markerfacecolor(P.CLASS_COLOUR[cls])
        ax.set_xticks([])
        ax.set_title(f.label, fontsize=8.5, loc="center")
        ax.tick_params(labelsize=7.5)
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(color=P.CLASS_COLOUR[c], label=config.LABELS[c]) for c in (0, 1)],
               loc="upper right", ncol=2)
    fig.suptitle("Outliers (points beyond 1.5 x IQR) appear in both classes",
                 x=0.01, ha="left", fontweight="bold", fontsize=12)
    fig.tight_layout()
    return P.save(fig, "04_outliers_boxplot.png")


def _heatmap(ax, matrix, vmax, title, labels, show_y=True):
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("div", P.DIVERGING)
    cmap.set_bad(P.SURFACE)
    matrix = matrix.copy()
    np.fill_diagonal(matrix, np.nan)
    im = ax.imshow(matrix, cmap=cmap, vmin=-vmax, vmax=vmax)
    n = len(labels)
    ax.set_xticks(range(n), labels, rotation=45, ha="right")
    ax.set_yticks(range(n), labels if show_y else [""] * n)
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    for i in range(n):
        for j in range(n):
            if i != j:
                v = matrix[i, j]
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if abs(v) > 0.6 * vmax else P.INK_2)
    ax.set_title(title)
    return im


def fig_correlations(X, y, diff):
    labels = [SHORT[c] for c in FEATURES]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.2))
    _heatmap(axes[0], X.corr().values, 0.5, "Correlation, all training rows", labels)
    im = _heatmap(axes[1], diff.values, 0.5,
                  "Correlation difference, Potable minus Not potable", labels, show_y=False)
    fig.colorbar(im, ax=axes, shrink=0.75, pad=0.02)
    return P.save(fig, "05_correlations.png")


def fig_hidden_structure(X, y, table):
    top = table.head(3)
    fig, axes = plt.subplots(2, 3, figsize=(11, 6.6), sharex="col", sharey="col")
    for col, (_, row) in enumerate(top.iterrows()):
        a, b = row["pair"].split(" x ")
        for r, cls in enumerate((0, 1)):
            ax = axes[r, col]
            sub = X.loc[y == cls, [a, b]].dropna()
            ax.scatter(sub[a], sub[b], s=6, alpha=0.35, color=P.CLASS_COLOUR[cls], linewidths=0)
            slope, intercept = np.polyfit(sub[a], sub[b], 1)
            grid = np.linspace(sub[a].min(), sub[a].max(), 50)
            ax.plot(grid, slope * grid + intercept, color=P.INK, linewidth=1.5)
            r_val = sub[a].corr(sub[b])
            ax.set_title(f"{config.LABELS[cls]}: r = {r_val:+.2f}", fontsize=9.5)
            if r == 1:
                ax.set_xlabel(LABEL[a])
            ax.set_ylabel(LABEL[b])
    fig.suptitle("The hidden signal: the same pairs correlate in opposite directions by class",
                 x=0.01, ha="left", fontweight="bold", fontsize=12)
    fig.tight_layout()
    return P.save(fig, "06_hidden_structure.png")


def fig_effect_sizes(tests):
    t = tests.set_index("feature").loc[FEATURES]
    order = np.argsort(t["mutual_information"].values)
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    names = [LABEL[c] for c in np.array(FEATURES)[order]]

    axes[0].barh(names, t["mutual_information"].values[order], color=P.BLUE, height=0.6)
    axes[0].set_xlabel("Mutual information with the label (nats)")
    axes[0].set_title("Univariate information is tiny")
    axes[0].grid(axis="y", visible=False)

    ratio = t["sd_ratio"].values[order]
    axes[1].axvline(1, color=P.AXIS, linewidth=1)
    axes[1].hlines(range(len(ratio)), 1, ratio, color=P.GREY, linewidth=1.2)
    axes[1].scatter(ratio, range(len(ratio)), s=46, zorder=3, color=P.BLUE,
                    edgecolors=P.SURFACE, linewidths=1.5)
    for i, (r, p) in enumerate(zip(ratio, t["levene_p"].values[order])):
        if p < 0.001:
            axes[1].text(r + (0.01 if r > 1 else -0.01), i, "p<0.001",
                         va="center", ha="left" if r > 1 else "right", fontsize=7.5,
                         color=P.INK_2)
    axes[1].set_xlim(min(ratio) - 0.1, max(ratio) + 0.08)
    axes[1].set_xlabel("Std. deviation ratio, Potable / Not potable")
    axes[1].set_title("But the spread differs by class (Levene test)")
    axes[1].grid(axis="y", visible=False)
    fig.tight_layout()
    return P.save(fig, "07_effect_sizes.png")


# ------------------------------------------------------------------------ run

def run(X, y, log=print):
    config.ensure_dirs()
    tables = {}

    tables["overview"] = overview(X)
    tables["missingness"], co_missing = missingness(X, y)
    tables["class_tests"] = class_tests(X, y)
    diff, tables["class_correlation_difference"] = class_correlation_difference(X, y)
    tables["vif"] = vif(X)
    tables["outliers"], outlier_summary = outliers(X, y)

    for name, table in tables.items():
        table.to_csv(config.TABLE_DIR / f"eda_{name}.csv", index=False)
    pd.DataFrame([outlier_summary]).to_csv(config.TABLE_DIR / "eda_outlier_summary.csv",
                                           index=False)

    fig_class_balance(y)
    fig_missing(X, y)
    fig_distributions(X, y)
    fig_boxplots(X, y)
    fig_correlations(X, y, diff)
    fig_hidden_structure(X, y, tables["class_correlation_difference"])
    fig_effect_sizes(tables["class_tests"])

    log("  variable types     : " + ", ".join(
        f"{t}={n}" for t, n in tables["overview"]["variable_type"].value_counts().items())
        + " (+ binary nominal target)")
    log("  missing values     :")
    log(tables["missingness"][["feature", "missing_pct", "chi2_p_vs_target",
                               "min_bonferroni_p_vs_other_features", "verdict"]]
        .to_string(index=False, header=True).replace("\n", "\n    ").join(["    ", ""]))
    log(f"  rows missing 2+    : {co_missing}")
    log(f"  outliers           : {outlier_summary}")
    log("  top class-contrast pairs:")
    log(tables["class_correlation_difference"].head(5).to_string(index=False)
        .replace("\n", "\n    ").join(["    ", ""]))
    return tables, {"co_missing": co_missing, "outliers": outlier_summary}
