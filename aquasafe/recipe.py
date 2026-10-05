"""Build a full pipeline from a recipe: a plain, JSON-serialisable dict.

    bounds       True to turn physically impossible readings into NaN
    imputer      median | mean | knn | iterative | hybrid | none
    indicator    True to append missing-value indicator columns
    outliers     none | winsor_1 | winsor_05 | iqr | iforest
    domain       True to append guideline flags and the TDS / conductivity ratio
    scaler       standard | minmax | robust | quantile | none
    interactions number of class-contrast interaction pairs (0 for none)
    deviation    True to append |z| features
    balance      none | class_weight | smote | smote_tomek
    model        a key of models.NAMES
    params       model hyper-parameters

Every step lives inside the pipeline, so cross-validation refits all of them on
each training fold and nothing is learned from held-out rows.
"""

from imblearn.combine import SMOTETomek
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from imblearn import FunctionSampler
from sklearn.preprocessing import (MinMaxScaler, QuantileTransformer, RobustScaler,
                                   StandardScaler)

from . import config, models
from . import transformers as T

DEFAULT = {
    "bounds": True,
    "imputer": "median",
    "indicator": False,
    "outliers": "none",
    "domain": False,
    "scaler": "standard",
    "interactions": 0,
    "deviation": False,
    "balance": "none",
    "model": "svm",
    "params": {},
}


def complete(recipe):
    return {**DEFAULT, **recipe}


def _scaler(name):
    return {
        "standard": StandardScaler(),
        "minmax": MinMaxScaler(),
        "robust": RobustScaler(),
        "quantile": QuantileTransformer(n_quantiles=500, output_distribution="normal",
                                        random_state=config.RANDOM_STATE),
        "none": T.Passthrough(),
    }[name]


def _outliers(name):
    if name == "winsor_1":
        return T.Winsorizer("quantile", q=0.01)
    if name == "winsor_05":
        return T.Winsorizer("quantile", q=0.005)
    if name == "iqr":
        return T.Winsorizer("iqr", k=1.5)
    raise ValueError(f"unknown outlier strategy: {name}")


def build(recipe):
    r = complete(recipe)
    steps = []

    if r["bounds"]:
        steps.append(("bounds", T.PhysicalBounds()))
    if r["outliers"] == "iforest":
        steps.append(("drop_outliers", FunctionSampler(
            func=T.remove_outliers, validate=False,
            kw_args={"random_state": config.RANDOM_STATE})))

    steps.append(("impute", T.Imputer(r["imputer"], r["indicator"],
                                      random_state=config.RANDOM_STATE)))

    if r["outliers"] not in ("none", "iforest"):
        steps.append(("winsorize", _outliers(r["outliers"])))
    if r["domain"]:
        steps.append(("domain", T.DomainFeatures()))

    steps.append(("scale", _scaler(r["scaler"])))

    if r["interactions"]:
        steps.append(("interactions", T.InteractionFeatures(k=r["interactions"])))
    if r["deviation"]:
        steps.append(("deviation", T.DeviationFeatures()))

    if r["balance"] == "smote":
        steps.append(("smote", SMOTE(random_state=config.RANDOM_STATE)))
    elif r["balance"] == "smote_tomek":
        steps.append(("smote", SMOTETomek(random_state=config.RANDOM_STATE)))

    class_weight = r["balance"] == "class_weight" and models.supports_class_weight(r["model"])
    steps.append(("model", models.make(r["model"], r["params"], class_weight)))
    return Pipeline(steps)


def describe(recipe):
    r = complete(recipe)
    parts = [models.NAMES[r["model"]], f"impute={r['imputer']}"]
    if r["indicator"]:
        parts.append("+indicators")
    if r["outliers"] != "none":
        parts.append(f"outliers={r['outliers']}")
    if r["domain"]:
        parts.append("+domain")
    parts.append(f"scale={r['scaler']}")
    if r["interactions"]:
        parts.append(f"+{r['interactions']} interactions")
    if r["deviation"]:
        parts.append("+|z|")
    if r["balance"] != "none":
        parts.append(f"balance={r['balance']}")
    return ", ".join(parts)
