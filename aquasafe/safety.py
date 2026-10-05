"""Dataset 2 (water quality, is_safe): score one sample with every model from the study.

The bundle is written by scripts/safety_zoo.py. The final model is the one the
study selected on training-split cross-validation; the others are shown so a
sample can be compared across every model that was tried.
"""

import numpy as np
import pandas as pd

from . import config

ZOO_PATH = config.ARTIFACT_DIR / "safety_zoo.joblib"
MAX_READING = 1e6


class SafetyNotTrained(RuntimeError):
    pass


class SafetyPredictor:
    def __init__(self, bundle):
        self.bundle = bundle
        self.features = bundle["features"]

    @classmethod
    def load(cls, path=ZOO_PATH):
        if not path.exists():
            raise SafetyNotTrained("Dataset 2 models not found; run scripts/safety_study.py "
                                   "then scripts/safety_zoo.py")
        import joblib
        return cls(joblib.load(path))

    def info(self):
        b = self.bundle
        return {
            "final_model": b["final"],
            "final_cut": b["models"][b["final"]]["cut"],
            "features": [{"name": f, "label": b["labels"][f], "limit": b["limits"][f], **b["ranges"][f]}
                         for f in self.features],
            "leaderboard": b["leaderboard"],
            "n_test": b["n_test"],
            "baseline": b["baseline"],
            "examples": b["examples"],
        }

    def validate(self, values):
        issues = []
        for f in self.features:
            v = values.get(f)
            label = self.bundle["labels"][f]
            if v is None:
                issues.append({"level": "error", "feature": f, "message": f"{label} is missing; all 20 readings are needed"})
            elif not np.isfinite(v) or v < 0 or v > MAX_READING:
                issues.append({"level": "error", "feature": f,
                               "message": f"{label} = {v:g} is not a valid concentration"})
        unknown = sorted(set(values) - set(self.features))
        if unknown:
            issues.append({"level": "error", "feature": None, "message": "unknown fields: " + ", ".join(unknown)})
        return issues

    def predict(self, values):
        issues = self.validate(values)
        if issues:
            return {"ok": False, "issues": issues}
        b = self.bundle
        x = pd.DataFrame([[float(values[f]) for f in self.features]], columns=self.features)
        for f in self.features:
            r = b["ranges"][f]
            if not r["min"] <= x.at[0, f] <= r["max"]:
                issues.append({"level": "warning", "feature": f,
                               "message": f"{b['labels'][f]} = {x.at[0, f]:g} is outside the training range "
                                          f"({r['min']:g} to {r['max']:g}); predictions are extrapolations"})

        board = {r["model"]: r for r in b["leaderboard"]}
        probs, rows = {}, []
        for name, spec in b["models"].items():
            if "model" in spec:
                p = float(spec["model"].predict_proba(x)[0, 1])
            else:
                base = np.array([[probs[m] for m in spec["members"]]])
                p = float(spec["meta"].predict_proba(base)[0, 1]) if "meta" in spec else float(base.mean())
            probs[name] = p
            rows.append({"model": name, "group": spec["group"], "probability_safe": p, "cut": spec["cut"],
                         "verdict": "safe" if p >= spec["cut"] else "unsafe", "final": name == b["final"],
                         "test_f1": board[name]["f1"], "test_accuracy": board[name]["accuracy"],
                         "test_precision": board[name]["precision"]})

        final = next(r for r in rows if r["final"])
        limits = [{"feature": f, "label": b["labels"][f], "value": float(values[f]), "limit": b["limits"][f],
                   "status": "above" if values[f] > b["limits"][f] else "ok"} for f in self.features]
        return {
            "ok": True, "issues": issues,
            "final_model": final["model"], "probability_safe": final["probability_safe"], "cut": final["cut"],
            "verdict": final["verdict"],
            "votes_safe": sum(r["verdict"] == "safe" for r in rows), "n_models": len(rows),
            "models": rows, "limits": limits,
        }
