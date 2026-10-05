"""Load the trained artifact and score new samples. Used by the API and the CLI."""

import json
import threading
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd

from . import config, guard, schema

VERDICT_TEXT = {
    "potable": "Likely potable",
    "not_potable": "Likely not potable",
    "uncertain": "Uncertain: confirm with a laboratory test",
}


class ModelNotTrained(RuntimeError):
    pass


class Predictor:
    def __init__(self, bundle):
        self.bundle = bundle
        self.model = bundle["model"]
        self.profile = bundle["profile"]
        self.thresholds = bundle["thresholds"]
        self._log_lock = threading.Lock()

    @classmethod
    def load(cls, path=config.MODEL_PATH):
        if not path.exists():
            raise ModelNotTrained(f"no model at {path}; run: python -m aquasafe.pipeline")
        return cls(joblib.load(path))

    # ----------------------------------------------------------------- scoring

    def _frame(self, rows):
        return pd.DataFrame([[np.nan if v is None else v for v in r] for r in rows],
                            columns=schema.FEATURE_NAMES)

    def probability(self, rows):
        return self.model.predict_proba(self._frame(rows))[:, 1]

    def verdict(self, p):
        t = self.thresholds
        if p >= t["potable"]:
            return "potable"
        if p <= t["not_potable"]:
            return "not_potable"
        return "uncertain"

    def contributions(self, values, p):
        """How far each reading moves the probability away from a typical sample.

        Each present reading is swapped for the training median in turn and the
        sample rescored. A positive number means the reading pushes towards
        'potable'. One batch of at most nine extra rows per request.
        """
        present = [i for i, v in enumerate(values) if v is not None]
        if not present:
            return []
        rows = []
        for i in present:
            row = list(values)
            row[i] = float(self.profile.median_[i])
            rows.append(row)
        swapped = self.probability(rows)
        out = [{"feature": schema.FEATURE_NAMES[i], "label": schema.FEATURES[i].label,
                "effect": round(float(p - q), 4)} for i, q in zip(present, swapped)]
        return sorted(out, key=lambda c: abs(c["effect"]), reverse=True)

    def predict(self, record, explain=True):
        values, issues, ok = guard.check(record, self.profile)
        result = {
            "ok": ok,
            "inputs": dict(zip(schema.FEATURE_NAMES, values)),
            "issues": issues,
            "guidelines": [{"feature": f.name, "label": f.label, "unit": f.unit, "value": v,
                            "status": schema.guideline_status(f.name, v),
                            "min": f.guideline_min, "max": f.guideline_max,
                            "source": f.guideline_source}
                           for f, v in zip(schema.FEATURES, values)],
            "model_version": self.bundle["version"],
        }
        if not ok:
            return result

        p = float(self.probability([values])[0])
        _, pct = self.profile.typicality(self._frame([values]).values)
        unusual = bool(pct[0] < 0.01)
        if unusual:
            issues.append({"feature": None, "level": "warning",
                           "message": "This combination of readings is unusual compared with "
                                      "the training data; treat the prediction with caution"})
        verdict = self.verdict(p)
        result.update({
            "probability_potable": round(p, 4),
            "label": int(p >= self.thresholds["decision"]),
            "label_text": config.LABELS[int(p >= self.thresholds["decision"])],
            "verdict": verdict,
            "verdict_text": VERDICT_TEXT[verdict],
            "typicality_percentile": round(float(pct[0]) * 100, 1),
            "unusual_input": unusual,
            "thresholds": self.thresholds,
        })
        if explain:
            result["contributions"] = self.contributions(values, p)
        self._log(result)
        return result

    def predict_batch(self, frame):
        """Score a DataFrame. Rows that fail the guard come back with ok = False."""
        results = [self.predict(rec, explain=False)
                   for rec in frame.reindex(columns=schema.FEATURE_NAMES).to_dict("records")]
        return results

    # ---------------------------------------------------------------- logging

    def _log(self, result):
        entry = {"time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                 "version": result["model_version"], "inputs": result["inputs"],
                 "probability": result.get("probability_potable"),
                 "verdict": result.get("verdict"), "unusual": result.get("unusual_input")}
        try:
            with self._log_lock, open(config.PREDICTION_LOG, "a") as handle:
                handle.write(json.dumps(entry) + "\n")
        except OSError:
            pass   # logging must never break a prediction

    def info(self):
        b = self.bundle
        return {
            "version": b["version"], "trained_at": b["trained_at"],
            "model": b["description"], "thresholds": self.thresholds,
            "metrics": b["metrics"], "features": schema.as_dicts(),
            "training_ranges": self.profile.as_dict(),
        }
