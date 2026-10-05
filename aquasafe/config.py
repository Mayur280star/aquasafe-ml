"""Paths and run settings shared by every stage."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "raw" / "water_potability.csv"

ARTIFACT_DIR = ROOT / "artifacts"
MODEL_PATH = ARTIFACT_DIR / "aquasafe_model.joblib"
MODEL_CARD_PATH = ARTIFACT_DIR / "model_card.json"
SPLIT_PATH = ARTIFACT_DIR / "split_indices.json"
PREDICTION_LOG = ARTIFACT_DIR / "prediction_log.jsonl"

REPORT_DIR = ROOT / "reports"
FIGURE_DIR = REPORT_DIR / "figures"
TABLE_DIR = REPORT_DIR / "tables"

TARGET = "Potability"
LABELS = {0: "Not potable", 1: "Potable"}

RANDOM_STATE = 42
TEST_SIZE = 0.20

# Model selection: 5 folds x 3 repeats. Tuning uses 5 x 2 to keep trials cheap.
CV_FOLDS = 5
CV_REPEATS = 3
TUNE_REPEATS = 2

# Operating points for the three-way verdict, chosen on out-of-fold predictions.
TARGET_PRECISION_POTABLE = 0.80   # "Potable" is only said when this precise
TARGET_PRECISION_UNSAFE = 0.75    # "Not potable" is only said when this precise


def ensure_dirs():
    for path in (ARTIFACT_DIR, FIGURE_DIR, TABLE_DIR):
        path.mkdir(parents=True, exist_ok=True)
