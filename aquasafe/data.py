"""Loading, validation and the one train/test split every stage shares."""

import json

import pandas as pd
from sklearn.model_selection import train_test_split

from . import config, schema


class DataError(ValueError):
    pass


def load(path=config.DATA_PATH):
    """Read the raw CSV and check it matches the expected schema."""
    if not path.exists():
        raise DataError(
            f"{path} not found. Download water_potability.csv from "
            "https://www.kaggle.com/datasets/adityakadiwal/water-potability")

    frame = pd.read_csv(path)
    expected = schema.FEATURE_NAMES + [config.TARGET]
    missing = set(expected) - set(frame.columns)
    if missing:
        raise DataError(f"columns missing from {path.name}: {sorted(missing)}")

    frame = frame[expected].apply(pd.to_numeric, errors="coerce")
    frame = frame.dropna(subset=[config.TARGET]).reset_index(drop=True)
    frame[config.TARGET] = frame[config.TARGET].astype(int)

    labels = set(frame[config.TARGET].unique())
    if not labels <= {0, 1}:
        raise DataError(f"target must be 0/1, found {sorted(labels)}")
    return frame


def split(frame):
    """Stratified 80/20 split. The indices are saved so every stage reuses them."""
    train_idx, test_idx = train_test_split(
        frame.index, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE,
        stratify=frame[config.TARGET])
    config.ensure_dirs()
    config.SPLIT_PATH.write_text(json.dumps(
        {"train": sorted(map(int, train_idx)), "test": sorted(map(int, test_idx))}))
    return xy(frame.loc[sorted(train_idx)]), xy(frame.loc[sorted(test_idx)])


def xy(frame):
    return frame[schema.FEATURE_NAMES], frame[config.TARGET]
