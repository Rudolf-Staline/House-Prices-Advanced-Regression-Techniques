"""Data loading utilities for the Kaggle House Prices dataset."""

from pathlib import Path

import pandas as pd

from .config import RAW_DATA_DIR, TEST_PATH, TRAIN_PATH


MISSING_DATA_MESSAGE = f"""
Kaggle data files were not found.

Expected files:
- {TRAIN_PATH}
- {TEST_PATH}

Download the competition files from Kaggle and place them manually in:
{RAW_DATA_DIR}

Required filenames:
- train.csv
- test.csv
- sample_submission.csv (recommended for format reference)

This project never downloads data automatically and never stores Kaggle credentials.
""".strip()


def _ensure_file_exists(path: Path) -> None:
    """Raise a clear error when a required Kaggle file is missing."""
    if not path.exists():
        raise FileNotFoundError(MISSING_DATA_MESSAGE)


def load_train_test() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load Kaggle train and test CSV files from `data/raw/`.

    Returns:
        A tuple `(train_df, test_df)`.

    Raises:
        FileNotFoundError: if `train.csv` or `test.csv` is missing.
    """
    _ensure_file_exists(TRAIN_PATH)
    _ensure_file_exists(TEST_PATH)

    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    return train_df, test_df
