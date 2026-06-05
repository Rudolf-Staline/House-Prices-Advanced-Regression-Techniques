"""Project paths for the Kaggle House Prices baseline."""

from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
SUBMISSION_DIR = DATA_DIR / "submissions"

TRAIN_PATH = RAW_DATA_DIR / "train.csv"
TEST_PATH = RAW_DATA_DIR / "test.csv"
SAMPLE_SUBMISSION_PATH = RAW_DATA_DIR / "sample_submission.csv"
MODEL_PATH = PROCESSED_DATA_DIR / "baseline_model.joblib"
SUBMISSION_PATH = SUBMISSION_DIR / "submission_baseline.csv"
