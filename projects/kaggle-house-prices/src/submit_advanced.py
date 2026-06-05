"""Train advanced models and create Kaggle submission files.

Run from `projects/kaggle-house-prices/` with:

    python -m src.submit_advanced
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .advanced_features import prepare_advanced_features
from .advanced_model import (
    blend_predictions,
    evaluate_advanced_models,
    fit_model_by_name,
    fit_models_by_names,
    recommended_blend_weights,
)
from .config import SUBMISSION_DIR
from .data import load_train_test
from .submit import validate_submission

ADVANCED_BEST_SINGLE_PATH = SUBMISSION_DIR / "submission_advanced_best_single.csv"
ADVANCED_BLEND_PATH = SUBMISSION_DIR / "submission_advanced_blend.csv"


def _to_sale_price(predictions_log: np.ndarray) -> np.ndarray:
    """Convert log-scale predictions to strictly positive sale prices."""
    predictions = np.expm1(predictions_log)
    return np.maximum(predictions, 1e-9)


def _save_submission(predictions_log: np.ndarray, test_df: pd.DataFrame, path) -> pd.DataFrame:
    """Validate and save one Kaggle-compatible submission file."""
    submission = pd.DataFrame(
        {
            "Id": test_df["Id"].values,
            "SalePrice": _to_sale_price(predictions_log),
        }
    )
    validate_submission(submission, test_df)
    path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(path, index=False)
    print(f"Submission saved to: {path}")
    return submission


def create_advanced_submissions() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Train V2 models and write best-single and blended submission CSVs."""
    train_df, test_df = load_train_test()
    X_train, X_test, y = prepare_advanced_features(train_df, test_df)

    print(f"Advanced feature matrix: train={X_train.shape}, test={X_test.shape}")
    skewed_cols = X_train.attrs.get("skewed_log1p_columns", [])
    print(f"log1p-transformed skewed numeric features ({len(skewed_cols)}): {skewed_cols}")

    scores = evaluate_advanced_models(X_train, y)
    print("Cross-validation scores (RMSE on log target):")
    print(scores.to_string(index=False))

    best_name = str(scores.iloc[0]["model"])
    print(f"Best single model: {best_name}")
    best_model = fit_model_by_name(best_name, X_train, y)
    best_predictions_log = best_model.predict(X_test)
    best_submission = _save_submission(best_predictions_log, test_df, ADVANCED_BEST_SINGLE_PATH)

    available_names = scores["model"].tolist()
    blend_weights = recommended_blend_weights(available_names)
    blend_model_names = [name for name in blend_weights if name in available_names]
    fitted_blend_models = fit_models_by_names(blend_model_names, X_train, y)
    blend_predictions_log = blend_predictions(fitted_blend_models, X_test, blend_weights)
    blend_submission = _save_submission(blend_predictions_log, test_df, ADVANCED_BLEND_PATH)

    normalized_weight_sum = sum(
        weight for name, weight in blend_weights.items() if name in fitted_blend_models
    )
    normalized_weights = {
        name: weight / normalized_weight_sum
        for name, weight in blend_weights.items()
        if name in fitted_blend_models
    }
    print("Blend weights used:")
    for name, weight in normalized_weights.items():
        print(f"- {name}: {weight:.3f}")

    return scores, best_submission, blend_submission


if __name__ == "__main__":
    create_advanced_submissions()
