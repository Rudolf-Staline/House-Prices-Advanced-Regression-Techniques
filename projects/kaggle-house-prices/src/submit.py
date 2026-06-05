"""Train the baseline model and create a Kaggle submission file.

Run from `projects/kaggle-house-prices/` with:

    python -m src.submit
"""

import numpy as np
import pandas as pd

from .config import MODEL_PATH, SUBMISSION_PATH
from .data import load_train_test
from .features import prepare_features
from .model import fit_best_model, save_model, train_baseline_models


def create_submission() -> pd.DataFrame:
    """Train baseline models and write a Kaggle-compatible submission CSV."""
    train_df, test_df = load_train_test()
    X_train, X_test, y = prepare_features(train_df, test_df)

    scores = train_baseline_models(X_train, y)
    print("Cross-validation scores (RMSE on log target):")
    print(scores.to_string(index=False))

    model = fit_best_model(X_train, y)
    save_model(model, MODEL_PATH)

    predictions_log = model.predict(X_test)
    predictions = np.expm1(predictions_log)
    predictions = np.maximum(predictions, 0)

    submission = pd.DataFrame(
        {
            "Id": test_df["Id"].values,
            "SalePrice": predictions,
        }
    )

    SUBMISSION_PATH.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(SUBMISSION_PATH, index=False)
    print(f"Submission saved to: {SUBMISSION_PATH}")
    return submission


if __name__ == "__main__":
    create_submission()
