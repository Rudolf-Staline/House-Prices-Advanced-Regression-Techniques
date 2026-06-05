"""Feature preprocessing for the Kaggle House Prices baseline."""

import numpy as np
import pandas as pd


def prepare_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_col: str = "SalePrice",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Prepare aligned train/test feature matrices and log-transformed target.

    The preprocessing intentionally stays simple and robust for a first
    reproducible baseline:
    - remove the target from training features;
    - concatenate train and test features before encoding;
    - fill numeric missing values with the train/test median;
    - fill categorical missing values with the explicit value `Missing`;
    - one-hot encode categorical variables with pandas;
    - split the combined matrix back into aligned train/test matrices;
    - transform the target with `log1p`, matching Kaggle's RMSE-on-log objective.
    """
    if target_col not in train_df.columns:
        raise ValueError(f"Target column `{target_col}` was not found in train data.")

    y = np.log1p(train_df[target_col])
    train_features = train_df.drop(columns=[target_col]).copy()
    test_features = test_df.copy()

    n_train = len(train_features)
    combined = pd.concat([train_features, test_features], axis=0, ignore_index=True)

    numeric_cols = combined.select_dtypes(include=["number"]).columns
    categorical_cols = combined.select_dtypes(exclude=["number"]).columns

    if len(numeric_cols) > 0:
        combined[numeric_cols] = combined[numeric_cols].fillna(combined[numeric_cols].median())

    if len(categorical_cols) > 0:
        combined[categorical_cols] = combined[categorical_cols].fillna("Missing")

    combined_encoded = pd.get_dummies(combined, columns=categorical_cols, drop_first=False)

    X_train = combined_encoded.iloc[:n_train].copy()
    X_test = combined_encoded.iloc[n_train:].copy()

    return X_train, X_test, y
