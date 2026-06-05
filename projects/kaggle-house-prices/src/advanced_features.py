"""Advanced feature preprocessing for Kaggle House Prices V2.

The V2 preprocessing keeps the baseline contract of returning aligned train/test
matrices and a log-transformed target, but adds domain-aware missing-value
handling, hand-coded ordinal variables, high-signal engineered features, and a
careful log1p transform for strongly right-skewed numeric predictors.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

NONE_CATEGORICAL_COLS = [
    "Alley",
    "BsmtQual",
    "BsmtCond",
    "BsmtExposure",
    "BsmtFinType1",
    "BsmtFinType2",
    "FireplaceQu",
    "GarageType",
    "GarageFinish",
    "GarageQual",
    "GarageCond",
    "PoolQC",
    "Fence",
    "MiscFeature",
    "MasVnrType",
]

ZERO_NUMERIC_COLS = [
    "GarageYrBlt",
    "GarageCars",
    "GarageArea",
    "BsmtFinSF1",
    "BsmtFinSF2",
    "BsmtUnfSF",
    "TotalBsmtSF",
    "BsmtFullBath",
    "BsmtHalfBath",
    "MasVnrArea",
]

QUALITY_COLS = [
    "ExterQual",
    "ExterCond",
    "BsmtQual",
    "BsmtCond",
    "HeatingQC",
    "KitchenQual",
    "FireplaceQu",
    "GarageQual",
    "GarageCond",
    "PoolQC",
]

ORDINAL_COLS = QUALITY_COLS + [
    "BsmtExposure",
    "BsmtFinType1",
    "BsmtFinType2",
    "GarageFinish",
    "LotShape",
    "LandSlope",
    "PavedDrive",
]

QUALITY_MAPPING = {
    "None": 0,
    "Missing": 0,
    "Po": 1,
    "Fa": 2,
    "TA": 3,
    "Gd": 4,
    "Ex": 5,
}

OTHER_ORDINAL_MAPPINGS = {
    "BsmtExposure": {"None": 0, "Missing": 0, "No": 1, "Mn": 2, "Av": 3, "Gd": 4},
    "BsmtFinType1": {
        "None": 0,
        "Missing": 0,
        "Unf": 1,
        "LwQ": 2,
        "Rec": 3,
        "BLQ": 4,
        "ALQ": 5,
        "GLQ": 6,
    },
    "BsmtFinType2": {
        "None": 0,
        "Missing": 0,
        "Unf": 1,
        "LwQ": 2,
        "Rec": 3,
        "BLQ": 4,
        "ALQ": 5,
        "GLQ": 6,
    },
    "GarageFinish": {"None": 0, "Missing": 0, "Unf": 1, "RFn": 2, "Fin": 3},
    "LotShape": {"Reg": 0, "IR1": 1, "IR2": 2, "IR3": 3},
    "LandSlope": {"Gtl": 0, "Mod": 1, "Sev": 2},
    "PavedDrive": {"N": 0, "P": 1, "Y": 2},
}


def _add_numeric_feature(df: pd.DataFrame, name: str, terms: list[str]) -> None:
    """Add a sum feature, treating absent source columns as zero."""
    df[name] = sum(df[col] if col in df.columns else 0 for col in terms)


def _positive_indicator(df: pd.DataFrame, col: str) -> pd.Series:
    """Return a binary indicator for a positive numeric column if present."""
    if col not in df.columns:
        return pd.Series(0, index=df.index, dtype=int)
    return (df[col] > 0).astype(int)


def _add_advanced_features(combined: pd.DataFrame) -> pd.DataFrame:
    """Create high-signal domain features on the concatenated feature table."""
    combined = combined.copy()

    _add_numeric_feature(combined, "TotalSF", ["TotalBsmtSF", "1stFlrSF", "2ndFlrSF"])
    _add_numeric_feature(combined, "TotalFlrSF", ["1stFlrSF", "2ndFlrSF"])
    _add_numeric_feature(
        combined,
        "TotalPorchSF",
        ["OpenPorchSF", "EnclosedPorch", "3SsnPorch", "ScreenPorch"],
    )

    full_bath = combined["FullBath"] if "FullBath" in combined.columns else 0
    half_bath = combined["HalfBath"] if "HalfBath" in combined.columns else 0
    bsmt_full_bath = combined["BsmtFullBath"] if "BsmtFullBath" in combined.columns else 0
    bsmt_half_bath = combined["BsmtHalfBath"] if "BsmtHalfBath" in combined.columns else 0
    combined["TotalBathrooms"] = full_bath + 0.5 * half_bath + bsmt_full_bath + 0.5 * bsmt_half_bath

    if {"YrSold", "YearBuilt"}.issubset(combined.columns):
        combined["HouseAge"] = (combined["YrSold"] - combined["YearBuilt"]).clip(lower=0)
    if {"YrSold", "YearRemodAdd"}.issubset(combined.columns):
        combined["RemodAge"] = (combined["YrSold"] - combined["YearRemodAdd"]).clip(lower=0)
    if {"YrSold", "GarageYrBlt"}.issubset(combined.columns):
        combined["GarageAge"] = np.where(
            combined["GarageYrBlt"] > 0,
            (combined["YrSold"] - combined["GarageYrBlt"]).clip(lower=0),
            0,
        )

    combined["HasBasement"] = _positive_indicator(combined, "TotalBsmtSF")
    combined["HasGarage"] = _positive_indicator(combined, "GarageArea")
    combined["HasFireplace"] = _positive_indicator(combined, "Fireplaces")
    combined["HasPool"] = _positive_indicator(combined, "PoolArea")
    combined["HasPorch"] = _positive_indicator(combined, "TotalPorchSF")

    if {"YearBuilt", "YearRemodAdd"}.issubset(combined.columns):
        combined["WasRemodeled"] = (combined["YearRemodAdd"] != combined["YearBuilt"]).astype(int)
    if {"YearBuilt", "YrSold"}.issubset(combined.columns):
        combined["IsNewHouse"] = (combined["YearBuilt"] == combined["YrSold"]).astype(int)

    if {"OverallQual", "TotalSF"}.issubset(combined.columns):
        combined["OverallQual_TotalSF"] = combined["OverallQual"] * combined["TotalSF"]
    if {"OverallQual", "GrLivArea"}.issubset(combined.columns):
        combined["OverallQual_GrLivArea"] = combined["OverallQual"] * combined["GrLivArea"]
    if {"OverallQual", "GarageArea"}.issubset(combined.columns):
        combined["OverallQual_GarageArea"] = combined["OverallQual"] * combined["GarageArea"]
    if {"OverallQual", "TotalBathrooms"}.issubset(combined.columns):
        combined["OverallQual_TotalBathrooms"] = combined["OverallQual"] * combined["TotalBathrooms"]

    return combined


def _fill_categorical_with_mode(series: pd.Series) -> pd.Series:
    """Fill a categorical series with its mode, falling back to Missing."""
    mode = series.mode(dropna=True)
    fill_value = mode.iloc[0] if not mode.empty else "Missing"
    return series.fillna(fill_value)


def _encode_ordinals(combined: pd.DataFrame) -> pd.DataFrame:
    """Encode important ordered categorical columns with domain mappings."""
    combined = combined.copy()
    mappings = {col: QUALITY_MAPPING for col in QUALITY_COLS}
    mappings.update(OTHER_ORDINAL_MAPPINGS)

    for col, mapping in mappings.items():
        if col in combined.columns:
            combined[col] = combined[col].map(mapping).fillna(0).astype(int)

    return combined


def _log_transform_skewed_numeric_features(
    combined: pd.DataFrame,
    n_train: int,
    ordinal_cols: list[str],
    skew_threshold: float = 0.75,
) -> tuple[pd.DataFrame, list[str]]:
    """Apply log1p to skewed numeric predictors.

    Skew is estimated on the training rows only to avoid using the test
    distribution for model-selection decisions. The transform is applied to the
    full concatenated matrix afterward so train and test stay aligned. Binary
    flags, manually encoded ordinal variables, and columns with negative values
    are excluded; this keeps interpretable order encodings intact and guarantees
    that ``np.log1p`` is mathematically safe.
    """
    combined = combined.copy()
    numeric_cols = list(combined.select_dtypes(include=[np.number]).columns)
    train_numeric = combined.iloc[:n_train][numeric_cols]
    skewness = train_numeric.skew(numeric_only=True).sort_values(ascending=False)

    skewed_cols: list[str] = []
    ordinal_set = set(ordinal_cols)
    for col, skew_value in skewness.items():
        if col in ordinal_set:
            continue
        if train_numeric[col].nunique(dropna=False) <= 2:
            continue
        if combined[col].min() < 0:
            continue
        if skew_value > skew_threshold:
            skewed_cols.append(col)

    if skewed_cols:
        combined[skewed_cols] = np.log1p(combined[skewed_cols])

    return combined, skewed_cols


def prepare_advanced_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_col: str = "SalePrice",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Prepare aligned advanced train/test features and log target.

    Steps:
    - split and log-transform ``SalePrice``;
    - remove ``Id`` from model features while leaving original frames unchanged;
    - concatenate train/test temporarily for consistent preprocessing;
    - impute missing values with domain semantics where ``NaN`` means absence;
    - add engineered area, age, indicator, and interaction features;
    - encode important ordinal variables manually;
    - log-transform strongly right-skewed numeric predictors;
    - one-hot encode all remaining categorical variables.

    Returns:
        ``(X_train, X_test, y)`` where ``y`` is ``log1p(SalePrice)``.
    """
    if target_col not in train_df.columns:
        raise ValueError(f"Target column `{target_col}` was not found in train data.")

    y = np.log1p(train_df[target_col])
    train_features = train_df.drop(columns=[target_col, "Id"], errors="ignore").copy()
    test_features = test_df.drop(columns=["Id"], errors="ignore").copy()

    n_train = len(train_features)
    combined = pd.concat([train_features, test_features], axis=0, ignore_index=True)

    none_categorical = [col for col in NONE_CATEGORICAL_COLS if col in combined.columns]
    zero_numeric = [col for col in ZERO_NUMERIC_COLS if col in combined.columns]
    combined[none_categorical] = combined[none_categorical].fillna("None")
    combined[zero_numeric] = combined[zero_numeric].fillna(0)

    numeric_cols = combined.select_dtypes(include=[np.number]).columns.difference(zero_numeric)
    categorical_cols = combined.select_dtypes(exclude=[np.number]).columns.difference(none_categorical)

    if len(numeric_cols) > 0:
        combined[numeric_cols] = combined[numeric_cols].fillna(combined[numeric_cols].median())

    for col in categorical_cols:
        combined[col] = _fill_categorical_with_mode(combined[col])

    combined = _add_advanced_features(combined)
    combined = _encode_ordinals(combined)
    combined, skewed_cols = _log_transform_skewed_numeric_features(combined, n_train, ORDINAL_COLS)

    remaining_categorical = combined.select_dtypes(exclude=[np.number]).columns
    combined_encoded = pd.get_dummies(combined, columns=remaining_categorical, drop_first=False)

    X_train = combined_encoded.iloc[:n_train].copy()
    X_test = combined_encoded.iloc[n_train:].copy()
    X_train.attrs["skewed_log1p_columns"] = skewed_cols
    X_test.attrs["skewed_log1p_columns"] = skewed_cols

    return X_train, X_test, y
