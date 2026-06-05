"""Advanced model utilities for Kaggle House Prices V2."""

from __future__ import annotations

import importlib.util
from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import ElasticNet, Lasso, Ridge
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

RANDOM_STATE = 42

DEFAULT_BLEND_WEIGHTS = {
    "elasticnet": 0.25,
    "lasso": 0.20,
    "ridge": 0.10,
    "gradient_boosting": 0.25,
    "hist_gradient_boosting": 0.20,
}

OPTIONAL_BLEND_WEIGHTS = {
    "elasticnet": 0.15,
    "lasso": 0.10,
    "ridge": 0.05,
    "gradient_boosting": 0.20,
    "hist_gradient_boosting": 0.15,
    "xgboost": 0.20,
    "lightgbm": 0.15,
}


def rmse_cv(model, X: pd.DataFrame, y: pd.Series, cv: int = 5) -> float:
    """Return mean cross-validated RMSE on the log-transformed target."""
    kfold = KFold(n_splits=cv, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_val_score(
        model,
        X,
        y,
        scoring="neg_root_mean_squared_error",
        cv=kfold,
        n_jobs=-1,
    )
    return float(-scores.mean())


def _linear_model(model) -> Pipeline:
    """Wrap regularized linear models with robust scaling."""
    return Pipeline(
        steps=[
            ("scaler", RobustScaler()),
            ("model", model),
        ]
    )


def get_advanced_models() -> dict[str, object]:
    """Build the advanced model zoo.

    XGBoost and LightGBM are optional. They are added only when their packages
    are installed, so the project remains usable with the core requirements.
    """
    models: dict[str, object] = {
        "ridge": _linear_model(Ridge(alpha=10.0, random_state=RANDOM_STATE)),
        "lasso": _linear_model(
            Lasso(alpha=0.0005, max_iter=50000, random_state=RANDOM_STATE)
        ),
        "elasticnet": _linear_model(
            ElasticNet(
                alpha=0.0005,
                l1_ratio=0.8,
                max_iter=50000,
                random_state=RANDOM_STATE,
            )
        ),
        "gradient_boosting": GradientBoostingRegressor(
            n_estimators=3000,
            learning_rate=0.03,
            max_depth=3,
            max_features="sqrt",
            min_samples_leaf=15,
            min_samples_split=10,
            loss="squared_error",
            random_state=RANDOM_STATE,
        ),
        "hist_gradient_boosting": HistGradientBoostingRegressor(
            learning_rate=0.03,
            max_iter=800,
            l2_regularization=0.01,
            random_state=RANDOM_STATE,
        ),
    }

    if importlib.util.find_spec("xgboost") is None:
        print("XGBoost is not installed; skipping optional xgboost model.")
    else:
        from xgboost import XGBRegressor

        models["xgboost"] = XGBRegressor(
            n_estimators=3000,
            learning_rate=0.02,
            max_depth=3,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="reg:squarederror",
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbosity=0,
        )

    if importlib.util.find_spec("lightgbm") is None:
        print("LightGBM is not installed; skipping optional lightgbm model.")
    else:
        from lightgbm import LGBMRegressor

        models["lightgbm"] = LGBMRegressor(
            n_estimators=3000,
            learning_rate=0.02,
            num_leaves=31,
            feature_fraction=0.8,
            bagging_fraction=0.8,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=-1,
        )

    return models


def evaluate_advanced_models(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """Evaluate all available advanced models and return a sorted score table."""
    results = []
    for name, model in get_advanced_models().items():
        print(f"Evaluating {name}...")
        score = rmse_cv(model, X, y, cv=5)
        results.append({"model": name, "cv_rmse_log": score})

    return pd.DataFrame(results).sort_values("cv_rmse_log").reset_index(drop=True)


def fit_model_by_name(name: str, X: pd.DataFrame, y: pd.Series):
    """Fit one named advanced model on the full training data."""
    models = get_advanced_models()
    if name not in models:
        available = ", ".join(sorted(models))
        raise ValueError(f"Unknown model `{name}`. Available models: {available}")

    model = clone(models[name])
    model.fit(X, y)
    return model


def fit_models_by_names(names: Sequence[str], X: pd.DataFrame, y: pd.Series) -> dict[str, object]:
    """Fit several named advanced models on the full training data."""
    models = get_advanced_models()
    fitted_models: dict[str, object] = {}
    for name in names:
        if name not in models:
            print(f"Skipping unavailable model `{name}` during fitting.")
            continue
        print(f"Fitting {name}...")
        model = clone(models[name])
        model.fit(X, y)
        fitted_models[name] = model
    return fitted_models


def blend_predictions(
    fitted_models: Mapping[str, object],
    X_test: pd.DataFrame,
    weights: Mapping[str, float] | None = None,
) -> np.ndarray:
    """Return a normalized weighted blend of log-scale predictions.

    Args:
        fitted_models: Mapping from model name to an already fitted estimator.
        X_test: Feature matrix to predict.
        weights: Optional model weights. Missing or unavailable models are
            ignored, and remaining weights are normalized to sum to one.
    """
    if not fitted_models:
        raise ValueError("At least one fitted model is required for blending.")

    raw_weights = dict(weights or {name: 1.0 for name in fitted_models})
    active_weights = {
        name: float(weight)
        for name, weight in raw_weights.items()
        if name in fitted_models and float(weight) > 0
    }
    if not active_weights:
        raise ValueError("No positive blend weights match the fitted models.")

    weight_sum = sum(active_weights.values())
    normalized_weights = {name: weight / weight_sum for name, weight in active_weights.items()}

    predictions = np.zeros(len(X_test), dtype=float)
    for name, weight in normalized_weights.items():
        predictions += weight * np.asarray(fitted_models[name].predict(X_test), dtype=float)

    return predictions


def recommended_blend_weights(available_model_names: Sequence[str]) -> dict[str, float]:
    """Choose default or optional blend weights based on available estimators."""
    available = set(available_model_names)
    if {"xgboost", "lightgbm"}.issubset(available):
        return OPTIONAL_BLEND_WEIGHTS.copy()
    return DEFAULT_BLEND_WEIGHTS.copy()
