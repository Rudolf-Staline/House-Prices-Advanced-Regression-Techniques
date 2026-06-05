"""Baseline model training utilities."""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def rmse_cv(model, X: pd.DataFrame, y: pd.Series, cv: int = 5) -> float:
    """Return mean cross-validated RMSE on the log-transformed target."""
    kfold = KFold(n_splits=cv, shuffle=True, random_state=42)
    scores = cross_val_score(
        model,
        X,
        y,
        scoring="neg_root_mean_squared_error",
        cv=kfold,
        n_jobs=-1,
    )
    return float(-scores.mean())


def _candidate_models() -> dict[str, object]:
    """Define simple, reliable baseline models."""
    return {
        "ridge": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("model", Ridge(alpha=10.0, random_state=42)),
            ]
        ),
        "random_forest": RandomForestRegressor(
            n_estimators=300,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        ),
    }


def train_baseline_models(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """Evaluate baseline models and return a score table sorted by RMSE."""
    results = []
    for name, model in _candidate_models().items():
        score = rmse_cv(model, X, y, cv=5)
        results.append({"model": name, "cv_rmse_log": score})

    return pd.DataFrame(results).sort_values("cv_rmse_log").reset_index(drop=True)


def fit_best_model(X: pd.DataFrame, y: pd.Series):
    """Fit the best baseline model on all training data and return it."""
    scores = train_baseline_models(X, y)
    best_name = str(scores.iloc[0]["model"])
    model = _candidate_models()[best_name]
    model.fit(X, y)
    return model


def save_model(model, path: str | Path) -> None:
    """Persist a fitted model with joblib."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
