# Kaggle House Prices — Advanced Regression Techniques

This project builds a clean and reproducible first baseline for the Kaggle competition **House Prices — Advanced Regression Techniques**. The goal is to predict `SalePrice` for homes in the Ames Housing dataset and generate a Kaggle-compatible submission file.

## Objective

Predict house sale prices from tabular features such as lot size, quality ratings, year built, neighborhood, and garage information. Kaggle evaluates submissions with RMSE on the logarithm of the predicted and actual prices, so this project trains models on `log1p(SalePrice)` and converts predictions back with `expm1`.

## Project structure

```text
projects/kaggle-house-prices/
├── README.md
├── notebooks/
│   └── 01_eda_baseline.ipynb
├── src/
│   ├── config.py
│   ├── data.py
│   ├── features.py
│   ├── model.py
│   └── submit.py
├── data/
│   ├── raw/
│   ├── processed/
│   └── submissions/
└── reports/
    └── baseline_report.md
```

## Installation

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows PowerShell, activate the environment with:

```powershell
.venv\Scripts\Activate.ps1
```

## Data setup

This repository does **not** store Kaggle datasets and does **not** hardcode Kaggle credentials.

Download the competition files from Kaggle:

<https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data>

Then place the files manually in:

```text
projects/kaggle-house-prices/data/raw/
```

Expected filenames:

```text
train.csv
test.csv
sample_submission.csv
```

If you use the Kaggle CLI, configure credentials outside the repository and run a command similar to:

```bash
kaggle competitions download -c house-prices-advanced-regression-techniques -p projects/kaggle-house-prices/data/raw
unzip projects/kaggle-house-prices/data/raw/house-prices-advanced-regression-techniques.zip -d projects/kaggle-house-prices/data/raw
```

Do not commit `kaggle.json`, downloaded ZIP files, or CSV datasets.

## How to run the notebook

From the repository root:

```bash
jupyter notebook projects/kaggle-house-prices/notebooks/01_eda_baseline.ipynb
```

The notebook contains EDA, preprocessing, baseline validation, and a submission-generation cell.

## How to generate a submission

From the project folder:

```bash
cd projects/kaggle-house-prices
python -m src.submit
```

The script writes:

```text
projects/kaggle-house-prices/data/submissions/submission_baseline.csv
```

The submission contains the required Kaggle columns:

- `Id`
- `SalePrice`

## ML approach

The first baseline intentionally prioritizes reliability and reproducibility:

1. Load `train.csv` and `test.csv` from `data/raw/`.
2. Split the target `SalePrice` from training features.
3. Concatenate train/test features for consistent one-hot encoding.
4. Impute missing numeric values with the median.
5. Impute missing categorical values with `Missing`.
6. One-hot encode categorical variables.
7. Train on `log1p(SalePrice)`.
8. Compare Ridge and RandomForest with 5-fold cross-validation using RMSE on the log target.
9. Fit the best baseline on the full training set.
10. Predict on the test set and apply `expm1`.


## Advanced pipeline

The V2 advanced pipeline keeps the baseline intact and adds a stronger, Kaggle-oriented workflow designed to improve the log-RMSE score:

- domain-aware missing-value handling where absent basement, garage, fireplace, pool, fence, alley, and veneer values are encoded explicitly;
- engineered area, bathroom, age, binary indicator, and quality-interaction features;
- manual ordinal encoding for quality/exposure/finish variables;
- `log1p` transformation of strongly right-skewed numeric predictors while avoiding binary and ordinal variables;
- regularized models (`Ridge`, `Lasso`, `ElasticNet`), boosted models (`GradientBoostingRegressor`, `HistGradientBoostingRegressor`), and optional `XGBoost`/`LightGBM` if installed;
- a simple normalized weighted blend in addition to the best single model.

From the project folder, generate the advanced submissions with:

```bash
cd projects/kaggle-house-prices
python -m src.submit_advanced
```

The script evaluates available models with 5-fold CV RMSE on the log target, prints a sorted score table, fits the best single model, fits the blend models, validates submissions, and writes:

```text
projects/kaggle-house-prices/data/submissions/submission_advanced_best_single.csv
projects/kaggle-house-prices/data/submissions/submission_advanced_blend.csv
```

Difference versus the baseline:

| Pipeline | Preprocessing | Models | Output |
| --- | --- | --- | --- |
| V1 baseline | generic median/`Missing` imputation + one-hot encoding | Ridge, RandomForest | `submission_baseline.csv` |
| V2 advanced | domain imputation + engineered features + ordinal encoding + skew correction | Ridge, Lasso, ElasticNet, boosting, optional XGBoost/LightGBM, blend | `submission_advanced_best_single.csv`, `submission_advanced_blend.csv` |

## Current limitations

### V1 baseline limitations

- Missing values are handled generically instead of using domain-specific meaning.
- Numeric skew is not corrected beyond the target transformation.
- Feature engineering is minimal.
- The model search is limited to Ridge and RandomForest with simple defaults.

### V2 advanced pipeline limitations

- Hyperparameters are still manually chosen defaults rather than the result of a systematic search.
- No explicit influential-outlier removal is applied yet.
- The blend is a fixed weighted average, not an out-of-fold stacking model.
- Optional XGBoost and LightGBM models run only when those packages are installed locally.
- No model interpretation is included yet.

## Next improvements

- Add total surface features such as total basement plus first/second floor area.
- Treat missing categorical values according to data dictionary semantics.
- Apply log transforms to skewed numeric features.
- Tune Ridge/Lasso/ElasticNet and tree-based models.
- Add simple stacking.
- Add permutation importance or SHAP analysis.
