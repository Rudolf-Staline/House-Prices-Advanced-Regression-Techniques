
````md
# Kaggle House Prices — Advanced Regression Techniques

This project builds a clean, reproducible machine learning pipeline for the Kaggle competition **House Prices — Advanced Regression Techniques**.

The goal is to predict `SalePrice` for homes in the Ames Housing dataset using tabular regression methods, then progressively improve from a simple baseline to a stronger feature-engineered and blended model.

The project is now closed with a final public Kaggle score of **0.11840**.

---

## Final results

| Submission | Description | Public Kaggle score |
| --- | --- | ---: |
| `submission_baseline.csv` | V1 baseline: Ridge / RandomForest with generic preprocessing | 0.14389 |
| `submission_advanced_best_single.csv` | V2 best single model: Lasso | 0.12534 |
| `submission_advanced_blend.csv` | V2 blended model | **0.11840** |

The final blended V2 model improves the public Kaggle score from **0.14389** to **0.11840**, which represents an absolute gain of **0.02549** and a relative improvement of approximately **17.7%**.

---

## Objective

Predict house sale prices from tabular features such as lot size, quality ratings, year built, neighborhood, basement, garage, and living area information.

Kaggle evaluates submissions with RMSE on the logarithm of predicted and actual prices. This project therefore trains models on:

```text
log1p(SalePrice)
````

and converts final predictions back with:

```text
expm1(prediction)
```

---

## Project structure

```text
projects/kaggle-house-prices/
├── README.md
├── notebooks/
│   ├── 01_eda_baseline.ipynb
│   └── 02_advanced_pipeline.ipynb
├── src/
│   ├── config.py
│   ├── data.py
│   ├── features.py
│   ├── model.py
│   ├── submit.py
│   ├── advanced_features.py
│   ├── advanced_model.py
│   └── submit_advanced.py
├── data/
│   ├── raw/
│   ├── processed/
│   └── submissions/
└── reports/
    ├── baseline_report.md
    └── advanced_report.md
```

The `data/` directory is intentionally not versioned. Kaggle datasets, generated submissions, ZIP files, credentials, and model artifacts must remain local.

---

## Installation

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

---

## Data setup

This repository does **not** store Kaggle datasets and does **not** hardcode Kaggle credentials.

Download the competition files from Kaggle:

[https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data)

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

If using the Kaggle CLI, configure credentials outside the repository and run:

```bash
kaggle competitions download \
  -c house-prices-advanced-regression-techniques \
  -p projects/kaggle-house-prices/data/raw

unzip projects/kaggle-house-prices/data/raw/house-prices-advanced-regression-techniques.zip \
  -d projects/kaggle-house-prices/data/raw
```

Do not commit:

```text
kaggle.json
train.csv
test.csv
sample_submission.csv
*.zip
generated submission files
model artifacts
```

---

## V1 baseline pipeline

The first pipeline prioritizes reliability and reproducibility.

### Approach

1. Load `train.csv` and `test.csv` from `data/raw/`.
2. Split the target `SalePrice`.
3. Remove `Id` from model features.
4. Concatenate train/test features for consistent one-hot encoding.
5. Impute missing numeric values with the median.
6. Impute missing categorical values with `Missing`.
7. One-hot encode categorical variables.
8. Train on `log1p(SalePrice)`.
9. Compare Ridge and RandomForest with 5-fold cross-validation.
10. Generate a Kaggle-compatible submission.

### Run the baseline

From the project folder:

```bash
cd projects/kaggle-house-prices
python -m src.submit
```

The script writes:

```text
projects/kaggle-house-prices/data/submissions/submission_baseline.csv
```

### Baseline result

| Model                 | CV RMSE log |
| --------------------- | ----------: |
| RandomForestRegressor |    0.144940 |
| Ridge Regression      |    0.149150 |

Public Kaggle score:

```text
0.14389
```

---

## V2 advanced pipeline

The V2 pipeline keeps the baseline intact and adds a stronger Kaggle-oriented workflow.

### Main improvements

* domain-aware missing-value handling;
* explicit encoding of missing basement, garage, fireplace, pool, fence, alley, and veneer information;
* engineered area, bathroom, age, binary indicator, and interaction features;
* manual ordinal encoding for quality, exposure, slope, finish, and drive variables;
* `log1p` transformation of strongly right-skewed numeric predictors;
* regularized linear models: `Ridge`, `Lasso`, `ElasticNet`;
* boosted models: `GradientBoostingRegressor`, `HistGradientBoostingRegressor`;
* optional `XGBoost` and `LightGBM` if installed locally;
* simple weighted blending of multiple models.

### Feature engineering

The V2 pipeline adds features such as:

```text
TotalSF
TotalFlrSF
TotalPorchSF
TotalBathrooms
HouseAge
RemodAge
GarageAge
HasBasement
HasGarage
HasFireplace
HasPool
HasPorch
WasRemodeled
IsNewHouse
OverallQual_TotalSF
OverallQual_GrLivArea
OverallQual_GarageArea
OverallQual_TotalBathrooms
```

### Run the advanced pipeline

From the project folder:

```bash
cd projects/kaggle-house-prices
python -m src.submit_advanced
```

The script evaluates available models, prints a sorted cross-validation table, trains the best single model, trains the blend, validates both submissions, and writes:

```text
projects/kaggle-house-prices/data/submissions/submission_advanced_best_single.csv
projects/kaggle-house-prices/data/submissions/submission_advanced_blend.csv
```

### Advanced cross-validation results

| Model                         |  CV RMSE log |
| ----------------------------- | -----------: |
| Lasso                         | **0.126678** |
| ElasticNet                    |     0.126809 |
| GradientBoostingRegressor     |     0.128875 |
| Ridge                         |     0.130698 |
| HistGradientBoostingRegressor |     0.133612 |

XGBoost and LightGBM were not installed in the local execution environment, so they were skipped.

### Advanced Kaggle results

| Submission                                             | Public Kaggle score |
| ------------------------------------------------------ | ------------------: |
| V2 best single — `submission_advanced_best_single.csv` |             0.12534 |
| V2 blend — `submission_advanced_blend.csv`             |         **0.11840** |

The final selected model is the **V2 blended model**.

---

## Difference between V1 and V2

| Pipeline       | Preprocessing                                                                | Models                                                           | Public score |
| -------------- | ---------------------------------------------------------------------------- | ---------------------------------------------------------------- | -----------: |
| V1 baseline    | generic median/`Missing` imputation + one-hot encoding                       | Ridge, RandomForest                                              |      0.14389 |
| V2 best single | domain imputation + engineered features + ordinal encoding + skew correction | Lasso                                                            |      0.12534 |
| V2 blend       | same V2 features + weighted model blend                                      | ElasticNet, Lasso, Ridge, GradientBoosting, HistGradientBoosting |  **0.11840** |

---

## Notebooks

### `01_eda_baseline.ipynb`

Contains:

* first data inspection;
* target distribution analysis;
* missing-value overview;
* baseline preprocessing;
* Ridge / RandomForest validation;
* baseline submission generation.

### `02_advanced_pipeline.ipynb`

Contains:

* V2 feature engineering walkthrough;
* skewed-feature transformation;
* advanced model comparison;
* best model selection;
* blending;
* V1/V2 comparison.

---

## Reports

### `baseline_report.md`

Documents the V1 baseline pipeline, preprocessing choices, validation scores, generated submission file, and limitations.

### `advanced_report.md`

Documents the V2 pipeline, engineered features, model comparison, blend weights, local CV results, public Kaggle scores, and final closure of the project.

---

## Current limitations

The project is intentionally closed at V2. Remaining improvements are documented as future work rather than implemented in this version.

### V1 baseline limitations

* Missing values are handled generically.
* Numeric skew is not corrected beyond the target transformation.
* Feature engineering is minimal.
* Model search is limited to Ridge and RandomForest with simple defaults.

### V2 advanced pipeline limitations

* Hyperparameters are manually chosen defaults rather than the result of systematic tuning.
* No explicit influential-outlier removal is applied.
* The blend is a fixed weighted average, not an out-of-fold stacking model.
* XGBoost and LightGBM are optional and were not used in the reported local run.
* No SHAP or permutation-importance interpretation is included.

---

## Future work

Potential next steps, deliberately left outside the closed V2 version:

* tune Ridge, Lasso, ElasticNet and boosted models more systematically;
* test controlled removal of influential outliers, especially large-area low-price homes;
* add XGBoost and LightGBM to the model zoo;
* build an out-of-fold stacking pipeline;
* add neighborhood-level features and interactions;
* add SHAP or permutation-importance analysis;
* compare public and private leaderboard behavior.

---

## Closure note

This project is closed as a portfolio-ready machine learning case study.

It demonstrates the full workflow of a tabular regression project:

1. starting from a clean baseline;
2. validating the baseline locally and on Kaggle;
3. improving preprocessing with domain knowledge;
4. adding meaningful engineered features;
5. comparing multiple models;
6. using model blending;
7. documenting the progression from V1 to V2.

Final public Kaggle score:

```text
0.11840
```

```
```
