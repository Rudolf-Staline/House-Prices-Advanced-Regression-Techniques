# Advanced Report — Kaggle House Prices V2

## Objectif

La V2 vise à dépasser la baseline simple du projet Kaggle House Prices en passant d'un pipeline générique Ridge/RandomForest autour de `0.145` RMSE log CV à une base avancée visant `< 0.130`, puis extensible vers `< 0.120` après tuning, nettoyage d'outliers et éventuel stacking.

## Baseline V1

La V1 reste inchangée et reproductible via :

```bash
cd projects/kaggle-house-prices
python -m src.submit
```

Elle applique une imputation générique, un one-hot encoding global, un target `log1p(SalePrice)`, puis compare Ridge et RandomForest en validation croisée 5 folds.

## Améliorations V2

La V2 ajoute :

- un prétraitement métier des valeurs manquantes ;
- des features de surfaces, d'âges, d'indicateurs et d'interactions ;
- un encodage ordinal manuel des variables de qualité ;
- une transformation `log1p` des prédicteurs numériques asymétriques ;
- des modèles linéaires régularisés et modèles boostés ;
- un blending pondéré simple ;
- deux fichiers de soumission séparés : meilleur single model et blend.

## Prétraitement métier

Les colonnes catégorielles où `NaN` signifie l'absence d'un élément (`Alley`, variables de sous-sol, garage, piscine, clôture, etc.) sont imputées par `"None"`.

Les colonnes numériques où `NaN` signifie absence (`GarageYrBlt`, `GarageCars`, `GarageArea`, variables de sous-sol, `MasVnrArea`, etc.) sont imputées par `0`.

Les autres numériques sont imputées par la médiane et les autres catégorielles par le mode, avec fallback `"Missing"`.

## Feature engineering

Features ajoutées :

- `TotalSF`, `TotalFlrSF`, `TotalPorchSF`, `TotalBathrooms` ;
- `HouseAge`, `RemodAge`, `GarageAge` ;
- `HasBasement`, `HasGarage`, `HasFireplace`, `HasPool`, `HasPorch`, `WasRemodeled`, `IsNewHouse` ;
- `OverallQual_TotalSF`, `OverallQual_GrLivArea`, `OverallQual_GarageArea`, `OverallQual_TotalBathrooms`.

Les mappings ordinaux sont appliqués notamment à `ExterQual`, `BsmtQual`, `KitchenQual`, `FireplaceQu`, `GarageQual`, `PoolQC`, `BsmtExposure`, `BsmtFinType1/2`, `GarageFinish`, `LotShape`, `LandSlope` et `PavedDrive`.

## Modèles testés

Modèles obligatoires :

| Modèle | Rôle |
| --- | --- |
| Ridge | modèle linéaire régularisé L2 robuste |
| Lasso | sélection de variables via L1 |
| ElasticNet | compromis L1/L2 souvent performant sur ce dataset |
| GradientBoostingRegressor | boosting non linéaire stable |
| HistGradientBoostingRegressor | boosting histogramme rapide |

Modèles optionnels :

| Modèle | Condition |
| --- | --- |
| XGBoost | utilisé seulement si `xgboost` est installé |
| LightGBM | utilisé seulement si `lightgbm` est installé |

Ces dépendances ne sont pas obligatoires : si elles sont absentes, le script les ignore explicitement.

## Tableau CV RMSE log

À compléter après exécution avec les fichiers Kaggle locaux :

| Modèle | CV RMSE log |
| --- | ---: |
| ElasticNet | à compléter |
| Lasso | à compléter |
| Ridge | à compléter |
| GradientBoostingRegressor | à compléter |
| HistGradientBoostingRegressor | à compléter |
| XGBoost | optionnel |
| LightGBM | optionnel |

Commande :

```bash
cd projects/kaggle-house-prices
python -m src.submit_advanced
```

## Meilleur modèle single

À compléter après exécution locale. Le script trie le tableau CV et entraîne automatiquement le modèle avec le plus faible RMSE log.

## Blend utilisé

Blend par défaut si seuls les modèles obligatoires sont disponibles :

| Modèle | Poids initial |
| --- | ---: |
| ElasticNet | 0.25 |
| Lasso | 0.20 |
| Ridge | 0.10 |
| GradientBoostingRegressor | 0.25 |
| HistGradientBoostingRegressor | 0.20 |

Blend proposé si XGBoost et LightGBM sont disponibles :

| Modèle | Poids initial |
| --- | ---: |
| ElasticNet | 0.15 |
| Lasso | 0.10 |
| Ridge | 0.05 |
| GradientBoostingRegressor | 0.20 |
| HistGradientBoostingRegressor | 0.15 |
| XGBoost | 0.20 |
| LightGBM | 0.15 |

Les poids sont toujours normalisés sur les modèles effectivement entraînés.

## Fichiers de soumission générés

Le script V2 génère :

```text
projects/kaggle-house-prices/data/submissions/submission_advanced_best_single.csv
projects/kaggle-house-prices/data/submissions/submission_advanced_blend.csv
```

Ces fichiers sont ignorés par Git et ne doivent pas être commités.

## Score Kaggle public

à compléter après soumission

## Limites

- Pas encore de tuning exhaustif des hyperparamètres.
- Pas encore de suppression explicite des outliers influents.
- Blending simple, sans stacking out-of-fold.
- Pas encore d'explication des prédictions par SHAP ou permutation importance.
- Le score final dépendra des versions de librairies et de la présence éventuelle de XGBoost/LightGBM.

## Prochaines itérations

1. Tester un nettoyage contrôlé des outliers, notamment les très grandes surfaces avec prix atypique.
2. Ajouter une recherche d'hyperparamètres pour Ridge/Lasso/ElasticNet et boosting.
3. Construire un stacking out-of-fold avec méta-modèle Ridge/ElasticNet.
4. Ajouter des features par quartier (`Neighborhood`) et qualité.
5. Comparer les scores Kaggle public/private et renseigner le rapport.
