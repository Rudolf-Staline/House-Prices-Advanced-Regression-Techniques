# Baseline Report — Kaggle House Prices

## Objectif

Construire une première baseline reproductible pour prédire `SalePrice` sur le dataset Ames Housing du défi Kaggle **House Prices — Advanced Regression Techniques**.

## Données utilisées

Les fichiers attendus sont les fichiers officiels Kaggle placés localement dans `projects/kaggle-house-prices/data/raw/` :

- `train.csv`
- `test.csv`
- `sample_submission.csv`

Les données ne sont pas versionnées dans Git.

## Prétraitement

Le prétraitement est volontairement simple et robuste :

- séparation de la cible `SalePrice` ;
- conservation de `Id` uniquement pour le fichier de soumission, pas comme variable modèle ;
- transformation de la cible avec `log1p(SalePrice)` ;
- concaténation temporaire de train/test pour garantir un encodage cohérent ;
- imputation des variables numériques par la médiane ;
- imputation des variables catégorielles avec la modalité explicite `Missing` ;
- encodage one-hot des variables catégorielles ;
- séparation finale en matrices train/test alignées.

## Modèles testés

Deux modèles de baseline sont prévus :

- Ridge Regression avec standardisation ;
- RandomForestRegressor comme baseline non linéaire optionnelle.

## Score validation croisée

Le score est calculé avec une validation croisée 5-fold et la métrique `neg_root_mean_squared_error` sur la cible logarithmique.

Scores obtenus localement après exécution du pipeline sur les fichiers Kaggle officiels (`train.csv` et `test.csv`) :

| Modèle | CV RMSE log 5-fold |
| --- | ---: |
| RandomForestRegressor | 0.144940 |
| Ridge Regression | 0.149150 |

Commande utilisée :

```bash
cd projects/kaggle-house-prices
python -m src.submit
```

Score public Kaggle : non renseigné, car aucune soumission Kaggle n'a été effectuée dans cette session.

## Modèle retenu

Le script compare les scores de validation croisée et entraîne automatiquement le modèle ayant le plus faible RMSE moyen sur `log1p(SalePrice)`. Avec les résultats ci-dessus, le modèle retenu est `RandomForestRegressor`.

## Fichier de soumission généré

Le fichier généré est :

```text
projects/kaggle-house-prices/data/submissions/submission_baseline.csv
```

Il respecte le format Kaggle attendu :

- `Id`
- `SalePrice`

## Limites

- Gestion générique des valeurs manquantes, sans exploitation complète de la signification métier.
- Peu de feature engineering.
- Pas de transformation des variables numériques asymétriques.
- Hyperparamètres non optimisés.
- Pas d'analyse d'importance des variables.

## Prochaines itérations

- Feature engineering sur surface totale.
- Meilleure gestion des valeurs manquantes selon signification métier.
- Log-transform sur features asymétriques.
- Modèles régularisés avancés.
- Stacking simple.
- Analyse SHAP ou permutation importance.
