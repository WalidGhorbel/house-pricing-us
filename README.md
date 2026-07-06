# House Prices: Advanced Regression 

Predicting residential sale prices from 79 explanatory features using a tuned,
stacked ensemble of linear and gradient-boosted models. Built for the Kaggle
[House Prices: Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques)
competition.

**Best cross-validated score:** RMSE **0.1075** (log SalePrice, 5×3 RepeatedKFold)

---

## Problem

Given 1,460 labelled homes (79 features each), predict the sale price of 1,459
unlabelled homes. Submissions are scored on **RMSE of log(SalePrice)**, so the
target is log-transformed throughout and predictions are inverted (`expm1`)
before submission.

The metric penalises *proportional* error equally across price ranges, which is
why every modelling decision here works in log space.

---


## Pipeline detail

### 1. Outlier removal
The dataset's author (De Cock, 2011) documents a handful of non-representative
sales. Two homes with `GrLivArea > 4000` sold for under \$300k (partial/family
sales) distort linear fits and are removed from the training set only. Verified
visually with a `GrLivArea` vs `SalePrice` scatter before removal.

### 2. Missing values — the "NA = absent" insight
The single most important data insight here: for most columns, a missing value
does **not** mean "unknown" — it means the feature is absent. Per the data
dictionary, `PoolQC = NA` means *no pool*, `GarageType = NA` means *no garage*,
and so on. These are filled with `"None"` (categorical) or `0` (numeric),
preserving real signal instead of destroying it with mean imputation.

A small number of columns are genuinely missing and imputed accordingly:
`LotFrontage` by **neighbourhood median** (nearby homes have similar frontage),
and a few one-off categoricals by mode. `Functional` defaults to `Typ` per the
data dictionary's explicit instruction.

### 3. Ordinal encoding
Quality ratings (`Ex > Gd > TA > Fa > Po`) carry a real order, so they are mapped
to integers (5→1) rather than one-hot encoded — preserving the ranking the model
can exploit. Applied to exterior/basement/kitchen/garage quality, basement
finish types, functionality, etc.

### 4. Feature engineering (mechanism-first)
Features were added only where a concrete real-estate mechanism justified them —
not by brute-force search. The standouts, confirmed by model feature-importance:

| Feature | Mechanism |
|---|---|
| `Qual_x_TotalSF` | Appraisers price *quality-adjusted* \$/sqft — quality and space compound, they don't just add. **Top feature by importance.** |
| `Qual_x_GrLivArea` | Same, for above-grade living area (priced differently from basement). |
| `OverallQual_2` | The luxury premium is convex — top quality pays off disproportionately. |
| `TotalQualScore` | A home's overall build quality is one impression scattered across 7 columns; summing reconstructs it. |
| `TotalSF`, `TotalBath` | Buyers think in *total* space and baths; the raw data splits these across many columns. |
| `EffectiveAge` | `YrSold − YearRemodAdd`: a remodelled old home functions like a new one. |

### 5. Encoding & scaling
Skewed numeric features (|skew| > 0.75) are `log1p`-transformed to help the
linear models. Remaining nominal categoricals are one-hot encoded, yielding
~275 features. Linear models use `RobustScaler` (resistant to the price outliers).

---

## Models

Six base models, each individually tuned with **Optuna** (40 trials, 5-fold CV):

| Model | Honest CV (RMSE) | Role |
|---|---|---|
| CatBoost | **0.1103** | Best single model; strong on categoricals |
| Lasso | 0.1113 | Regularised linear; automatic feature selection |
| ElasticNet | 0.1114 | Lasso/Ridge hybrid |
| XGBoost | 0.1130 | Gradient-boosted trees |
| Ridge | 0.1131 | Regularised linear |
| LightGBM | 0.1144 | Fast gradient boosting |

### Ensemble

A `StackingRegressor` combines all six via a `LassoCV` meta-model that learns the
optimal combination from out-of-fold predictions.

| Ensemble | Honest CV (RMSE) |
|---|---|
| **Stack (6 models)** | **0.1075** |
| Equal blend | 0.1077 |
| Best-3 blend | 0.1085 |



---
## Reproducing

```bash
# environment
pip install numpy pandas scipy scikit-learn xgboost lightgbm catboost optuna
```

1. Place `train.csv` and `test.csv` in the data folder.
2. Run the notebook top to bottom. Cells are ordered: setup → EDA →
   cleaning → feature engineering → Optuna tuning → ensemble → submission.
3. **Tuning takes ~1 hour** across the three boosting models. The best
   hyperparameters found are hardcoded in the "rebuild" cell so the pipeline
   can be reproduced in minutes without re-searching.
4. Output: `submission_tuned_stack.csv`, ready to upload to Kaggle.

> **Tip:** tuning is expensive — persist the tuned models/params to disk right
> after searching so a runtime disconnect can't cost you the work.

---



## Sources

Data: Dean De Cock, *"Ames, Iowa: Alternative to the Boston Housing Data Set"*,
Journal of Statistics Education (2011). Competition hosted by Kaggle.
