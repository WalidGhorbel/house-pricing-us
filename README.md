# House Prices — Advanced Regression (Ames, Iowa)

Predicting residential sale prices from 79 explanatory features using a tuned,
stacked ensemble of linear and gradient-boosted models. Built for the Kaggle
[House Prices: Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques)
competition.

**Best cross-validated score:** RMSE **0.1075** (log SalePrice, 5×3 RepeatedKFold)
**Public leaderboard:** _`<add your score after submitting>`_

---

## Problem

Given 1,460 labelled homes (79 features each), predict the sale price of 1,459
unlabelled homes. Submissions are scored on **RMSE of log(SalePrice)**, so the
target is log-transformed throughout and predictions are inverted (`expm1`)
before submission.

The metric penalises *proportional* error equally across price ranges, which is
why every modelling decision here works in log space.

---

## Approach at a glance

```
raw CSV
  └─ outlier removal (2 documented bad sales)
      └─ missing-value handling ("NA = absent", not "missing")
          └─ ordinal encoding (quality scales → ordered integers)
              └─ feature engineering (mechanism-justified)
                  └─ skew correction + one-hot encoding
                      └─ 6 tuned base models (Optuna)
                          └─ stacked ensemble (meta-model)
                              └─ submission
```

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

The stack wins, and beats every single model — the base learners make partly
*uncorrelated* errors (linear vs tree vs foundation), so combining them cancels
error.

---

## Validation methodology

Every comparison uses **RepeatedKFold (5 folds × 3 repeats)** rather than a single
split, so scores are averaged over 15 fits and are not an artifact of one lucky
partition.

**A note on honesty:** cross-validation is an *estimate*, and on this competition
it runs optimistic relative to the leaderboard (CV reuses the same homes). Model
selection is done on CV; the leaderboard is treated as the final judge. Several
tempting techniques (weight-optimised blending, target encoding) were tested and
**rejected** because they improved CV without improving held-out performance —
classic overfitting, caught by honest validation.

---

## Results & diagnostics

- **Predicted vs actual (out-of-fold):** tight clustering along the diagonal;
  larger errors only at the price extremes (cheap and luxury homes), where
  training examples are sparse.
- **Residuals:** random scatter around zero — no systematic bias.
- **Feature importance:** engineered features (`Qual_x_TotalSF`, `OverallQual_2`,
  `TotalQualScore`) rank among the most important, validating the
  mechanism-first engineering.

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

## Project structure

```
house-prices/
├── README.md
├── house_prices_pipeline.ipynb   # full notebook: EDA → model → submission
├── data/
│   ├── train.csv
│   ├── test.csv
│   └── data_description.txt
└── submissions/
    └── submission_tuned_stack.csv
```

---

## Key takeaways

- **Data understanding beat model complexity.** The "NA = absent" insight and
  mechanism-justified features mattered more than any single algorithm.
- **Honest validation is the real skill.** Every model converged near the same
  score; the discipline was in *rejecting* improvements that only looked good on
  CV, and in refusing data-leakage shortcuts that inflate leaderboard scores on
  this particular competition.
- **The ensemble earns its keep** by combining uncorrelated error patterns, not
  by adding complexity for its own sake.

---

## Acknowledgements

Data: Dean De Cock, *"Ames, Iowa: Alternative to the Boston Housing Data Set"*,
Journal of Statistics Education (2011). Competition hosted by Kaggle.
