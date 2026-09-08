---
name: sklearn
description: scikit-learn best practices — use when building, training, tuning, evaluating, or persisting ML models, Pipelines, or preprocessing in this repo. Source: https://scikit-learn.org/stable/
---

# scikit-learn best practices

## Prevent data leakage (the #1 rule)
- Wrap ALL preprocessing + estimator in a single `Pipeline`. Fit it on train only.
- Use `ColumnTransformer` for per-column preprocessing (scale numerics, encode categoricals).
- Never fit a scaler/encoder on the full dataset before splitting.

## Estimator API
- Contract: `fit(X, y)`, `transform`/`predict(X)`. Compose with `Pipeline`.
- Set `random_state` everywhere it exists for reproducibility.

## Model selection
- Split first, tune with cross-validation: `cross_val_score`, `GridSearchCV`,
  `RandomizedSearchCV` — the search fits the whole Pipeline per fold.
- Never tune on or peek at the held-out test set.

## Time series (relevant to shorting signals)
- Use `TimeSeriesSplit`, not `KFold`. Never shuffle time-ordered data
  (`shuffle=False` in splits). No future rows in training folds.

## Metrics
- Choose task-appropriate metrics: `classification_report`, ROC-AUC, PR-AUC.
- Trading data is usually imbalanced → prefer PR-AUC / F1 over accuracy;
  consider `class_weight="balanced"`.

## Persistence & reproducibility
- `joblib.dump`/`load` the whole fitted Pipeline. Record the sklearn version.
- Set seeds; log params + CV scores.
