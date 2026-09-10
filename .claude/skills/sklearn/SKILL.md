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

## Clustering (unsupervised)
- Scale first — distance-based methods need it. Put the scaler + clusterer in a
  `Pipeline` (fit on train only, same leakage rule).
- Pick by data shape, not habit:
  - `KMeans` / `MiniBatchKMeans` — convex blobs, large n; must set `n_clusters`,
    `init="k-means++"`, `random_state`. MiniBatch when it's too slow.
  - `DBSCAN` / `HDBSCAN` — non-convex, uneven density, labels outliers `-1`.
    DBSCAN's `eps` is the knob (k-distance "knee" plot); HDBSCAN self-tunes via
    `min_cluster_size` and needs no `eps`.
  - `AgglomerativeClustering` — hierarchical / connectivity constraints;
    `linkage="ward"` for regular sizes.
  - `GaussianMixture` — soft/probabilistic assignments, choose k via BIC/AIC.
- Choosing k: sweep and score, don't eyeball. Silhouette or Calinski-Harabasz
  over `range(2, N)`.
- Evaluation:
  - No labels: `silhouette_score` (‑1..1, O(n²)), `calinski_harabasz_score`
    (fast, convex-biased), `davies_bouldin_score` (lower better).
  - With labels: `adjusted_rand_score` / `adjusted_mutual_info_score` — both
    chance-corrected; prefer these over raw NMI.
- Shorting context: cluster on scaled features/regimes, not raw prices; on time
  series still respect ordering — don't leak future rows into fitted transforms.

## Persistence & reproducibility
- `joblib.dump`/`load` the whole fitted Pipeline. Record the sklearn version.
- Set seeds; log params + CV scores.
