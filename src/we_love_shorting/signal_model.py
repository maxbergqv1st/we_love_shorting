"""ML layer: the fitted estimators (regression, direction classifier, asset
clustering). All interactive-only — models are refit per request and never
persisted, so there is no stale-model-on-disk state to reason about."""

import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

type Regressor = LinearRegression | RandomForestRegressor | Pipeline

# train()'s model knobs, in one place: everyone who forwards a hyperparameter
# dict into a fit (modes, the live panel) picks exactly these keys from it.
MODEL_KEYS = ("alpha", "model_kind", "n_estimators", "max_depth")


def train(
    df: pd.DataFrame,
    features: list[str],
    target: str,
    alpha: float = 0.0,
    model_kind: str = "linear",
    n_estimators: int = 100,
    max_depth: int = 6,
) -> Regressor:
    """Fit the price→target regressor.

    `model_kind="linear"`: `alpha` is the L2 penalty — 0 keeps plain OLS
    (scale-invariant, no scaler needed); alpha > 0 switches to Ridge, which
    penalises large coefficients (a smoother, less overfit fit) and is NOT
    scale-invariant, so it gets a StandardScaler in front.
    `model_kind="forest"`: a RandomForestRegressor (captures non-linear
    interactions trees can express but a linear model can't); `alpha` is
    ignored, `n_estimators`/`max_depth` bound size and overfitting.
    `random_state` pins the forest so a rerun is reproducible.
    """
    model: Regressor
    if model_kind == "forest":
        model = RandomForestRegressor(
            n_estimators=n_estimators, max_depth=max_depth, random_state=0, n_jobs=-1
        )
    elif alpha > 0:
        model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
    else:
        model = LinearRegression()
    model.fit(df[features], df[target])
    return model


def predict(df: pd.DataFrame, model: Regressor, features: list[str]) -> pd.Series:
    return pd.Series(model.predict(df[features]), index=df.index, name="predicted")


def weights(model: Regressor, features: list[str]) -> pd.Series:
    """What the fitted regressor leans on, per feature: coefficients for the
    linear family (Ridge's are on scaled features, so they ARE comparable across
    features; raw OLS coefficients carry each feature's unit) or impurity-based
    feature importances for the forest (always non-negative, sum to 1)."""
    est = model[-1] if isinstance(model, Pipeline) else model
    vals = getattr(est, "feature_importances_", None)
    if vals is None:
        vals = est.coef_
    return pd.Series(vals, index=features, name="weight")


def train_direction(x: pd.DataFrame, y: pd.Series, c: float = 1.0) -> Pipeline:
    """Fit a scaled multinomial logistic classifier of the direction labels `y`
    (up / flat / down, from evaluation.direction_labels) on features `x`. `c` is
    LogisticRegression's inverse regularisation strength: smaller = stronger
    penalty (the classifier's analogue of Ridge's alpha).

    For a return target the magnitude is ~white noise (regressing it collapses
    to the mean — a flat line), but the direction is the tractable question.
    Unlike OLS, logistic regression is not scale-invariant, so a StandardScaler
    is required: raw price levels (~1000s) would otherwise swamp returns
    (~0.01). Interactive-only, so it's never persisted.
    """
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, C=c))
    model.fit(x, y)
    return model


def predict_direction(x: pd.DataFrame, model: Pipeline) -> pd.Series:
    """Predicted direction class (1/0/-1) for each row, aligned to `x.index`."""
    return pd.Series(model.predict(x), index=x.index, name="predicted_dir")


def cluster_assets(
    returns: pd.DataFrame, n_groups: int = 3, linkage: str = "average"
) -> pd.Series:
    """Group asset columns by co-movement. Distance = 1 − correlation (assets
    that move together are 'close'), so agglomerative clustering puts the
    precious metals in one group, oil in another, etc. Correlation is computed
    on returns (stationary) — clustering price *levels* would just group by era.
    `linkage` picks how cluster-to-cluster distance is measured (average /
    complete / single; ward is unavailable with a precomputed distance).
    Returns a group id per column (`n_groups` clamped to the column count).
    """
    corr = returns.corr()
    model = AgglomerativeClustering(
        n_clusters=min(n_groups, len(corr)), metric="precomputed", linkage=linkage
    )
    labels = model.fit_predict(1 - corr.to_numpy())
    return pd.Series(labels, index=corr.columns, name="group")
