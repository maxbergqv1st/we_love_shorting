"""ML layer: the fitted estimators (regression, direction classifier, asset
clustering). All interactive-only — models are refit per request and never
persisted, so there is no stale-model-on-disk state to reason about."""

import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from .features import FEATURES, TARGET


def train(
    df: pd.DataFrame,
    features: list[str] = FEATURES,
    target: str = TARGET,
    alpha: float = 0.0,
) -> LinearRegression | Pipeline:
    """Fit the price→target regressor. `alpha` is the L2 penalty: 0 keeps plain
    OLS (scale-invariant, no scaler needed); alpha > 0 switches to Ridge, which
    penalises large coefficients (a smoother, less overfit fit) — Ridge is NOT
    scale-invariant, so it needs a StandardScaler in front.
    """
    model: LinearRegression | Pipeline
    if alpha > 0:
        model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
    else:
        model = LinearRegression()
    model.fit(df[features], df[target])
    return model


def predict(
    df: pd.DataFrame,
    model: LinearRegression | Pipeline,
    features: list[str] = FEATURES,
) -> pd.Series:
    return pd.Series(model.predict(df[features]), index=df.index, name="predicted")


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
