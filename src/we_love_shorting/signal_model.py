"""ML layer: regress news tone on the two daily prices. Low predicted tone =
bearish sentiment = a shorting cue. Swap LinearRegression to classify instead."""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from .features import FEATURES, TARGET

MODEL_PATH = Path("data/model.joblib")


def train(
    df: pd.DataFrame,
    features: list[str] = FEATURES,
    target: str = TARGET,
    persist: bool = True,
) -> LinearRegression:
    model = LinearRegression()  # OLS is scale-invariant, so no StandardScaler needed
    model.fit(df[features], df[target])
    if persist:  # skip the disk write on interactive re-fits (see controller.run)
        MODEL_PATH.parent.mkdir(exist_ok=True)
        joblib.dump(model, MODEL_PATH)
    return model


def predict(
    df: pd.DataFrame,
    model: LinearRegression | None = None,
    features: list[str] = FEATURES,
) -> pd.Series:
    if model is None:
        model = joblib.load(MODEL_PATH)
    return pd.Series(model.predict(df[features]), index=df.index, name="predicted")


def train_direction(x: pd.DataFrame, y: pd.Series) -> Pipeline:
    """Fit a scaled multinomial logistic classifier of the direction labels `y`
    (up / flat / down, from evaluation.direction_labels) on features `x`.

    For a return target the magnitude is ~white noise (regressing it collapses
    to the mean — a flat line), but the direction is the tractable question.
    Unlike OLS, logistic regression is not scale-invariant, so a StandardScaler
    is required: raw price levels (~1000s) would otherwise swamp returns
    (~0.01). Interactive-only, so it's never persisted.
    """
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    model.fit(x, y)
    return model


def predict_direction(x: pd.DataFrame, model: Pipeline) -> pd.Series:
    """Predicted direction class (1/0/-1) for each row, aligned to `x.index`."""
    return pd.Series(model.predict(x), index=x.index, name="predicted_dir")


def cluster_assets(returns: pd.DataFrame, n_groups: int = 3) -> pd.Series:
    """Group asset columns by co-movement. Distance = 1 − correlation (assets
    that move together are 'close'), so agglomerative clustering puts the
    precious metals in one group, oil in another, etc. Correlation is computed
    on returns (stationary) — clustering price *levels* would just group by era.
    Returns a group id per column (`n_groups` clamped to the column count).
    """
    corr = returns.corr()
    model = AgglomerativeClustering(
        n_clusters=min(n_groups, len(corr)), metric="precomputed", linkage="average"
    )
    labels = model.fit_predict(1 - corr.to_numpy())
    return pd.Series(labels, index=corr.columns, name="group")
