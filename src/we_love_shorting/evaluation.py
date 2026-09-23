"""Train/test evaluation: chronological split, naive baseline, regression
metrics, and residuals. Kept separate from signal_model.py (the model-fitting
layer) since these are orthogonal evaluation concerns, not part of fitting."""

import logging

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    mean_absolute_error,
    root_mean_squared_error,
)

log = logging.getLogger(__name__)


def drop_market_closed(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows where `market_closed` is True, before any train/test split.

    build_features' ffill carries both `_close` and `_ret` forward unchanged
    on a closed day, so a run of closed days (e.g. Fri+Sat+Sun) shares one
    identical `_ret` value across several rows while `tone` keeps changing
    daily. Training on those rows would teach the model a duplicated,
    artificial relationship instead of genuine day-over-day signal.

    `market_closed` tracks ONE reference calendar (the S&P 500 / US market),
    so this drop is a proxy that removes the dominant staleness case. A foreign
    stream (gold, OMX, ...) closed on a day the US traded still carries a ffill'd
    `_ret`/`_close` on a kept row. That's accepted feature noise, not test
    leakage — a per-stream staleness model would either drop far more rows or
    thread a mask through the whole pipeline, not worth it at PoC scale.

    Eval-only: the live chart (controller.run) intentionally keeps every row,
    closed-market included, so this must not run there.
    """
    closed = df["market_closed"]
    log.info("dropped %d market-closed rows before eval split", int(closed.sum()))
    return df[~closed].reset_index(drop=True)


def chronological_split(
    df: pd.DataFrame, test_frac: float = 0.2
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split `df` by date: the last `test_frac` rows are the test set, the
    rest are train.

    Chronological, not random: the streams are autocorrelated (ffill'd
    prices, day-over-day returns), so a random shuffle would leak nearby rows
    into both train and test and overstate accuracy. Sorting by date first
    and taking the tail keeps the test set a genuinely held-out future
    window.
    """
    ordered = df.sort_values("date").reset_index(drop=True)
    split_at = int(len(ordered) * (1 - test_frac))
    train = ordered.iloc[:split_at].reset_index(drop=True)
    test = ordered.iloc[split_at:].reset_index(drop=True)
    return train, test


def baseline_kind(target: str) -> str:
    """Which naive baseline `naive_baseline` uses for `target`.

    `"mean"` for a move column — `<stem>_ret`, or a `<target>_diff` computed
    by controller.prepare_target: moves are close to stationary/white noise,
    so "yesterday's move predicts today's" is a weak baseline — the historical
    mean is the standard naive forecast for a move series.
    `"persistence"` for anything else (price levels, tone): those are highly
    autocorrelated, so carrying the last actual value forward is the
    standard, much stronger naive benchmark for a level series.
    """
    return "mean" if target.endswith(("_ret", "_diff")) else "persistence"


def naive_baseline(train_target: pd.Series, test_target: pd.Series) -> pd.Series:
    """Naive forecast for `test_target` to compare the model against, chosen
    by `baseline_kind(train_target.name)` — see that function for the rule.

    The persistence baseline predicts each test row from the actual previous
    row's value; the first test row has no in-test predecessor, so it's
    seeded with the last train value.
    """
    if baseline_kind(train_target.name) == "mean":
        return pd.Series(train_target.mean(), index=test_target.index, name="baseline")
    shifted = test_target.shift(1)  # each row predicted by the previous actual
    shifted.iloc[0] = train_target.iloc[-1]  # first test row has no in-test predecessor
    return shifted.rename("baseline")


def regression_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float]:
    """RMSE and MAE for `y_pred` against the ground truth `y_true`."""
    return {
        "rmse": float(root_mean_squared_error(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
    }


def direction_threshold(train_returns: pd.Series, flat_frac: float = 0.25) -> float:
    """Dead-zone half-width for the 'flat' class: a return within
    ±(flat_frac × train std) counts as unchanged. Data-driven so it adapts to
    each stream's volatility instead of a hard-coded percentage.
    """
    # ponytail: flat_frac=0.25 is the one calibration knob; widen for a bigger
    # 'unchanged' bucket, 0.0 collapses to a pure up/down split.
    return float(flat_frac * train_returns.std())


def direction_labels(returns: pd.Series, threshold: float) -> pd.Series:
    """Bucket returns into 1 (up) / 0 (flat) / -1 (down) using the dead-zone."""
    labels = pd.Series(0, index=returns.index, name="direction")
    labels[returns > threshold] = 1
    labels[returns < -threshold] = -1
    return labels


def majority_baseline(train_labels: pd.Series, test_index: pd.Index) -> pd.Series:
    """Predict train's most frequent class for every test row — the naive
    benchmark a classifier must beat. Works for any number of classes (2 or 3).
    """
    majority = train_labels.mode().iloc[0]
    return pd.Series(majority, index=test_index, name="baseline")


def direction_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float]:
    """Accuracy of a direction prediction against the realised direction."""
    return {"accuracy": float(accuracy_score(y_true, y_pred))}


def residuals(y_true: pd.Series, y_pred: pd.Series) -> pd.Series:
    """Actual minus predicted, indexed like `y_true` so it stays plottable in
    the same (chronological, post-split) row order as the test set."""
    return pd.Series(
        np.asarray(y_true) - np.asarray(y_pred), index=y_true.index, name="residual"
    )
