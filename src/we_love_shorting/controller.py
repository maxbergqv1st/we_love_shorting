"""Controller: wire the separate sources -> DB -> features -> ML -> results."""

import logging
from collections.abc import Callable
from dataclasses import dataclass

import pandas as pd

from . import db, evaluation, features, signal_model
from .sources import gdelt, yahoo

log = logging.getLogger(__name__)


def _fetch_all(specs: dict) -> tuple[dict, dict]:
    """Run each {table: fetch-callable}, upserting successes independently.

    Every source commits on its own, so one source's failure doesn't discard the
    others. Returns two dicts:
      retry  = {table: fetch-callable} that hit a rate-limit (HTTP 429) — worth
               retrying after a cooldown.
      errors = {table: message} that failed for any other reason (bad ticker,
               empty backfill, …) — retrying won't help, surface immediately.
    An incremental fetch that's already current returns an empty frame, which
    db.upsert no-ops on, so "nothing new" is neither a retry nor an error.
    """
    retry, errors = {}, {}
    for table, fetch in specs.items():
        try:
            db.upsert(table, fetch())
        except Exception as e:  # noqa: BLE001 - keep going; classify below
            log.warning("fetch for %s failed: %s", table, e)
            if getattr(e, "code", None) == 429:  # urllib.error.HTTPError rate-limit
                retry[table] = fetch
            else:
                errors[table] = str(e)
    return retry, errors


def _sources(query: str, *, incremental: bool) -> dict:
    """Every stream keyed by its DB table -> a fetch-callable. Tone (GDELT) sits
    beside the price streams (Yahoo); only its data source differs. `incremental`
    fetches from each table's last stored date; otherwise a full ~5y / rolling
    backfill.
    """
    src: dict[str, Callable[..., pd.DataFrame]] = {}
    for stem, ticker in features.TICKERS.items():
        col = f"{stem}_close"
        if incremental:
            src[stem] = lambda t=ticker, c=col, s=stem: yahoo.fetch_prices(
                t, value_col=c, start=db.last_date(s)
            )
        else:
            src[stem] = lambda t=ticker, c=col: yahoo.fetch_prices(t, "5y", c)
    if incremental:
        src["tone"] = lambda: gdelt.fetch_tone(query, start=db.last_date("tone"))
    else:
        src["tone"] = lambda: gdelt.fetch_tone(query, timespan="60m")
    return src


def backfill(query: str = "recession") -> tuple[dict, dict]:
    """First fill: pull ~5 years of history for every stream into the DB. Tickers
    are fixed in features.TICKERS (the UI no longer asks). GDELT tone only covers
    a rolling window (~2017 on), so its table may start later than the price
    history. Run once, then keep it current with update().
    Returns _fetch_all's (retry, errors) pair.
    """
    return _fetch_all(_sources(query, incremental=False))


def update(query: str = "recession") -> tuple[dict, dict]:
    """Top up each table from its newest stored date to today (no full refetch).
    An empty table (never backfilled) falls back to the default fetch window.
    Returns _fetch_all's (retry, errors) pair.
    """
    return _fetch_all(_sources(query, incremental=True))


def get_data() -> pd.DataFrame:
    """Read the stored data and join into the wide feature table (no fetch, no
    model). Fill the DB first via backfill()/update() — the DB is the source of
    truth. Kept separate from run() so the UI can cache this once and re-train on
    different feature/target picks without touching the data sources.
    """
    prices = {stem: db.load(stem) for stem in features.TICKERS}
    return features.build_features(db.load("tone"), prices)


def run(
    df: pd.DataFrame,
    feature_cols: list[str],
    target: str,
) -> pd.DataFrame:
    """Train on the chosen features/target and add a `predicted_{target}` column.

    The model is handed straight to predict, so we skip persisting it — the UI
    calls this on every rerun and doesn't reload from disk.
    """
    model = signal_model.train(df, feature_cols, target, persist=False)
    df[f"predicted_{target}"] = signal_model.predict(df, model, feature_cols)
    return df


def predict_live(
    df: pd.DataFrame,
    feature_cols: list[str],
    target: str,
    live_values: dict[str, float],
) -> float:
    """Train on stored history, then predict one point using live-fetched
    feature values in place of the corresponding stored ones.

    A feature missing from `live_values` (no live source, e.g. `tone`, or a
    ticker whose live fetch failed) falls back to the most recent stored
    value for that column — the same carry-forward idea as a market-closed
    day in features.build_features.
    """
    model = signal_model.train(df, feature_cols, target, persist=False)
    latest = df.iloc[-1]
    row = {c: live_values.get(c, latest[c]) for c in feature_cols}
    live_df = pd.DataFrame([row])
    return float(signal_model.predict(live_df, model, feature_cols).iloc[0])


@dataclass
class EvaluationResult:
    """Chronological train/test evaluation output: the test set carries the
    model's predictions and the naive baseline's side by side (plus each
    one's residual), and `metrics` holds both sides' RMSE/MAE.
    """

    train_df: pd.DataFrame
    test_df: pd.DataFrame
    baseline_kind: str
    metrics: dict[str, dict[str, float]]


def evaluate(
    df: pd.DataFrame,
    feature_cols: list[str],
    target: str,
    test_frac: float = 0.2,
) -> EvaluationResult:
    """Chronologically split `df`, train on the train split only, and compare
    the model's held-out test predictions against a naive baseline.

    Market-closed (carry-forward) rows are dropped before the split — see
    evaluation.drop_market_closed — since ffill duplicates `_close`/`_ret`
    across closed days while `tone` keeps changing, which would otherwise
    teach the model an artificial repeated relationship. This is separate
    from run(), which trains on every row for the live chart.
    """
    clean = evaluation.drop_market_closed(df)
    train_df, test_df = evaluation.chronological_split(clean, test_frac)

    model = signal_model.train(train_df, feature_cols, target, persist=False)
    predicted = signal_model.predict(test_df, model, feature_cols)
    baseline = evaluation.naive_baseline(train_df[target], test_df[target])

    test_df[f"predicted_{target}"] = predicted
    test_df[f"baseline_{target}"] = baseline
    test_df["residual"] = evaluation.residuals(test_df[target], predicted)
    test_df["baseline_residual"] = evaluation.residuals(test_df[target], baseline)

    metrics = {
        "model": evaluation.regression_metrics(test_df[target], predicted),
        "baseline": evaluation.regression_metrics(test_df[target], baseline),
    }
    return EvaluationResult(
        train_df, test_df, evaluation.baseline_kind(target), metrics
    )


@dataclass
class DirectionResult:
    """Direction-classification evaluation: test_df carries the realised and
    predicted direction classes (1/0/-1), `metrics` holds the model's and the
    majority baseline's accuracy, and `threshold` is the dead-zone half-width
    that split flat from up/down.
    """

    train_df: pd.DataFrame
    test_df: pd.DataFrame
    metrics: dict[str, dict[str, float]]
    threshold: float


def evaluate_direction(
    df: pd.DataFrame,
    feature_cols: list[str],
    target: str,
    test_frac: float = 0.2,
) -> DirectionResult:
    """Like evaluate(), but classifies the *direction* of `target` (up / flat /
    down) instead of regressing its value — the tractable question for a
    near-white-noise return series (see signal_model.train_direction). Model
    accuracy is compared against always guessing train's majority direction.

    The flat-class dead-zone is fitted on the train returns only and reused to
    label the test actuals, so the threshold never sees held-out data.
    """
    clean = evaluation.drop_market_closed(df)
    train_df, test_df = evaluation.chronological_split(clean, test_frac)

    threshold = evaluation.direction_threshold(train_df[target])
    y_train = evaluation.direction_labels(train_df[target], threshold)
    y_test = evaluation.direction_labels(test_df[target], threshold)

    model = signal_model.train_direction(train_df[feature_cols], y_train)
    predicted = signal_model.predict_direction(test_df[feature_cols], model)
    baseline = evaluation.majority_baseline(y_train, y_test.index)

    test_df["actual_dir"] = y_test.to_numpy()
    test_df["predicted_dir"] = predicted.to_numpy()

    metrics = {
        "model": evaluation.direction_metrics(y_test, predicted),
        "baseline": evaluation.direction_metrics(y_test, baseline),
    }
    return DirectionResult(train_df, test_df, metrics, threshold)


def run_direction(
    df: pd.DataFrame,
    feature_cols: list[str],
    target: str,
) -> pd.DataFrame:
    """In-sample direction fit for the live chart: classify every row's
    up/flat/down and add `actual_dir`/`predicted_dir`, so a return target can
    show a followable direction view instead of a flat regression line. Trains
    on all rows (no split) like run(), so it's a fit-quality view, not held-out.
    """
    threshold = evaluation.direction_threshold(df[target])
    y = evaluation.direction_labels(df[target], threshold)
    model = signal_model.train_direction(df[feature_cols], y)
    df["actual_dir"] = y.to_numpy()
    df["predicted_dir"] = signal_model.predict_direction(
        df[feature_cols], model
    ).to_numpy()
    return df


@dataclass
class AssetGroupResult:
    """Asset-similarity grouping: which streams move together. `corr` is the
    return-correlation matrix reordered so grouped assets are adjacent (ready to
    render as a heatmap), and `groups` maps each `<stem>_ret` column to its
    group id, in the same order.
    """

    corr: pd.DataFrame
    groups: pd.Series


def group_assets(
    df: pd.DataFrame,
    feature_cols: list[str],
    n_groups: int = 3,
) -> AssetGroupResult:
    """Group the selected streams by how their daily returns co-move. Uses each
    selected stream's `_ret` column (returns are stationary — the honest basis
    for correlation, unlike trending price levels). Needs ≥2 streams.
    """
    stems = dict.fromkeys(features.stream_of(c) for c in feature_cols)  # order-stable
    ret_cols = [f"{s}_ret" for s in stems if f"{s}_ret" in df.columns]
    if len(ret_cols) < 2:
        raise ValueError("Välj minst två streams med avkastning (_ret) att gruppera.")

    returns = df[ret_cols]
    groups = signal_model.cluster_assets(returns, n_groups)
    order = groups.sort_values().index
    return AssetGroupResult(returns.corr().loc[order, order], groups.loc[order])


if __name__ == "__main__":  # one-time backfill: python -m we_love_shorting.controller
    import sys

    logging.basicConfig(level=logging.INFO)
    retry, errors = backfill()
    if (
        retry or errors
    ):  # partial backfill -> non-zero exit so it isn't mistaken for done
        for table, msg in {
            **{t: "rate-limited (429)" for t in retry},
            **errors,
        }.items():
            log.error("backfill incomplete for %s: %s", table, msg)
        sys.exit(1)
