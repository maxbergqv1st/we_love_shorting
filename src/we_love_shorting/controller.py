"""Controller: wire the separate sources -> DB -> features -> ML -> results."""

import logging
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
    src = {}
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
