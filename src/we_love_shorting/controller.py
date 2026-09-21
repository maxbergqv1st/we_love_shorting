"""Controller: wire the separate sources -> DB -> features -> ML -> results."""

import logging
from collections.abc import Callable
from dataclasses import dataclass

import pandas as pd
from sklearn.linear_model import LinearRegression

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
    fetches from each table's last stored date; otherwise a full ~10y / rolling
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
            src[stem] = lambda t=ticker, c=col: yahoo.fetch_prices(t, "10y", c)
    if incremental:
        src["tone"] = lambda: gdelt.fetch_tone(query, start=db.last_date("tone"))
    else:
        # 120 months to match the 10y price backfill; GDELT DOC only indexes
        # ~2017 on, so tone still floors there (the chart starts at that overlap).
        src["tone"] = lambda: gdelt.fetch_tone(query, timespan="120m")
    return src


def backfill(query: str = "recession") -> tuple[dict, dict]:
    """First fill: pull ~10 years of history for every stream into the DB. Tickers
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


def _forward_model(
    df: pd.DataFrame, feature_cols: list[str], target: str, horizon: int
) -> LinearRegression:
    """Fit today's features -> the forward target over `horizon` days, via a
    throwaway `_fwd` label column. Shared by run() and forecast_next()."""
    design = df.assign(_fwd=features.forward_target(df, target, horizon))
    return signal_model.train(design, feature_cols, "_fwd", persist=False)


def run(
    df: pd.DataFrame,
    feature_cols: list[str],
    target: str,
    horizon: int = 0,
) -> pd.DataFrame:
    """Train on the chosen features/target and add a `predicted_*` column.

    The model is handed straight to predict, so we skip persisting it — the UI
    calls this on every rerun and doesn't reload from disk. `horizon=0` is the
    contemporaneous (same-day) fit; `horizon>0` forecasts `horizon` trading days
    ahead from today's features and aligns each prediction to the date it is FOR
    (so the chart overlays past forecasts on the realised price).
    """
    display_col = features.display_column(target)
    if horizon == 0:
        model = signal_model.train(df, feature_cols, target, persist=False)
        df[f"predicted_{target}"] = signal_model.predict(df, model, feature_cols)
        if target.endswith("_ret"):  # rebuild a price line from the predicted change
            df[f"predicted_{display_col}"] = features.reconstruct_close(
                df[display_col], df[f"predicted_{target}"]
            )
        return df

    model = _forward_model(df, feature_cols, target, horizon)
    pred_fwd = signal_model.predict(df, model, feature_cols)  # forward return at t
    if target.endswith("_ret"):
        df[f"predicted_{target}"] = pred_fwd  # predicted return, for the honest scatter
        forecast = df[display_col] * (1 + pred_fwd)  # price predicted FOR t+horizon
    else:
        forecast = pred_fwd
    df[f"predicted_{display_col}"] = forecast.shift(horizon)  # align to realised date
    return df


def forecast_next(
    df: pd.DataFrame, feature_cols: list[str], target: str, horizon: int = 1
) -> dict[str, float | str]:
    """Forecast `horizon` trading days past the last row: train the forward-target
    model (today's features -> return over the next `horizon` days) on all cleaned
    history, then feed the last row's ACTUAL features. For a `_ret` target the
    reconstructed future close is included too.
    """
    clean = df.dropna(subset=[*feature_cols, target]).reset_index(drop=True)
    model = _forward_model(clean, feature_cols, target, horizon)
    latest = clean[feature_cols].iloc[[-1]]  # last actual features -> predict h ahead
    pred = float(model.predict(latest)[0])
    out: dict[str, float | str] = {
        "from_date": str(clean["date"].iloc[-1]),
        "horizon": horizon,
        target: pred,
    }
    if target.endswith("_ret"):
        close = features.display_column(target)
        out[close] = float(clean[close].iloc[-1]) * (1 + pred)
    return out


def backtest_date(
    df: pd.DataFrame,
    feature_cols: list[str],
    target: str,
    target_date: object,  # datetime.date
    horizon: int,
) -> dict[str, float | str | int]:
    """Point-in-time forecast of an ALREADY-PAST `target_date`: train ONLY on rows
    before it, predict it from the features known `horizon` trading days earlier,
    and return predicted vs actual so you can see directly whether it landed.
    `horizon=0` tests the same-day fit out-of-sample. No lookahead — appending
    later rows to `df` can't change the result. Raises if the date isn't a usable
    trading row for this pick, or there's too little history before it.
    """
    clean = evaluation.drop_market_closed(df)
    clean = clean.dropna(subset=[*feature_cols, target]).reset_index(drop=True)
    match = clean.index[clean["date"] == target_date]
    if len(match) == 0:
        raise ValueError(f"{target_date} är ingen användbar handelsdag för valet")
    i = int(match[0])
    if i - horizon <= 0:
        raise ValueError("för lite historik före det datumet för att prognosticera det")
    display_col = features.display_column(target)
    past = clean.iloc[:i]  # strictly before the target date -> no leakage
    if horizon == 0:
        model = signal_model.train(past, feature_cols, target, persist=False)
        anchor, base = i, i - 1  # same-day features; reconstruct off yesterday's close
    else:
        model = _forward_model(past, feature_cols, target, horizon)
        anchor = base = i - horizon  # features h days back; reconstruct off that close
    pred = float(model.predict(clean[feature_cols].iloc[[anchor]])[0])
    predicted = (
        float(clean[display_col].iloc[base]) * (1 + pred)
        if target.endswith("_ret")
        else pred
    )
    actual = float(clean[display_col].iloc[i])
    # naive baseline = "no change": carry the anchor's value forward unchanged, the
    # honest reference the model has to beat (see evaluation.naive_baseline).
    baseline = float(clean[display_col].iloc[base])
    return {
        "date": str(clean["date"].iloc[i]),
        "column": display_col,
        "predicted": predicted,
        "actual": actual,
        "error": predicted - actual,
        "baseline": baseline,
        "baseline_error": baseline - actual,
        "trained_rows": len(past),
        "trained_until": str(clean["date"].iloc[i - 1]),
    }


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
    horizon: int = 0,
) -> EvaluationResult:
    """Chronologically split `df`, train on the train split only, and compare
    the model's held-out test predictions against a naive baseline.

    Market-closed (carry-forward) rows are dropped before the split — see
    evaluation.drop_market_closed — since ffill duplicates `_close`/`_ret`
    across closed days while `tone` keeps changing, which would otherwise
    teach the model an artificial repeated relationship. This is separate
    from run(), which trains on every row for the live chart.

    `horizon>0` overwrites the target column with its forward label (the return
    over the next `horizon` days) and evaluates that forecast against the SAME
    baseline — so the metrics answer "does today's signal predict the next
    day/week/month better than the naive forecast?". Keeping the original column
    name means baseline_kind still picks `mean` for a `_ret` target.
    """
    clean = evaluation.drop_market_closed(df)
    # Drop rows this pick can't use (NaN tone/target before its coverage): a
    # returns-only model keeps the full price history, a tone pick trims to ~2017.
    clean = clean.dropna(subset=[*feature_cols, target]).reset_index(drop=True)
    if horizon:  # forecast: predict the target's value/return `horizon` days ahead
        clean[target] = features.forward_target(clean, target, horizon)
        clean = clean.dropna(subset=[target]).reset_index(drop=True)
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
