"""Controller: wire the separate sources -> DB -> features -> ML -> results."""

import logging

import pandas as pd

from . import db, features, signal_model
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


def backfill(
    query: str = "recession",
    spy_symbol: str = "SPY",
    metal_symbol: str = "GC=F",
    oil_symbol: str = "CL=F",
) -> tuple[dict, dict]:
    """First fill: pull ~5 years of history for each source into the DB.

    GDELT tone only covers a rolling window (~2017 on), so the tone table may
    start later than the 5-year price history. Run once, then keep it current
    with update(). Prices fetch first so a GDELT hiccup still leaves them saved.
    Returns _fetch_all's (retry, errors) pair.
    """
    return _fetch_all(
        {
            "spy": lambda: yahoo.fetch_prices(spy_symbol, "5y", "spy_close"),
            "metal": lambda: yahoo.fetch_prices(metal_symbol, "5y", "metal_close"),
            "oil": lambda: yahoo.fetch_prices(oil_symbol, "5y", "oil_close"),
            "tone": lambda: gdelt.fetch_tone(query, timespan="60m"),
        }
    )


def update(
    query: str = "recession",
    spy_symbol: str = "SPY",
    metal_symbol: str = "GC=F",
    oil_symbol: str = "CL=F",
) -> tuple[dict, dict]:
    """Top up each table from its newest stored date to today (no full refetch).

    An empty table (never backfilled) falls back to the default fetch window.
    Returns _fetch_all's (retry, errors) pair.
    """
    return _fetch_all(
        {
            "spy": lambda: yahoo.fetch_prices(
                spy_symbol, value_col="spy_close", start=db.last_date("spy")
            ),
            "metal": lambda: yahoo.fetch_prices(
                metal_symbol, value_col="metal_close", start=db.last_date("metal")
            ),
            "oil": lambda: yahoo.fetch_prices(
                oil_symbol, value_col="oil_close", start=db.last_date("oil")
            ),
            "tone": lambda: gdelt.fetch_tone(query, start=db.last_date("tone")),
        }
    )


def get_data() -> pd.DataFrame:
    """Read the stored data and join into the wide feature table (no fetch, no
    model). Fill the DB first via backfill()/update() — the DB is the source of
    truth. Kept separate from run() so the UI can cache this once and re-train on
    different feature/target picks without touching the data sources.
    """
    return features.build_features(
        db.load("tone"), db.load("spy"), db.load("metal"), db.load("oil")
    )


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
