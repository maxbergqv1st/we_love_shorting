"""Controller: wire data -> DB -> ML -> results."""

import logging

import pandas as pd

from . import db, signal_model, sources

log = logging.getLogger(__name__)


def refresh(query: str, symbol: str) -> None:
    """Pull fresh data into the database (each fetch commits only on success)."""
    db.save("tone", sources.fetch_tone(query))
    db.save("prices", sources.fetch_prices(symbol))


def run(query: str = "recession", symbol: str = "SPY") -> pd.DataFrame:
    """Full flow: fetch -> store -> train -> predict. Returns scored frame for the view.

    Falls back to cached DB data if the fetch fails (e.g. GDELT rate-limits).
    """
    try:
        refresh(query, symbol)
    except Exception as e:  # noqa: BLE001 - any fetch failure should fall back to cache
        try:
            db.load("tone")
            db.load("prices")
        except Exception:  # noqa: BLE001 - no cache yet -> surface the original error
            raise RuntimeError(f"fetch failed and no cached data: {e}") from e
        log.warning("fetch failed (%s); using cached DB data", e)

    df = signal_model.build_features(db.load("tone"), db.load("prices"))
    model = signal_model.train(df)
    df["short_prob"] = signal_model.predict(df, model)
    return df
