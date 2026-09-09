"""Controller: wire the separate sources -> DB -> features -> ML -> results."""

import logging

import pandas as pd

from . import db, features, signal_model
from .sources import gdelt, yahoo

log = logging.getLogger(__name__)


def refresh(query: str, spy_symbol: str, metal_symbol: str) -> None:
    """Pull each source into its own table (each commits only on success)."""
    db.save("tone", gdelt.fetch_tone(query))
    db.save("spy", yahoo.fetch_prices(spy_symbol, value_col="spy_close"))
    db.save("metal", yahoo.fetch_prices(metal_symbol, value_col="metal_close"))


def run(
    query: str = "recession", spy_symbol: str = "SPY", metal_symbol: str = "GC=F"
) -> pd.DataFrame:
    """Full flow: fetch -> store -> join -> train -> predict tone.

    Falls back to cached DB data if a fetch fails (e.g. GDELT rate-limits).
    """
    try:
        refresh(query, spy_symbol, metal_symbol)
    except Exception as e:  # noqa: BLE001 - any fetch failure should fall back to cache
        try:
            db.load("tone")  # probe: raises if we have no cached data at all
        except Exception:  # noqa: BLE001 - no cache yet -> surface the original error
            raise RuntimeError(f"fetch failed and no cached data: {e}") from e
        log.warning("fetch failed (%s); using cached DB data", e)

    df = features.build_features(db.load("tone"), db.load("spy"), db.load("metal"))
    model = signal_model.train(df)
    df["predicted_tone"] = signal_model.predict(df, model)
    return df
