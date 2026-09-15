"""Controller: wire the separate sources -> DB -> features -> ML -> results."""

import logging

import pandas as pd

from . import db, features, signal_model
from .sources import gdelt, yahoo

log = logging.getLogger(__name__)


def refresh(query: str, spy_symbol: str, metal_symbol: str, oil_symbol: str) -> None:
    """Pull each source into its own table (each commits only on success)."""
    db.save("tone", gdelt.fetch_tone(query))
    db.save("spy", yahoo.fetch_prices(spy_symbol, value_col="spy_close"))
    db.save("metal", yahoo.fetch_prices(metal_symbol, value_col="metal_close"))
    db.save("oil", yahoo.fetch_prices(oil_symbol, value_col="oil_close"))


def get_data(
    query: str = "recession",
    spy_symbol: str = "SPY",
    metal_symbol: str = "GC=F",
    oil_symbol: str = "CL=F",
) -> pd.DataFrame:
    """Fetch -> store -> join into the wide feature table (no model).

    Falls back to cached DB data if a fetch fails (e.g. GDELT rate-limits).
    Kept separate from `run` so the UI can cache this once and re-train on
    different feature/target picks without re-hitting the data sources.
    """
    try:
        refresh(query, spy_symbol, metal_symbol, oil_symbol)
    except Exception as e:  # noqa: BLE001 - any fetch failure should fall back to cache
        try:
            db.load("tone")  # probe: raises if we have no cached data at all
        except Exception:  # noqa: BLE001 - no cache yet -> surface the original error
            raise RuntimeError(f"fetch failed and no cached data: {e}") from e
        log.warning("fetch failed (%s); using cached DB data", e)

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
