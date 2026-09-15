"""Yahoo Finance daily close (free, no auth). Same fetch for SPY and metals —
they differ only by ticker, so one function serves both feature streams."""

import logging

import pandas as pd

log = logging.getLogger(__name__)


def fetch_prices(
    symbol: str,
    period: str = "1y",
    value_col: str = "close",
    start: str | None = None,
) -> pd.DataFrame:
    """Daily close for any Yahoo ticker (SPY, GC=F gold, SI=F silver, ...).

    `value_col` names the price column so each stream is distinct in the DB
    (e.g. "spy_close" vs "metal_close") instead of a generic "close".

    `start` (ISO date) fetches from that day onward for incremental top-ups;
    otherwise `period` applies ("max" for a full backfill).
    """
    import yfinance as yf

    hist = yf.Ticker(symbol)
    if start is not None:
        log.info("yfinance fetch: %s (from %s)", symbol, start)
        df = hist.history(start=start).reset_index()
    else:
        log.info("yfinance fetch: %s (%s)", symbol, period)
        df = hist.history(period=period).reset_index()
    if df.empty:
        if start is not None:  # incremental: no new trading days -> not an error
            return pd.DataFrame(columns=["date", value_col])
        raise ValueError(f"no price data for ticker {symbol!r}")
    df.columns = [c.lower() for c in df.columns]
    df["date"] = pd.to_datetime(df["date"]).dt.date
    return df[["date", "close"]].rename(columns={"close": value_col})
