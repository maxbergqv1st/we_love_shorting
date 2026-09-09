"""Yahoo Finance daily close (free, no auth). Same fetch for SPY and metals —
they differ only by ticker, so one function serves both feature streams."""

import logging

import pandas as pd

log = logging.getLogger(__name__)


def fetch_prices(symbol: str, period: str = "1y") -> pd.DataFrame:
    """Daily close for any Yahoo ticker (SPY, GC=F gold, SI=F silver, ...)."""
    import yfinance as yf

    log.info("yfinance fetch: %s (%s)", symbol, period)
    df = yf.Ticker(symbol).history(period=period).reset_index()
    if df.empty:
        raise ValueError(f"no price data for ticker {symbol!r}")
    df.columns = [c.lower() for c in df.columns]
    df["date"] = pd.to_datetime(df["date"]).dt.date
    return df[["date", "close"]]
