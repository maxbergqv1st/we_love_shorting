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

    window = {"start": start} if start is not None else {"period": period}
    log.info("yfinance fetch: %s (%s)", symbol, start or period)
    df = yf.Ticker(symbol).history(**window).reset_index()
    if df.empty:
        if start is not None:  # incremental: no new trading days -> not an error
            return pd.DataFrame(columns=["date", value_col])
        raise ValueError(f"no price data for ticker {symbol!r}")
    df.columns = [c.lower() for c in df.columns]
    df["date"] = pd.to_datetime(df["date"]).dt.date
    return df[["date", "close"]].rename(columns={"close": value_col})


def search_tickers(query: str, max_results: int = 8) -> list[dict[str, str]]:
    """Free-text ticker search ("adverty" -> Adverty AB + its symbols) via
    Yahoo's search endpoint. Returns [{"symbol", "name", "exchange"}, ...];
    an empty list when nothing matches (a bare symbol typed here can still be
    fetched directly with fetch_prices)."""
    import yfinance as yf

    log.info("yfinance search: %s", query)
    quotes = yf.Search(query, max_results=max_results).quotes
    return [
        {
            "symbol": q["symbol"],
            "name": (q.get("shortname") or q.get("longname") or q["symbol"]).strip(),
            "exchange": q.get("exchange", ""),
        }
        for q in quotes
        if q.get("symbol")
    ]


def fetch_intraday_price(symbol: str, interval: str = "1m") -> pd.Series:
    """Latest intraday price for a Yahoo ticker, for a live-preview panel only.

    Not persisted to the DB and not part of the daily-close pipeline above —
    yfinance intraday quotes are typically ~15 min delayed and only meant as
    a preliminary glance, not model input.
    """
    import yfinance as yf

    log.info("yfinance intraday fetch: %s (%s)", symbol, interval)
    hist = yf.Ticker(symbol).history(period="1d", interval=interval)
    if hist.empty:
        raise ValueError(f"no intraday price data for ticker {symbol!r}")
    return hist.iloc[-1]
