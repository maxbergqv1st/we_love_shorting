"""Model / data layer: fetch GDELT tone + prices over HTTP (stdlib only)."""

import json
import logging
import time
import urllib.error
import urllib.parse
import urllib.request

import pandas as pd

log = logging.getLogger(__name__)

GDELT_DOC = "https://api.gdeltproject.org/api/v2/doc/doc"


def _get_json(url: str, retries: int = 4) -> dict:
    """GET JSON with backoff on 429 — GDELT rate-limits per IP."""
    req = urllib.request.Request(url, headers={"User-Agent": "we_love_shorting/0.1"})
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req) as r:  # noqa: S310 - fixed https host
                return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code != 429 or attempt == retries - 1:
                raise
            wait = 2**attempt
            log.warning("GDELT 429, retrying in %ss", wait)
            time.sleep(wait)
    raise RuntimeError("unreachable")


def fetch_tone(query: str, timespan: str = "12m") -> pd.DataFrame:
    """Daily average news tone for a query term (GDELT DOC 2.0 timelinetone)."""
    params = urllib.parse.urlencode(
        {"query": query, "mode": "timelinetone", "timespan": timespan, "format": "json"}
    )
    url = f"{GDELT_DOC}?{params}"
    log.info("GDELT fetch: %s", url)
    data = _get_json(url)
    rows = data["timeline"][0]["data"]
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    return df.rename(columns={"value": "tone"})[["date", "tone"]]


def fetch_prices(symbol: str = "SPY", period: str = "1y") -> pd.DataFrame:
    """Daily close from Yahoo Finance (free, no auth)."""
    import yfinance as yf

    log.info("yfinance fetch: %s (%s)", symbol, period)
    df = yf.Ticker(symbol).history(period=period).reset_index()
    df.columns = [c.lower() for c in df.columns]
    df["date"] = pd.to_datetime(df["date"]).dt.date
    return df[["date", "close"]]
