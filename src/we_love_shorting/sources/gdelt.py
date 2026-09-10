"""GDELT DOC 2.0 news tone (stdlib HTTP only). One source, one job."""

import json
import logging
import urllib.error
import urllib.parse
import urllib.request

import pandas as pd

from ..retry import retry

log = logging.getLogger(__name__)

GDELT_DOC = "https://api.gdeltproject.org/api/v2/doc/doc"


@retry(times=4, exceptions=(urllib.error.HTTPError,))
def _get_json(url: str) -> dict:
    """GET JSON with backoff — GDELT rate-limits per IP with 429s."""
    req = urllib.request.Request(url, headers={"User-Agent": "we_love_shorting/0.1"})
    with urllib.request.urlopen(req) as r:
        return json.load(r)


def fetch_tone(query: str, timespan: str = "12m") -> pd.DataFrame:
    """Daily average news tone for a query term (GDELT DOC 2.0 timelinetone)."""
    params = urllib.parse.urlencode(
        {"query": query, "mode": "timelinetone", "timespan": timespan, "format": "json"}
    )
    url = f"{GDELT_DOC}?{params}"
    log.info("GDELT fetch: %s", url)
    data = _get_json(url)
    timeline = data.get("timeline") or [{}]  # empty query result -> [] -> [{}]
    rows = timeline[0].get("data", [])
    if not rows:
        raise ValueError(f"GDELT returned no data for query {query!r}")
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    return df.rename(columns={"value": "tone"})[["date", "tone"]]
