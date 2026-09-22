"""The meeting point: the fixed price streams and news tone join here into one
wide feature table. Any numeric column can be the prediction target; the rest
are candidate features (picked in the UI)."""

import pandas as pd

# Fixed Yahoo streams: column-stem -> ticker. The UI no longer asks for tickers.
# Each is stored as a `<stem>_close` column.
TICKERS = {
    "sp500": "^GSPC",  # S&P 500 (US)
    "omx30": "^OMX",  # OMX Stockholm 30 (Sweden)
    "eurostoxx": "^STOXX50E",  # EURO STOXX 50 (Europe)
    "gold": "GC=F",
    "silver": "SI=F",
    "copper": "HG=F",
    "oil": "CL=F",  # crude oil
}
# Per-stream columns: the raw level/return plus derived context features
# (rolling mean = trend, rolling std of returns = volatility) — the standard
# extra signal a forecaster wants beyond today's raw numbers.
STREAM_SUFFIXES = ("close", "ret", "ma5", "ma21", "vol21")
# Every measured value the model can use. tone (GDELT) is one column in the pool
# beside the price columns — any of them can be the TARGET or a FEATURE.
COLUMNS = [
    "tone",
    *(f"{stem}_{sfx}" for sfx in STREAM_SUFFIXES for stem in TICKERS),
]
TARGET = "tone"  # default target (swappable in the UI)
FEATURES = [c for c in COLUMNS if c != TARGET]  # default: predict from all the rest


def stream_of(column: str) -> str:
    """The stream a column belongs to: `sp500_close`, `sp500_ret`, `sp500_ma21`…
    all map to `sp500`; `tone` maps to itself. Works for any suffix because the
    stems in TICKERS never contain an underscore. Lets the UI treat a stream's
    columns as one group, so picking any of them as the target excludes the
    whole stream from features (a stream must not predict itself)."""
    return column.split("_")[0]


def build_features(tone: pd.DataFrame, prices: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Join every price stream + news tone by date into one wide table.

    The streams trade on different calendars (US, Stockholm, Europe, commodities),
    so an inner join would drop every day any one exchange was closed. Instead we
    outer-merge and forward-fill: a day one exchange is shut inherits that
    exchange's most recent close. News tone flows daily, so each tone date then
    inherits the most recent price block via an asof (backward) join — Friday's
    close carries over the weekend.

    Each stream also gets a `<stem>_ret` column: that day's percentage change,
    computed on the stream's OWN rows (its own trading calendar) before the
    outer-merge/ffill below. Computing it here, not after ffill on the merged
    table, matters: after ffill a closed day's close equals the prior day's
    close, so a naive post-merge pct_change() would read as a flat 0% on every
    day that exchange was shut, which is wrong — the real return is whatever it
    was on the last actual trading day, carried forward (see the ffill note
    below), not zero.

    `market_closed` flags carried-forward closes against ONE reference calendar
    (the first stream, i.e. the S&P 500 / US market) — a foreign exchange trading
    on a US holiday must not un-flag that carried-forward US close.
    """
    frames = []
    for stem, frame in prices.items():
        frame = frame.copy()
        # own trading calendar, pre-merge: see the `_ret` note above. Rolling
        # windows use min_periods so a short history yields partial-window
        # stats instead of NaN-dropping its first month (std needs ≥2 obs, so
        # the very first ret row is NaN and gets dropped like before).
        close, ret = frame[f"{stem}_close"], frame[f"{stem}_close"].pct_change()
        frame[f"{stem}_ret"] = ret
        frame[f"{stem}_ma5"] = close.rolling(5, min_periods=1).mean()
        frame[f"{stem}_ma21"] = close.rolling(21, min_periods=1).mean()
        # warmup rows are backfilled from the first computable std (and a
        # degenerate <3-row history gets 0.0) so the derived column never
        # widens the global dropna beyond what `_ret` already drops.
        frame[f"{stem}_vol21"] = (
            ret.rolling(21, min_periods=2).std().bfill().fillna(0.0)
        )
        frames.append(frame)
    ref_dates = pd.to_datetime(frames[0]["date"])  # reference market's trading days
    merged = frames[0]
    for frame in frames[1:]:
        merged = merged.merge(frame, on="date", how="outer")
    merged = merged.sort_values("date")
    # merge_asof needs datetime64 keys, not datetime.date objects.
    merged["date"] = pd.to_datetime(merged["date"])
    # market_date holds the date only on reference trading days (NaT otherwise);
    # ffill then makes it the most recent reference close, so days the reference
    # market was shut keep pointing back at it even if another exchange traded.
    merged["market_date"] = merged["date"].where(merged["date"].isin(ref_dates))
    # ponytail: ffill carries a paused/delisted stream's last close forward
    # indefinitely; add a max-carry-forward staleness guard if a feed goes dark.
    merged = merged.ffill()  # carry each stream's last close over its own closed days
    # ffill also carries each `_ret` forward unchanged on closed days: the carried
    # value is that stream's last real trading day's return, not 0% and not a
    # freshly computed one — same carry-forward behavior as `_close` above, called
    # out here since it's easy to assume ffill only touches price levels.
    tone = tone.sort_values("date")
    tone["date"] = pd.to_datetime(tone["date"])
    df = (
        pd.merge_asof(tone, merged, on="date", direction="backward")
        # dropna is unconditional over every registered column, not just
        # whatever TARGET/FEATURES the caller later picks in the UI — that's
        # pre-existing behavior, not new. `_ret` widens its reach by one known,
        # accepted cost: the very first stored day of history has no prior
        # close, so its `_ret` is NaN and that day gets dropped too. Treated as
        # a one-time PoC-stage simplification rather than teaching this
        # function about the target/features choice (which controller.get_data
        # caches before that choice is even made).
        .dropna()
        .reset_index(drop=True)
    )
    if df.empty:
        raise ValueError("no overlapping dates across prices and tone")
    # closed = price block was carried forward from an earlier trading day
    df["market_closed"] = df["date"] != df["market_date"]
    # The latest date is "today": a carried-forward close there just means the
    # market hasn't closed yet (pending), not that it's a non-trading day. Only
    # weekends are truly closed on the current day.
    # ponytail: weekday holidays on the current day slip through; add a holiday
    # calendar if that edge matters.
    last = df.index[-1]
    if df.loc[last, "market_closed"] and df.loc[last, "date"].dayofweek < 5:
        df.loc[last, "market_closed"] = False
    df = df.drop(columns="market_date")
    df["date"] = df["date"].dt.date
    return df
