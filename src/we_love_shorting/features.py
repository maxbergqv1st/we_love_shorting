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
# Every measured value the model can use. tone (GDELT) is one column in the pool
# beside the price columns — any of them can be the TARGET or a FEATURE.
COLUMNS = [
    "tone",
    *(f"{stem}_close" for stem in TICKERS),
    *(f"{stem}_ret" for stem in TICKERS),
]
TARGET = "tone"  # default target (swappable in the UI)
FEATURES = [c for c in COLUMNS if c != TARGET]  # default: predict from all the rest


def stream_of(column: str) -> str:
    """The stream a column belongs to: `sp500_close` and `sp500_ret` both map to
    `sp500`; `tone` maps to itself. Lets the UI treat a stream's level and return
    as one group, so picking either as the target excludes both from features
    (a stream must not predict itself via its own level/return)."""
    return column.removesuffix("_close").removesuffix("_ret")


def display_column(target: str) -> str:
    """The column to show/reconstruct for a target: a `_ret` target maps to its
    stream's `_close` (the price line the chart draws from the predicted return);
    any other target shows itself. One home for the _ret->_close convention used
    by controller.run (reconstruction) and the UI (chart/metrics)."""
    return f"{stream_of(target)}_close" if target.endswith("_ret") else target


def reconstruct_close(close: pd.Series, predicted_ret: pd.Series) -> pd.Series:
    """Walk-forward price from a predicted daily return: yesterday's ACTUAL
    close × (1 + today's prediction). One-step, so errors don't accumulate.
    First row is NaN (no prior close). Lets the model stay in stationary
    returns (clean residuals) while the chart still shows a price line."""
    return close.shift(1) * (1 + predicted_ret)


def build_features(tone: pd.DataFrame, prices: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Join every price stream + news tone by date into one wide table.

    The streams trade on different calendars (US, Stockholm, Europe, commodities),
    so an inner join would drop every day any one exchange was closed. Instead we
    outer-merge and forward-fill: a day one exchange is shut inherits that
    exchange's most recent close. The row backbone is every date the prices OR
    tone touch, so pre-tone price history survives (tone is just NaN there); an
    asof (backward) join carries the most recent price block and the most recent
    tone onto each date — Friday's close over the weekend, NaN tone before GDELT's
    coverage begins. Rows are gated only on the price columns, so a returns-only
    model keeps the full history; tone/target NaNs are dropped per-model at fit
    time (see signal_model.train / controller.evaluate).

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
        # own trading calendar, pre-merge: see the `_ret` note above.
        frame[f"{stem}_ret"] = frame[f"{stem}_close"].pct_change()
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
    # Backbone = every date the prices OR tone touch, so pre-tone trading history
    # survives. asof carries the most recent price block + most recent tone onto
    # each date (Friday's close over the weekend; NaN tone before GDELT starts).
    dates = pd.Index(merged["date"]).union(tone["date"])  # sorted, deduped union
    df = pd.merge_asof(
        pd.DataFrame({"date": dates}), merged, on="date", direction="backward"
    )
    df = pd.merge_asof(df, tone, on="date", direction="backward")
    # Gate on the price columns only: this drops the priming first row (its `_ret`
    # is NaN) and any date before a stream has data. tone is intentionally allowed
    # to stay NaN — it's dropped per-model at fit time, so a `_ret`-only model
    # keeps the full price history while a tone pick trims to tone's coverage.
    price_cols = [f"{s}_close" for s in prices] + [f"{s}_ret" for s in prices]
    df = df.dropna(subset=price_cols).reset_index(drop=True)
    if df.empty:
        raise ValueError("no price data to build features from")
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
