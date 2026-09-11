"""The meeting point: the three separate streams (SPY price, metal price, tone)
join here into one feature table. Features = the two daily prices; target = tone."""

import pandas as pd

FEATURES = ["spy_close", "metal_close"]
TARGET = "tone"


def build_features(
    tone: pd.DataFrame, spy: pd.DataFrame, metal: pd.DataFrame
) -> pd.DataFrame:
    """Join by date: SPY close + precious-metal close predict that day's news tone.

    Markets close on weekends/holidays but news tone still flows every day, so an
    inner join would silently drop ~2 days a week. Instead each tone date keeps its
    own news and inherits the most recent close (Friday's, over a weekend) via an
    asof (backward) join.
    """
    prices = spy.merge(metal, on="date").sort_values("date")
    tone = tone.sort_values("date")
    # merge_asof needs datetime64 keys, not datetime.date objects.
    for frame in (prices, tone):
        frame["date"] = pd.to_datetime(frame["date"])
    prices["market_date"] = prices[
        "date"
    ]  # survives asof to flag carried-forward closes
    df = (
        pd.merge_asof(tone, prices, on="date", direction="backward")
        .dropna()
        .reset_index(drop=True)
    )
    if df.empty:
        raise ValueError("no overlapping dates across prices and tone")
    # closed = price was carried forward from an earlier trading day (weekend/holiday)
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
