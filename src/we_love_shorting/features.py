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
    df = (
        pd.merge_asof(tone, prices, on="date", direction="backward")
        .dropna()
        .reset_index(drop=True)
    )
    df["date"] = df["date"].dt.date
    if df.empty:
        raise ValueError("no overlapping dates across prices and tone")
    return df
