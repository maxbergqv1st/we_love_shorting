"""The meeting point: the three separate streams (SPY price, metal price, tone)
join here into one feature table. Features = the two daily prices; target = tone."""

import pandas as pd

FEATURES = ["spy_close", "metal_close"]
TARGET = "tone"


def build_features(
    tone: pd.DataFrame, spy: pd.DataFrame, metal: pd.DataFrame
) -> pd.DataFrame:
    """Join by date: SPY close + precious-metal close predict that day's news tone."""
    df = (
        spy.rename(columns={"close": "spy_close"})
        .merge(metal.rename(columns={"close": "metal_close"}), on="date")
        .merge(tone, on="date")
        .sort_values("date")
        .dropna()
        .reset_index(drop=True)
    )
    if df.empty:
        raise ValueError("no overlapping dates across prices and tone")
    return df
