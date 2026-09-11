"""The meeting point: the four separate streams (SPY price, metal price, oil
price, tone) join here into one feature table. Features = the three daily
prices; target = tone."""

import pandas as pd

FEATURES = ["spy_close", "metal_close", "oil_close"]
TARGET = "tone"


def build_features(
    tone: pd.DataFrame, spy: pd.DataFrame, metal: pd.DataFrame, oil: pd.DataFrame
) -> pd.DataFrame:
    """Join by date: SPY + precious-metal + oil close predict that day's news tone."""
    df = (
        spy.merge(metal, on="date")
        .merge(oil, on="date")
        .merge(tone, on="date")
        .sort_values("date")
        .dropna()
        .reset_index(drop=True)
    )
    if df.empty:
        raise ValueError("no overlapping dates across prices and tone")
    return df
