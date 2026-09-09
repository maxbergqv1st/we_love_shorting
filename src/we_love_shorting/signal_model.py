"""Model / ML layer: tone-based classifier for next-day down-moves (short signal)."""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

MODEL_PATH = Path("data/model.joblib")
FEATURES = ["tone", "tone_ma3"]


def build_features(tone: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    """Join tone+price by date; target = next day closes down (a shorting chance)."""
    df = prices.merge(tone, on="date", how="inner").sort_values("date")
    df["ret"] = df["close"].pct_change()
    df["tone_ma3"] = df["tone"].rolling(3).mean()
    df["target"] = (df["ret"].shift(-1) < 0).astype(int)
    return df.dropna().reset_index(drop=True)


def train(df: pd.DataFrame) -> Pipeline:
    model = make_pipeline(StandardScaler(), LogisticRegression())
    model.fit(df[FEATURES], df["target"])
    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    return model


def predict(df: pd.DataFrame) -> pd.Series:
    model = joblib.load(MODEL_PATH)
    return pd.Series(model.predict_proba(df[FEATURES])[:, 1], index=df.index, name="short_prob")
