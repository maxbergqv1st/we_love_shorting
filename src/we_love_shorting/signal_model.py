"""ML layer: regress news tone on the two daily prices. Low predicted tone =
bearish sentiment = a shorting cue. Swap LinearRegression to classify instead."""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LinearRegression

from .features import FEATURES, TARGET

MODEL_PATH = Path("data/model.joblib")


def train(df: pd.DataFrame) -> LinearRegression:
    model = LinearRegression()  # OLS is scale-invariant, so no StandardScaler needed
    model.fit(df[FEATURES], df[TARGET])
    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    return model


def predict(df: pd.DataFrame, model: LinearRegression | None = None) -> pd.Series:
    if model is None:
        model = joblib.load(MODEL_PATH)
    return pd.Series(model.predict(df[FEATURES]), index=df.index, name="predicted_tone")
