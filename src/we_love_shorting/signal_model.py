"""ML layer: regress news tone on the two daily prices. Low predicted tone =
bearish sentiment = a shorting cue. Swap LinearRegression to classify instead."""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from .features import FEATURES, TARGET

MODEL_PATH = Path("data/model.joblib")


def train(df: pd.DataFrame) -> Pipeline:
    model = make_pipeline(StandardScaler(), LinearRegression())
    model.fit(df[FEATURES], df[TARGET])
    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    return model


def predict(df: pd.DataFrame, model: Pipeline | None = None) -> pd.Series:
    if model is None:
        model = joblib.load(MODEL_PATH)
    return pd.Series(model.predict(df[FEATURES]), index=df.index, name="predicted_tone")
