"""ML layer: regress news tone on the two daily prices. Low predicted tone =
bearish sentiment = a shorting cue. Swap LinearRegression to classify instead."""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LinearRegression

from .features import FEATURES, TARGET

MODEL_PATH = Path("data/model.joblib")


def train(
    df: pd.DataFrame,
    features: list[str] = FEATURES,
    target: str = TARGET,
    persist: bool = True,
) -> LinearRegression:
    model = LinearRegression()  # OLS is scale-invariant, so no StandardScaler needed
    fit = df.dropna(subset=[*features, target])  # a _ret-only pick keeps all rows;
    model.fit(fit[features], fit[target])  # tone is NaN before its coverage starts
    if persist:  # skip the disk write on interactive re-fits (see controller.run)
        MODEL_PATH.parent.mkdir(exist_ok=True)
        joblib.dump(model, MODEL_PATH)
    return model


def predict(
    df: pd.DataFrame,
    model: LinearRegression | None = None,
    features: list[str] = FEATURES,
) -> pd.Series:
    if model is None:
        model = joblib.load(MODEL_PATH)
    usable = df.dropna(subset=features)  # rows the model can score (tone may be NaN)
    preds = model.predict(usable[features])
    out = pd.Series(preds, index=usable.index, name="predicted")
    return out.reindex(df.index)  # NaN back where a feature was missing
