"""Unsupervised layer: group trading days into regimes by their scaled features.
KMeans on the same two prices; k picked by best silhouette. Cluster labels let
the dashboard colour days by regime instead of only predicting tone."""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from .features import FEATURES

MODEL_PATH = Path("data/cluster.joblib")


def train(df: pd.DataFrame, k_range: range = range(2, 8)) -> Pipeline:
    """Sweep k, keep the KMeans with the best silhouette on the scaled features."""
    X = df[FEATURES]
    best_model, best_score = None, -1.0
    for k in k_range:
        if k >= len(X):  # need more samples than clusters
            break
        model = make_pipeline(StandardScaler(), KMeans(n_clusters=k, random_state=42))
        labels = model.fit_predict(X)
        score = silhouette_score(
            model.named_steps["standardscaler"].transform(X), labels
        )
        if score > best_score:
            best_model, best_score = model, score
    if best_model is None:
        raise ValueError(f"not enough samples ({len(X)}) to cluster")
    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(best_model, MODEL_PATH)
    return best_model


def predict(df: pd.DataFrame, model: Pipeline | None = None) -> pd.Series:
    if model is None:
        model = joblib.load(MODEL_PATH)
    return pd.Series(model.predict(df[FEATURES]), index=df.index, name="regime")


if __name__ == "__main__":
    import numpy as np

    rng = np.random.default_rng(0)
    blob = pd.DataFrame(
        {
            "spy_close": np.r_[rng.normal(400, 2, 30), rng.normal(500, 2, 30)],
            "metal_close": np.r_[rng.normal(20, 1, 30), rng.normal(30, 1, 30)],
        }
    )
    labels = predict(blob, train(blob))
    assert labels.nunique() == 2, f"expected 2 regimes, got {labels.nunique()}"
    print("ok:", labels.value_counts().to_dict())
