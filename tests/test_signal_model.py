import pandas as pd
import pytest

from we_love_shorting import controller, evaluation, signal_model


def test_dynamic_features_and_target_roundtrip():
    # Arbitrary pick: predict spy_close from tone + oil_close (not the defaults).
    df = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=5).date,
            "tone": [1.0, -2.0, 0.5, 3.0, -1.0],
            "spy_close": [100.0, 110.0, 105.0, 108.0, 102.0],
            "oil_close": [70.0, 72.0, 71.0, 73.0, 69.0],
        }
    )
    features, target = ["tone", "oil_close"], "spy_close"

    model = signal_model.train(df, features, target)
    pred = signal_model.predict(df, model, features)
    assert len(pred) == len(df)

    out = controller.run(df.copy(), features, target)
    assert f"predicted_{target}" in out.columns  # dynamic column name


def test_forest_model_and_weights():
    df = pd.DataFrame(
        {
            "tone": [1.0, -2.0, 0.5, 3.0, -1.0, 2.0, 0.0, 1.5],
            "oil_close": [70.0, 72.0, 71.0, 73.0, 69.0, 74.0, 70.0, 72.0],
            "spy_close": [100.0, 110.0, 105.0, 108.0, 102.0, 111.0, 101.0, 107.0],
        }
    )
    feats, target = ["tone", "oil_close"], "spy_close"

    forest = signal_model.train(df, feats, target, model_kind="forest")
    assert len(signal_model.predict(df, forest, feats)) == len(df)
    w = signal_model.weights(forest, feats)
    assert list(w.index) == feats
    assert w.sum() == pytest.approx(1.0)  # importances sum to 1

    linear = signal_model.train(df, feats, target)
    assert list(signal_model.weights(linear, feats).index) == feats  # coef path


def test_direction_labels_dead_zone():
    # Dead-zone: within ±threshold is flat (0), above is up (1), below down (-1).
    r = pd.Series([0.05, -0.05, 0.001, 0.0, -0.02])
    assert evaluation.direction_labels(r, threshold=0.01).tolist() == [1, -1, 0, 0, -1]


def test_direction_classifier_roundtrip():
    # Separable toy: oil_ret sign tracks tone's sign, so the classifier should
    # recover the direction perfectly. flat_frac=0 -> pure up/down, no flat bin.
    df = pd.DataFrame(
        {
            "tone": [2.0, -2.0, 1.5, -1.5, 3.0, -3.0, 0.5, -0.5],
            "oil_ret": [0.02, -0.02, 0.01, -0.03, 0.04, -0.01, 0.02, -0.02],
        }
    )
    threshold = evaluation.direction_threshold(df["oil_ret"], flat_frac=0.0)
    y = evaluation.direction_labels(df["oil_ret"], threshold)

    model = signal_model.train_direction(df[["tone"]], y)
    pred = signal_model.predict_direction(df[["tone"]], model)
    assert pred.tolist() == y.tolist()  # direction recovered
