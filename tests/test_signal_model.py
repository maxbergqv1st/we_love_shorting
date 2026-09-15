import pandas as pd

from we_love_shorting import controller, signal_model


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
