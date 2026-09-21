import pandas as pd
import pytest

from we_love_shorting import controller, signal_model


def test_backtest_date_ignores_future_rows():
    # A point-in-time backtest must use ONLY data before the tested date: adding
    # later rows to the frame cannot change the prediction (no lookahead).
    dates = pd.date_range("2024-01-01", periods=30).date
    close = pd.Series([100.0 + k for k in range(30)])
    df = pd.DataFrame(
        {
            "date": dates,
            "tone": 0.0,
            "sp500_close": close,
            "sp500_ret": close.pct_change().fillna(0.0),
            "gold_close": close * 2,
            "gold_ret": (close * 2).pct_change().fillna(0.0),
            "market_closed": False,
        }
    )
    d = dates[20]
    full = controller.backtest_date(df, ["gold_ret"], "sp500_ret", d, horizon=1)
    trunc = controller.backtest_date(df.iloc[:22], ["gold_ret"], "sp500_ret", d, 1)
    assert full["predicted"] == pytest.approx(trunc["predicted"])  # future ignored
    assert full["actual"] == pytest.approx(trunc["actual"])
    assert full["baseline"] == pytest.approx(trunc["baseline"])  # baseline too
    assert full["trained_until"] < full["date"]  # trained strictly on the past


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

    model = signal_model.train(df, features, target, persist=False)
    pred = signal_model.predict(df, model, features)
    assert len(pred) == len(df)

    out = controller.run(df.copy(), features, target)
    assert f"predicted_{target}" in out.columns  # dynamic column name
