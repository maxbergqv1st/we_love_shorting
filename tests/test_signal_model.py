import pandas as pd

from we_love_shorting import signal_model


def test_build_features_labels_next_day_down():
    tone = pd.DataFrame({"date": pd.date_range("2024-01-01", periods=5).date, "tone": [1, 2, 3, 4, 5]})
    prices = pd.DataFrame(
        {"date": pd.date_range("2024-01-01", periods=5).date, "close": [100, 110, 105, 108, 104]}
    )
    df = signal_model.build_features(tone, prices)

    # first 2 rows drop (tone_ma3 NaN); kept closes 105,108,104 -> next-day up,down,(last NaN->0)
    assert df["target"].tolist() == [0, 1, 0]
    assert not df[signal_model.FEATURES].isna().any().any()
