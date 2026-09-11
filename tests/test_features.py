import pandas as pd

from we_love_shorting import features


def test_build_features_joins_streams_and_targets_tone():
    dates = pd.date_range("2024-01-01", periods=4).date
    tone = pd.DataFrame({"date": dates, "tone": [1.0, -2.0, 0.5, 3.0]})
    spy = pd.DataFrame({"date": dates, "spy_close": [100, 110, 105, 108]})
    metal = pd.DataFrame({"date": dates, "metal_close": [2000, 2010, 1990, 2020]})
    oil = pd.DataFrame({"date": dates, "oil_close": [70, 72, 71, 73]})

    df = features.build_features(tone, spy, metal, oil)

    assert features.FEATURES == ["spy_close", "metal_close", "oil_close"]
    assert features.TARGET == "tone"
    assert set(features.FEATURES + [features.TARGET]) <= set(df.columns)
    assert df["tone"].tolist() == [1.0, -2.0, 0.5, 3.0]
    assert not df[features.FEATURES].isna().any().any()
