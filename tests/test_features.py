import pandas as pd

from we_love_shorting import features


def test_build_features_joins_streams_and_targets_tone():
    dates = pd.date_range("2024-01-01", periods=4).date
    tone = pd.DataFrame({"date": dates, "tone": [1.0, -2.0, 0.5, 3.0]})
    prices = {
        "sp500": pd.DataFrame({"date": dates, "sp500_close": [100, 110, 105, 108]}),
        "gold": pd.DataFrame({"date": dates, "gold_close": [2000, 2010, 1990, 2020]}),
        "oil": pd.DataFrame({"date": dates, "oil_close": [70, 72, 71, 73]}),
    }

    df = features.build_features(tone, prices)

    assert features.FEATURES == [f"{n}_close" for n in features.TICKERS]
    assert "tone" in features.COLUMNS and features.TARGET == "tone"
    assert features.TARGET == "tone"
    cols = ["sp500_close", "gold_close", "oil_close", "tone"]
    assert set(cols) <= set(df.columns)
    assert df["tone"].tolist() == [1.0, -2.0, 0.5, 3.0]
    assert not df[cols].isna().any().any()


def test_weekend_tone_kept_with_friday_close():
    # Fri..Mon; markets closed Sat/Sun so prices only exist Fri and Mon.
    fri, sat, sun, mon = pd.date_range("2024-01-05", periods=4).date
    tone = pd.DataFrame({"date": [fri, sat, sun, mon], "tone": [1.0, 2.0, 3.0, 4.0]})
    prices = {
        "sp500": pd.DataFrame({"date": [fri, mon], "sp500_close": [100, 108]}),
        "gold": pd.DataFrame({"date": [fri, mon], "gold_close": [2000, 2020]}),
    }

    df = features.build_features(tone, prices)

    # All 4 tone days survive; Sat/Sun inherit Friday's close.
    assert df["tone"].tolist() == [1.0, 2.0, 3.0, 4.0]
    assert df["sp500_close"].tolist() == [100, 100, 100, 108]
    assert df["gold_close"].tolist() == [2000, 2000, 2000, 2020]
    # Sat/Sun flagged closed (carried-forward price); Fri/Mon are real trading days.
    assert df["market_closed"].tolist() == [False, True, True, False]


def test_different_exchange_calendars_do_not_drop_days():
    # One stream is closed on a day the other trades: the closed one carries
    # forward instead of the whole row being dropped (inner join would drop it).
    mon, tue, wed = pd.date_range("2024-01-01", periods=3).date
    tone = pd.DataFrame({"date": [mon, tue, wed], "tone": [1.0, 2.0, 3.0]})
    prices = {
        "sp500": pd.DataFrame(
            {"date": [mon, tue, wed], "sp500_close": [100, 101, 102]}
        ),
        "omx30": pd.DataFrame(
            {"date": [mon, wed], "omx30_close": [200, 202]}
        ),  # no Tue
    }

    df = features.build_features(tone, prices)

    assert df["tone"].tolist() == [1.0, 2.0, 3.0]  # all days kept
    assert df["omx30_close"].tolist() == [200, 200, 202]  # Tue carried from Mon


def test_reference_market_holiday_still_flagged_when_foreign_open():
    # Tue: reference (sp500) is shut but a foreign exchange trades. The row is
    # kept, but market_closed must stay True — sp500's close was carried forward.
    mon, tue, wed = pd.date_range("2024-01-01", periods=3).date  # Mon..Wed
    tone = pd.DataFrame({"date": [mon, tue, wed], "tone": [1.0, 2.0, 3.0]})
    prices = {
        "sp500": pd.DataFrame(
            {"date": [mon, wed], "sp500_close": [100, 102]}
        ),  # no Tue
        "omx30": pd.DataFrame(
            {"date": [mon, tue, wed], "omx30_close": [200, 201, 202]}
        ),
    }

    df = features.build_features(tone, prices)

    assert df["sp500_close"].tolist() == [100, 100, 102]  # Tue carried from Mon
    # Tue flagged closed even though omx30 traded; Mon/Wed are real reference days.
    assert df["market_closed"].tolist() == [False, True, False]


def test_pending_weekday_close_not_flagged():
    # Latest tone day is a weekday whose close hasn't published yet: pending, not closed.
    thu, fri = pd.date_range("2024-01-04", periods=2).date  # Thu, Fri
    tone = pd.DataFrame({"date": [thu, fri], "tone": [1.0, 2.0]})
    prices = {"sp500": pd.DataFrame({"date": [thu], "sp500_close": [100]})}

    df = features.build_features(tone, prices)

    assert df["market_closed"].tolist() == [False, False]
