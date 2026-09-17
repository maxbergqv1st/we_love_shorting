import pandas as pd
import pytest

from we_love_shorting import features


def test_stream_of_groups_close_and_ret():
    # a stream's level and return map to the same stream; tone stands alone.
    assert features.stream_of("sp500_close") == "sp500"
    assert features.stream_of("sp500_ret") == "sp500"
    assert features.stream_of("tone") == "tone"


def test_build_features_joins_streams_and_targets_tone():
    # day0 primes pct_change (a stream's first-ever row has no prior close, so
    # its `_ret` is NaN and the row gets dropped) — day0 isn't a tone date, so
    # it never surfaces in the output; it just gives day1 a real `_ret`.
    day0, *dates = pd.date_range("2023-12-31", periods=5).date
    tone = pd.DataFrame({"date": dates, "tone": [1.0, -2.0, 0.5, 3.0]})
    prices = {
        "sp500": pd.DataFrame(
            {"date": [day0, *dates], "sp500_close": [99, 100, 110, 105, 108]}
        ),
        "gold": pd.DataFrame(
            {"date": [day0, *dates], "gold_close": [1990, 2000, 2010, 1990, 2020]}
        ),
        "oil": pd.DataFrame(
            {"date": [day0, *dates], "oil_close": [69, 70, 72, 71, 73]}
        ),
    }

    df = features.build_features(tone, prices)

    assert features.FEATURES == [
        *(f"{n}_close" for n in features.TICKERS),
        *(f"{n}_ret" for n in features.TICKERS),
    ]
    assert "tone" in features.COLUMNS and features.TARGET == "tone"
    assert features.TARGET == "tone"
    cols = ["sp500_close", "gold_close", "oil_close", "tone"]
    assert set(cols) <= set(df.columns)
    assert df["tone"].tolist() == [1.0, -2.0, 0.5, 3.0]
    assert not df[cols].isna().any().any()


def test_weekend_tone_kept_with_friday_close():
    # Thu primes pct_change; Fri..Mon is the actual test window (markets
    # closed Sat/Sun so prices only exist Thu, Fri and Mon).
    thu, fri, sat, sun, mon = pd.date_range("2024-01-04", periods=5).date
    tone = pd.DataFrame({"date": [fri, sat, sun, mon], "tone": [1.0, 2.0, 3.0, 4.0]})
    prices = {
        "sp500": pd.DataFrame({"date": [thu, fri, mon], "sp500_close": [99, 100, 108]}),
        "gold": pd.DataFrame(
            {"date": [thu, fri, mon], "gold_close": [1990, 2000, 2020]}
        ),
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
    # day0 primes pct_change; Mon..Wed is the actual test window.
    day0, mon, tue, wed = pd.date_range("2023-12-31", periods=4).date
    tone = pd.DataFrame({"date": [mon, tue, wed], "tone": [1.0, 2.0, 3.0]})
    prices = {
        "sp500": pd.DataFrame(
            {"date": [day0, mon, tue, wed], "sp500_close": [99, 100, 101, 102]}
        ),
        "omx30": pd.DataFrame(
            {"date": [day0, mon, wed], "omx30_close": [199, 200, 202]}
        ),  # no Tue
    }

    df = features.build_features(tone, prices)

    assert df["tone"].tolist() == [1.0, 2.0, 3.0]  # all days kept
    assert df["omx30_close"].tolist() == [200, 200, 202]  # Tue carried from Mon


def test_reference_market_holiday_still_flagged_when_foreign_open():
    # Tue: reference (sp500) is shut but a foreign exchange trades. The row is
    # kept, but market_closed must stay True — sp500's close was carried forward.
    # day0 primes pct_change; Mon..Wed is the actual test window.
    day0, mon, tue, wed = pd.date_range("2023-12-31", periods=4).date
    tone = pd.DataFrame({"date": [mon, tue, wed], "tone": [1.0, 2.0, 3.0]})
    prices = {
        "sp500": pd.DataFrame(
            {"date": [day0, mon, wed], "sp500_close": [99, 100, 102]}
        ),  # no Tue
        "omx30": pd.DataFrame(
            {"date": [day0, mon, tue, wed], "omx30_close": [199, 200, 201, 202]}
        ),
    }

    df = features.build_features(tone, prices)

    assert df["sp500_close"].tolist() == [100, 100, 102]  # Tue carried from Mon
    # Tue flagged closed even though omx30 traded; Mon/Wed are real reference days.
    assert df["market_closed"].tolist() == [False, True, False]


def test_pending_weekday_close_not_flagged():
    # Latest tone day is a weekday whose close hasn't published yet: pending, not closed.
    # Wed primes pct_change; Thu/Fri is the actual test window.
    wed, thu, fri = pd.date_range("2024-01-03", periods=3).date
    tone = pd.DataFrame({"date": [thu, fri], "tone": [1.0, 2.0]})
    prices = {"sp500": pd.DataFrame({"date": [wed, thu], "sp500_close": [99, 100]})}

    df = features.build_features(tone, prices)

    assert df["market_closed"].tolist() == [False, False]


def test_ret_column_uses_own_calendar_not_post_ffill_pct_change():
    # gold's own calendar skips day2 (simulates a closed market day); its
    # `_ret` must be computed on its own rows (day1 -> day3 spans the closed
    # day) and then carried forward on day2 by ffill, not recomputed as a
    # flat 0% from the ffilled close.
    d0, d1, d2, d3 = pd.date_range("2024-01-01", periods=4).date
    tone = pd.DataFrame({"date": [d0, d1, d2, d3], "tone": [1.0, 1.0, 1.0, 1.0]})
    prices = {
        "sp500": pd.DataFrame(
            {"date": [d0, d1, d2, d3], "sp500_close": [100, 101, 102, 103]}
        ),
        "gold": pd.DataFrame({"date": [d0, d1, d3], "gold_close": [90, 100, 108]}),
    }

    df = features.build_features(tone, prices).set_index("date")

    # d1: real return on gold's own calendar (90 -> 100).
    assert df.loc[d1, "gold_ret"] == pytest.approx((100 - 90) / 90)
    # d2: gold was closed -> ret carried forward from d1, not recomputed as 0%.
    assert df.loc[d2, "gold_ret"] == pytest.approx((100 - 90) / 90)
    # d3: the day after the gap gets the real change spanning the closed day
    # (100 -> 108), not a value derived from the ffilled d2 close.
    assert df.loc[d3, "gold_ret"] == pytest.approx((108 - 100) / 100)
