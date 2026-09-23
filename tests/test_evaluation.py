import numpy as np
import pandas as pd
import pytest

from we_love_shorting import evaluation


def test_drop_market_closed_keeps_only_open_rows():
    df = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=5).date,
            "market_closed": [False, False, True, True, False],
            "tone": [1, 2, 3, 4, 5],
        }
    )

    kept = evaluation.drop_market_closed(df)

    assert kept["market_closed"].tolist() == [False, False, False]
    assert kept["tone"].tolist() == [1, 2, 5]
    assert kept.index.tolist() == [0, 1, 2]  # reset, not the original positions


def test_chronological_split_respects_order_and_ratio():
    df = pd.DataFrame(
        {"date": pd.date_range("2024-01-01", periods=10).date, "value": range(10)}
    )

    train, test = evaluation.chronological_split(df, test_frac=0.2)

    assert len(train) == 8
    assert len(test) == 2
    # test is strictly the chronological tail, not a random sample.
    assert train["value"].tolist() == list(range(8))
    assert test["value"].tolist() == [8, 9]


def test_baseline_kind_mean_for_ret_persistence_otherwise():
    assert evaluation.baseline_kind("sp500_ret") == "mean"
    assert evaluation.baseline_kind("tone_diff") == "mean"  # prepare_target's move
    assert evaluation.baseline_kind("sp500_close") == "persistence"
    assert evaluation.baseline_kind("tone") == "persistence"


def test_naive_baseline_uses_mean_for_ret_target():
    train_target = pd.Series([0.01, -0.02, 0.03, 0.0], name="sp500_ret")
    test_target = pd.Series([0.05, -0.01], name="sp500_ret")

    baseline = evaluation.naive_baseline(train_target, test_target)

    assert baseline.tolist() == pytest.approx([train_target.mean()] * 2)


def test_naive_baseline_uses_persistence_for_level_target():
    train_target = pd.Series([100.0, 101.0, 102.0], name="sp500_close")
    test_target = pd.Series([103.0, 104.0, 105.0], name="sp500_close")

    baseline = evaluation.naive_baseline(train_target, test_target)

    # first test row predicted by the last train value; the rest by the
    # actual previous test row (persistence, not the model's own output).
    assert baseline.tolist() == [102.0, 103.0, 104.0]


def test_naive_baseline_persistence_with_single_row_test_set():
    # test_target.iloc[:-1] is an empty slice here — must not crash, and the
    # lone row should be seeded straight from the last train value.
    train_target = pd.Series([100.0, 101.0, 102.0], name="sp500_close")
    test_target = pd.Series([103.0], name="sp500_close")

    baseline = evaluation.naive_baseline(train_target, test_target)

    assert baseline.tolist() == [102.0]


def test_regression_metrics_matches_hand_computed_rmse_mae():
    y_true = pd.Series([3.0, -0.5, 2.0, 7.0])
    y_pred = pd.Series([2.5, 0.0, 2.0, 8.0])

    metrics = evaluation.regression_metrics(y_true, y_pred)

    errors = np.array([0.5, -0.5, 0.0, -1.0])
    assert metrics["mae"] == pytest.approx(np.abs(errors).mean())
    assert metrics["rmse"] == pytest.approx(np.sqrt((errors**2).mean()))


def test_residuals_is_actual_minus_predicted():
    y_true = pd.Series([10.0, 20.0, 30.0])
    y_pred = pd.Series([9.0, 22.0, 30.0])

    resid = evaluation.residuals(y_true, y_pred)

    assert resid.tolist() == pytest.approx([1.0, -2.0, 0.0])
