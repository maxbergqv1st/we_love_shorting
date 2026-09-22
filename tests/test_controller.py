import numpy as np
import pandas as pd
import pytest

from we_love_shorting import controller


def _df(target_col: str, y: list[float]) -> pd.DataFrame:
    # 10 rows, 2 of them market_closed (idx 3 and 7) — evaluate() must drop
    # those before splitting, leaving 8 rows -> 6 train / 2 test (test_frac=0.2).
    closed = [False, False, False, True, False, False, False, True, False, False]
    return pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=10).date,
            "x": list(range(10)),
            target_col: y,
            "market_closed": closed,
        }
    )


def test_shift_target_makes_row_t_carry_t_plus_horizon():
    df = _df("y", [10, 12, 11, 13, 14, 15, 16, 17, 18, 19])

    out = controller.shift_target(df, "y", horizon=1)

    # row t's target is now t+1's actual; the last row (unknown future) is gone
    # and the features (x) stay untouched at their own day.
    assert len(out) == 9
    assert out["y"].tolist() == [12, 11, 13, 14, 15, 16, 17, 18, 19]
    assert out["x"].tolist() == list(range(9))
    # horizon 0 = nowcast, untouched frame
    assert controller.shift_target(df, "y", horizon=0) is df


def test_evaluate_drops_closed_rows_then_splits_chronologically():
    df = _df("y", [10, 12, 11, 13, 14, 15, 16, 17, 18, 19])

    result = controller.evaluate(df, ["x"], "y")

    # 10 rows - 2 closed = 8; 80/20 split -> 6 train, 2 test.
    assert len(result.train_df) == 6
    assert len(result.test_df) == 2
    assert not result.train_df["market_closed"].any()
    assert not result.test_df["market_closed"].any()
    # the closed rows' x-values (3, 7) must not appear on either side.
    assert 3 not in result.train_df["x"].tolist() + result.test_df["x"].tolist()
    assert 7 not in result.train_df["x"].tolist() + result.test_df["x"].tolist()


def test_evaluate_uses_persistence_baseline_for_level_target():
    df = _df("y", [10, 12, 11, 13, 14, 15, 16, 17, 18, 19])

    result = controller.evaluate(df, ["x"], "y")

    assert result.baseline_kind == "persistence"
    # train tail (after dropping closed rows) ends at y=16; test is [18, 19].
    assert result.test_df["baseline_y"].tolist() == [16.0, 18.0]
    assert result.test_df["baseline_residual"].tolist() == pytest.approx([2.0, 1.0])
    assert result.metrics["baseline"]["mae"] == pytest.approx(1.5)
    assert result.metrics["baseline"]["rmse"] == pytest.approx(np.sqrt(2.5))


def test_evaluate_uses_mean_baseline_for_ret_target():
    df = _df("y_ret", [0.01, 0.02, -0.01, 0.03, 0.0, 0.01, -0.02, 0.02, 0.01, -0.01])

    result = controller.evaluate(df, ["x"], "y_ret")

    assert result.baseline_kind == "mean"
    train_mean = result.train_df["y_ret"].mean()
    assert result.test_df["baseline_y_ret"].tolist() == pytest.approx(
        [train_mean, train_mean]
    )


def test_evaluate_result_shape_and_metrics_keys():
    df = _df("y", [10, 12, 11, 13, 14, 15, 16, 17, 18, 19])

    result = controller.evaluate(df, ["x"], "y")

    assert set(result.metrics) == {"model", "baseline"}
    for side in ("model", "baseline"):
        assert set(result.metrics[side]) == {"rmse", "mae"}
        assert result.metrics[side]["rmse"] >= 0
        assert result.metrics[side]["mae"] >= 0
    for col in ("predicted_y", "baseline_y", "residual", "baseline_residual"):
        assert col in result.test_df.columns
