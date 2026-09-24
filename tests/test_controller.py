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


def test_shift_target_shifts_over_trading_days_only():
    # closed rows (idx 3 and 7) hold carried-forward values; a calendar shift
    # would hand idx 2 its own value as "tomorrow". The shift must therefore
    # run on the open-market rows only.
    df = _df("y", [10, 12, 11, 13, 14, 15, 16, 17, 18, 19])

    out = controller.shift_target(df, "y", horizon=1)

    # 10 rows - 2 closed = 8 open rows, minus the last (unknown future) = 7.
    assert len(out) == 7
    # open-row targets are [10,12,11,14,15,16,18,19]; each row gets the NEXT one
    assert out["y"].tolist() == [12, 11, 14, 15, 16, 18, 19]
    # features stay on their own day (open rows' x, last one dropped)
    assert out["x"].tolist() == [0, 1, 2, 4, 5, 6, 8]
    # horizon 0 = nowcast, untouched frame
    assert controller.shift_target(df, "y", horizon=0) is df


def _level_df() -> pd.DataFrame:
    # closes with carried-forward values on the closed rows (idx 3 and 7),
    # like build_features' ffill produces; `_ret` computed the same way.
    close = [100.0, 102.0, 101.0, 101.0, 104.0, 106.0, 105.0, 105.0, 108.0, 110.0]
    df = _df("y_close", close)
    df["y_ret"] = pd.Series(close).pct_change().fillna(0.0)
    df["tone"] = [0.5, -0.2, 0.1, 0.4, -0.6, 0.3, 0.2, -0.1, 0.0, 0.7]
    return df


def test_prepare_target_retargets_levels_to_their_move():
    df = _level_df()
    # a close target trains on the stream's existing ret column
    df_h, spec = controller.prepare_target(df, "y_close")
    assert spec == controller.TargetSpec("y_ret", "y_close", "ret", 0)
    assert len(df_h) == len(df)  # nowcast: frame untouched
    # a level without a ret sibling gets a computed diff, first row dropped
    df_t, spec_t = controller.prepare_target(df, "tone")
    assert spec_t == controller.TargetSpec("tone_diff", "tone", "diff", 0)
    assert len(df_t) == len(df) - 1
    assert df_t["tone_diff"].iloc[0] == pytest.approx(-0.7)  # tone[1] - tone[0]
    # a ret target passes through untouched
    _, spec_r = controller.prepare_target(df, "y_ret")
    assert spec_r == controller.TargetSpec("y_ret", None, "identity", 0)


def test_prepare_target_shifts_the_move_not_the_level():
    df = _level_df()
    df_h, _spec = controller.prepare_target(df, "y_close", horizon=1)
    # open-row rets, each row handed the NEXT one; the level column stays put
    open_ret = df.loc[~df["market_closed"], "y_ret"].tolist()
    assert df_h["y_ret"].tolist() == pytest.approx(open_ret[1:])
    assert df_h["y_close"].tolist() == pytest.approx(
        df.loc[~df["market_closed"], "y_close"].tolist()[:-1]
    )


def test_reconstruction_recovers_the_actual_level_exactly():
    df = _level_df()
    # nowcast: base backed out of the row itself -> actual move recovers level
    df_h, spec = controller.prepare_target(df, "y_close")
    base = controller.level_base(df_h, spec)
    rebuilt = controller.reconstruct_level(base, df_h["y_ret"], spec)
    assert rebuilt.tolist() == pytest.approx(df_h["y_close"].tolist())
    # forecast: base = the row's own level -> actual move gives the NEXT level
    df_1, spec_1 = controller.prepare_target(df, "y_close", horizon=1)
    rebuilt = controller.reconstruct_level(
        controller.level_base(df_1, spec_1), df_1["y_ret"], spec_1
    )
    next_open_close = df.loc[~df["market_closed"], "y_close"].tolist()[1:]
    assert rebuilt.tolist() == pytest.approx(next_open_close)
    # additive kind: tone_diff rebuilds tone
    df_t, spec_t = controller.prepare_target(df, "tone")
    rebuilt = controller.reconstruct_level(
        controller.level_base(df_t, spec_t), df_t["tone_diff"], spec_t
    )
    assert rebuilt.tolist() == pytest.approx(df_t["tone"].tolist())


def test_add_level_view_persistence_equals_previous_level():
    df = _level_df()
    df_h, spec = controller.prepare_target(df, "y_close")
    result = controller.evaluate(df_h, ["x"], spec.train_col)
    result = controller.add_level_view(result, spec)
    # persistence on the level scale == predict zero move == the base itself
    test = result.test_df
    assert test["baseline_level"].tolist() == pytest.approx(
        (test["y_close"] / (1 + test["y_ret"])).tolist()
    )
    # nowcast: the actual level at target time is the row's own level
    assert test["actual_level"].tolist() == pytest.approx(test["y_close"].tolist())
    assert set(result.level_metrics) == {"model", "baseline"}
    for side in result.level_metrics.values():
        assert set(side) == {"rmse", "mae"}


def test_prepare_target_diff_spans_trading_days_at_horizon():
    # tone changes on closed days too; the forecast move must still be
    # open-row-to-open-row (Fri -> Mon), never Mon−Sun
    df = _level_df()
    open_tone = df.loc[~df["market_closed"], "tone"].to_numpy()
    df_h, spec = controller.prepare_target(df, "tone", horizon=1)
    assert df_h["tone_diff"].tolist() == pytest.approx(list(np.diff(open_tone)))
    # and the level view rebuilds the NEXT open day's tone exactly
    rebuilt = controller.reconstruct_level(
        controller.level_base(df_h, spec), df_h["tone_diff"], spec
    )
    assert rebuilt.tolist() == pytest.approx(list(open_tone[1:]))


def test_add_level_view_forecast_scores_against_the_next_level():
    df = _level_df()
    df_h, spec = controller.prepare_target(df, "y_close", horizon=1)
    result = controller.evaluate(df_h, ["x"], spec.train_col)
    result = controller.add_level_view(result, spec)
    test = result.test_df
    # at horizon 1 the actual is TOMORROW's level (base × (1 + actual move)),
    # not the row's own level column — scoring against the row's own level
    # would hand persistence a fake RMSE of 0
    assert test["actual_level"].tolist() == pytest.approx(
        (test["y_close"] * (1 + test["y_ret"])).tolist()
    )
    assert result.level_metrics["baseline"]["rmse"] > 0


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
