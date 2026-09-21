"""Each analysis mode returns render-agnostic Panels on toy data — no Streamlit
needed, which is the point of keeping the modes render-free."""

import numpy as np
import pandas as pd

from we_love_shorting import analysis, controller


def _toy_df(n: int = 60) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    close = 100 + np.cumsum(rng.normal(0, 1, n))
    gold = rng.normal(0, 0.01, n)
    return pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=n).date,
            "tone": rng.normal(0, 1, n),
            "spy_close": close,
            "spy_ret": np.r_[0.0, np.diff(close) / close[:-1]],
            "gold_ret": gold,
            "silver_ret": gold * 0.95 + rng.normal(0, 0.001, n),  # co-moves with gold
            "oil_ret": rng.normal(0, 0.01, n),
            "copper_ret": rng.normal(0, 0.01, n),
            "market_closed": False,
        }
    )


def _panel_kinds(panels):
    return [p.kind for p in panels]


def test_registry_has_the_predict_modes_only():
    # asset grouping is not a mode — it's a standalone target-independent view
    assert list(analysis.MODES) == ["Regression", "Riktning"]


def test_regression_mode_panels_and_context():
    df, mode = _toy_df(), analysis.MODES["Regression"]
    live = mode.live_panels(df, ["tone"], "spy_close", "S&P 500")
    assert _panel_kinds(live) == ["line", "line"]  # actual-vs-pred + error line

    result = mode.evaluate(df, ["tone"], "spy_close")
    panels = mode.evaluate_panels(result, "S&P 500")
    assert {"table", "scatter", "bar"} <= set(_panel_kinds(panels))
    assert "Mätvärden" in mode.context(result) or "Testperiod" in mode.context(result)


def test_direction_mode_panels_confusion_matrix():
    df, mode = _toy_df(), analysis.MODES["Riktning"]
    # rolling hit-rate line vs baseline, not a noisy per-day scatter
    assert _panel_kinds(mode.live_panels(df, ["tone"], "spy_ret", "SPY %"))[0] == "line"

    result = mode.evaluate(df, ["tone"], "spy_ret")
    tables = [p for p in mode.evaluate_panels(result, "SPY %") if p.kind == "table"]
    # accuracy table + confusion matrix
    assert len(tables) == 2
    assert mode.context(result) == ""  # no chatbot for the direction mode


def test_asset_grouping_panels():
    feats = ["gold_ret", "silver_ret", "oil_ret", "copper_ret"]
    panels = analysis.asset_grouping_panels(_toy_df(), feats)
    assert panels[0].kind == "table" and panels[0].gradient  # correlation heatmap
    assert any(p.kind == "table" for p in panels[1:])  # groups table


def test_group_assets_groups_comoving_streams():
    df = _toy_df()
    r = controller.group_assets(df, ["gold_ret", "silver_ret", "oil_ret", "copper_ret"])
    assert r.corr.shape == (4, 4)
    # gold & silver are built to co-move -> land in the same group
    assert r.groups["gold_ret"] == r.groups["silver_ret"]
