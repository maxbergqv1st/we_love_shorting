"""Headless smoke test: app.py renders end-to-end without exceptions."""

from pathlib import Path

import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest

from we_love_shorting import features
from we_love_shorting.sources import yahoo

APP = str(Path(__file__).parents[1] / "app.py")


def _fake_df(rows: int = 120) -> pd.DataFrame:
    """A feature table shaped like features.build_features' output."""
    rng = np.random.default_rng(0)
    dates = pd.date_range("2024-01-02", periods=rows, freq="B")
    cols: dict = {
        "date": dates.date,
        "tone": rng.normal(size=rows),
        # always present in build_features' output; without it the evaluation
        # card fails into its except and the test would never exercise it
        "market_closed": np.zeros(rows, dtype=bool),
    }
    for stem in features.TICKERS:
        close = pd.Series(np.cumsum(rng.normal(size=rows)) + 100)
        cols[f"{stem}_close"] = close
        cols[f"{stem}_ret"] = close.pct_change().fillna(0)
        cols[f"{stem}_ma5"] = close.rolling(5, min_periods=1).mean()
        cols[f"{stem}_ma21"] = close.rolling(21, min_periods=1).mean()
        vol = cols[f"{stem}_ret"].rolling(21, min_periods=1).std()
        cols[f"{stem}_vol21"] = vol.fillna(0.01)
    return pd.DataFrame(cols)


def _no_network(*_args, **_kwargs):
    raise OSError("no network in tests")


def test_app_renders_without_exceptions(monkeypatch):
    # keep the test offline: the live panel degrades to its fallback caption
    monkeypatch.setattr(yahoo, "fetch_intraday_price", _no_network)
    at = AppTest.from_file(APP, default_timeout=120)
    at.session_state["df"] = _fake_df()
    at.run()
    assert not at.exception
    assert not at.warning  # a card that fails into its except shows st.warning
    assert at.main.children  # widgets actually rendered
