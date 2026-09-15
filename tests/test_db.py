import datetime as dt

import pandas as pd
import pytest

from we_love_shorting import db


@pytest.fixture(autouse=True)
def _tmp_db(tmp_path, monkeypatch):
    # _conn reads DB_PATH at call time, so pointing it at a tmp file isolates
    # every test from the real data/shorting.db.
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "test.db")


def test_upsert_dedups_refetched_dates_and_keeps_newest():
    day = dt.date(2024, 1, 1)
    later = dt.date(2024, 1, 2)
    db.upsert("tone", pd.DataFrame({"date": [day, later], "tone": [1.0, 2.0]}))
    # Refetch overlaps day 1 (new value) and adds day 3 -> dedup, keep="last".
    db.upsert(
        "tone",
        pd.DataFrame({"date": [day, dt.date(2024, 1, 3)], "tone": [9.0, 3.0]}),
    )

    out = db.load("tone").sort_values("date").reset_index(drop=True)

    assert out["date"].tolist() == ["2024-01-01", "2024-01-02", "2024-01-03"]
    assert out["tone"].tolist() == [9.0, 2.0, 3.0]  # day 1 overwritten, not doubled
    assert db.last_date("tone") == "2024-01-03"


def test_upsert_empty_frame_is_noop():
    db.upsert("tone", pd.DataFrame(columns=["date", "tone"]))
    assert db.last_date("tone") is None  # table never created
