"""Model / persistence layer: SQLite via pandas. Swap for Postgres by editing _conn."""

import sqlite3
from contextlib import closing
from pathlib import Path

import pandas as pd

DB_PATH = Path("data/shorting.db")
_TABLES = ("tone", "spy", "metal", "oil")


def _conn() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(exist_ok=True)
    return sqlite3.connect(DB_PATH)


def _table(name: str) -> str:
    if name not in _TABLES:  # no interpolating unvalidated names into SQL
        raise ValueError(f"unknown table {name!r}; expected one of {_TABLES}")
    return name


def upsert(table: str, df: pd.DataFrame) -> None:
    """Append rows, keeping the newest row per date. Idempotent top-up: refetching
    an already-stored date overwrites it instead of duplicating."""
    if df.empty:
        return  # nothing to add (e.g. an incremental fetch that's already current)
    name = _table(table)  # validate once; used for both read and write
    with closing(_conn()) as c, c:
        df = df.copy()
        # Fresh fetches carry datetime.date objects; SQLite stores dates as ISO
        # text. Normalise to str so concat/dedup/sort never mix the two types.
        df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
        try:
            old = pd.read_sql(f"select * from {name}", c)
            df = pd.concat([old, df]).drop_duplicates("date", keep="last")
        except pd.errors.DatabaseError:
            pass  # table doesn't exist yet -> first write
        # ponytail: full-table rewrite per top-up; switch to INSERT OR REPLACE
        # if a table ever reaches six-figure row counts.
        df.sort_values("date").to_sql(name, c, if_exists="replace", index=False)


def last_date(table: str) -> str | None:
    """Newest stored ISO date (YYYY-MM-DD sorts correctly as text), or None if empty."""
    with closing(_conn()) as c:
        try:
            m = pd.read_sql(f"select max(date) m from {_table(table)}", c).m[0]
        except pd.errors.DatabaseError:
            return None
    return m  # None when the table exists but holds no rows


def load(table: str) -> pd.DataFrame:
    with closing(_conn()) as c:
        return pd.read_sql(f"select * from {_table(table)}", c)
