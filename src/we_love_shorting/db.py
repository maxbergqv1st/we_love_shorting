"""Model / persistence layer: SQLite via pandas. Swap for Postgres by editing _conn."""

import sqlite3
from contextlib import closing
from pathlib import Path

import pandas as pd

DB_PATH = Path("data/shorting.db")
_TABLES = ("tone", "spy", "metal")


def _conn() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(exist_ok=True)
    return sqlite3.connect(DB_PATH)


def _table(name: str) -> str:
    if name not in _TABLES:  # no interpolating unvalidated names into SQL
        raise ValueError(f"unknown table {name!r}; expected one of {_TABLES}")
    return name


def save(table: str, df: pd.DataFrame) -> None:
    with closing(_conn()) as c, c:  # c: commits, closing() closes
        df.to_sql(_table(table), c, if_exists="replace", index=False)


def load(table: str) -> pd.DataFrame:
    with closing(_conn()) as c:
        return pd.read_sql(f"select * from {_table(table)}", c)
