"""Model / persistence layer: SQLite via pandas. Swap for Postgres by editing _conn."""

import sqlite3
from pathlib import Path

import pandas as pd

DB_PATH = Path("data/shorting.db")


def _conn() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(exist_ok=True)
    return sqlite3.connect(DB_PATH)


def save(table: str, df: pd.DataFrame) -> None:
    with _conn() as c:
        df.to_sql(table, c, if_exists="replace", index=False)


def load(table: str) -> pd.DataFrame:
    with _conn() as c:
        return pd.read_sql(f"select * from {table}", c)
