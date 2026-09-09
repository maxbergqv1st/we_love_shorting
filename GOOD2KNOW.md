Add end-to-end GDELT tone -> short-signal PoC
MVC-ish layers: sources (GDELT tone + yfinance prices), db (SQLite),
signal_model (sklearn), controller (orchestration), Streamlit app.
Retry/backoff on GDELT 429 + fallback to cached DB data.

# Ruff
    Typisk användning:
    - ruff check . — lintar koden
    - ruff check --fix . — lintar och fixar det som går automatiskt
    - ruff format . — formaterar koden