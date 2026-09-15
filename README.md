# we_love_shorting
A small end-to-end ML project: do **market prices** predict the **tone of the
news**? We pull several daily prices (major stock indices plus precious metals
and crude oil) and daily news sentiment, store them in a database, train a
model, and show the result in a Streamlit dashboard.

> **Hypothesis:** the day's market prices — stock indices (S&P 500, OMX
> Stockholm 30, EURO STOXX 50), precious metals (gold, silver, copper) and
> crude oil — carry information about how negative the news is. By default the
> model predicts the day's news **tone** from the price streams (a low
> predicted tone = bearish sentiment = a shorting cue), but in the dashboard
> **any numeric stream can be the target** and the rest become the features.

---

## How it fits together

Each data source is its own process in its own module. They stay **separate**
right up until `features.py`, where they meet and join on the date.

```
  sources/gdelt.py   GDELT ───► fetch_tone()        ─┐  (default target: news tone)
  sources/yahoo.py   Yahoo ───► fetch_prices()       ├─► db.py ─► features.py ─► signal_model.py ─► app.py
                     (^GSPC, ^OMX, ^STOXX50E)         │  SQLite    join on date   train + predict    Streamlit
                     (GC=F, SI=F, HG=F) fetch_prices()┤           = the meeting                       dashboard
                     (CL=F)             fetch_prices()┘
```

The price streams are fixed in `features.TICKERS`; the dashboard no longer asks
for tickers. Which stream is the **target** and which are **features** is picked
at predict time in the UI.

The code is split into small layers so anyone can work on one piece without
breaking the others:

| File | Layer | Responsibility |
|------|-------|----------------|
| `sources/gdelt.py` | data | Fetch news tone (GDELT) — the target |
| `sources/yahoo.py` | data | Fetch a price for any ticker (indices, metals, oil) |
| `db.py` | database | Accumulating SQLite archive: idempotent `upsert` + `last_date` |
| `features.py` | join | Where the streams **meet**: join on date, pick features + target |
| `signal_model.py` | ML | Train the model, predict the target |
| `controller.py` | glue | Runs the whole flow: fetch → store → join → train → predict |
| `app.py` | UI | Streamlit dashboard (what you actually click on) |

---

## Where the data comes from (separate streams)

The streams are fetched **independently** and only joined together in
`features.py`, on the date. Each stream is its own table in the database.

### News tone — GDELT (the default target)
[GDELT](https://gdeltproject.org/) monitors news media worldwide and computes an
average **"tone"** score for articles matching a search term. We fetch it over
plain HTTP (Python's built-in `urllib`, no library needed) in
`sources.gdelt.fetch_tone()`.

- **`tone`** = sentiment of the day's news. Roughly ranges −10 (very negative) to
  +10 (very positive); 0 is neutral. This is the default prediction target.
- GDELT rate-limits per IP, so we retry with backoff on HTTP 429.
- GDELT DOC only serves a rolling window (roughly 2017 onward), so the tone
  table starts later than the multi-year price history — the join clips to the
  overlapping span.

### Prices — Yahoo Finance (`yfinance`)
The price streams come from **Yahoo Finance** via the [`yfinance`](https://pypi.org/project/yfinance/)
library (free, pip-installable, no API key). One function serves every stream —
they differ only by ticker. The set is fixed in `features.TICKERS`:

| Column | Ticker | Stream |
|--------|--------|--------|
| `sp500_close` | `^GSPC` | S&P 500 (US) |
| `omx30_close` | `^OMX` | OMX Stockholm 30 (Sweden) |
| `eurostoxx_close` | `^STOXX50E` | EURO STOXX 50 (Europe) |
| `gold_close` | `GC=F` | Gold futures |
| `silver_close` | `SI=F` | Silver futures |
| `copper_close` | `HG=F` | Copper futures |
| `oil_close` | `CL=F` | WTI crude oil futures |

We keep only the `date` and closing-price columns from each. To add or change a
stream, edit `features.TICKERS` (a new stream automatically becomes a selectable
feature/target and gets its own DB table).

---

## The database connection

We use **SQLite** — a zero-setup database in a single file, `data/shorting.db`.
No server, no credentials. All DB access goes through `db.py`, which is tiny on
purpose.

- One table per stream: **`tone`** plus one per ticker (`sp500`, `omx30`,
  `eurostoxx`, `gold`, `silver`, `copper`, `oil`), each `(date, <stem>_close)`.
- The DB is an **accumulating archive**, not a throwaway cache. Instead of
  refetching everything on every run, we **backfill** ~5 years once, then **top
  up** from the last stored date to today. `db.upsert()` is idempotent: dates
  are normalised to ISO text and deduped, so refetching an overlapping day
  overwrites it rather than duplicating.
- `controller.run()` reads only the DB (no fetch), so training is decoupled from
  the network — the DB is the single source of truth.
- Want a real server DB later? Swap only `_conn()` (e.g. to Postgres). The rest
  of the code doesn't change.

The `.db` file is **git-ignored** — data is rebuilt via the fill buttons (or the
`python -m we_love_shorting.controller` backfill), so we never commit it.

---

## Project structure

```
src/we_love_shorting/
  sources/
    gdelt.py        # fetch GDELT news tone (the target)
    yahoo.py        # fetch a price for any ticker (indices, metals, oil)
  db.py             # accumulating SQLite archive: upsert / last_date / load
  features.py       # the meeting point: join streams on date, features + target
  signal_model.py   # train, predict the target
  controller.py     # orchestrates the flow
app.py                 # Streamlit dashboard
tests/                 # pytest
requirements.txt       # pinned runtime dependencies
requirements-dev.txt   # pinned dev tools (ruff, pytest, mypy)
pyproject.toml         # packaging; dependencies read from the files above
data/                  # SQLite DB + saved model (git-ignored)
```

---

## Run it

```bash
git clone <repo-url> && cd we_love_shorting
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"     # editable install; pulls runtime + dev deps

streamlit run app.py        # opens the dashboard in your browser
pytest                      # run the tests
```

Just want to run the app, no dev tools? `pip install -e .` (or `pip install -r requirements.txt`) is enough.

In the dashboard:

1. **Första fyllning (5 år)** — one-time backfill of ~5 years of history into
   the DB. (Or run it headless: `python -m we_love_shorting.controller`.)
2. **Fyll på till idag** — later, top up from the last stored date to today.
3. **Ladda data** — read the DB and build the feature table (no fetch).
4. Pick the **target** and **features**, and you get a chart of the actual value
   vs. the model's prediction.

The only free-text knob left is the GDELT query (e.g. `recession`); the price
tickers are fixed in `features.TICKERS`. Rate-limited sources (GDELT's HTTP 429)
auto-retry behind a short countdown; other errors surface immediately.

---

## Known limitations (it's a proof of concept)

- The model currently trains and predicts on the same data, so the accuracy
  numbers aren't trustworthy yet — the goal right now is to prove the **flow**
  works end to end. A proper time-based train/test split is the next step.
- Prices and tone are used **same-day** (contemporaneous), so this measures
  association, not a forecast. Lagging the features is the next step.
- News tone is at index/term level, not per-company — mapping tone to individual
  tickers is GDELT's weak spot.
