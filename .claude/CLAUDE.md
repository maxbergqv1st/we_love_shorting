# we_love_shorting — conventions

Python 3.14 project (`.venv` present). ML shorting-signal model + Streamlit dashboard.

## Base rule: NO REINVENTION
Before writing new code, always look to **upgrade or tweak existing code** first. Grep the repo for a helper, pattern, or module that already does the job and extend it — reimplementing what already lives here (or in the stdlib / an installed dependency) is not allowed. New code is the last resort, after reuse has been ruled out.

## Project brief (graded group assignment)
Complete AI/ML project end-to-end, three required parts:
1. Store the data in a **database** (backend).
2. **AI/ML modelling** in Python.
3. **Streamlit** frontend (Flask/Django allowed; we use Streamlit). Bonus: deploy to a public URL.

- Dataset is free choice (Kaggle etc.) but must be **approved by Antonio**. Show him a PoC first — it must prove the *flow* works end-to-end, not the details.
- **Milestone v.39 (week of 2026-09-21):** present the work to the other groups.
- Deliverables, submitted individually on LearnPoint: ~3-page technical report (PDF: background, main results, short tech spec, team/Git evaluation) + **public** GitHub repo with a clear README (CV-quality).
- Agree on the Git/GitHub workflow up front. Multiple people code here — keep changes reviewable. All aids/AI allowed, but the work is ours and we must understand everything submitted.

## Shared conventions
- Layout: `src/` package + `tests/`. Dependencies pinned in `requirements.txt`.
- Tooling: `ruff` (lint + format), `pytest`, type hints checked with `mypy`; per-tool config added when a default falls short.
- **Always ruff before calling code done:** run `ruff check --fix .` then `ruff format .` on any Python you write or edit, and confirm it passes, before reporting the work finished. Not optional.
- Pin dependencies. `logging`, never `print`, in library code.
- `.streamlit/secrets.toml` and real data files stay git-ignored.

## Architecture (layers, top-down)
`app.py` (Streamlit view, workspace of `card()`s, sidebar = settings + AI chat) → `analysis.py` (strategy pattern: `AnalysisMode` subclasses return render-agnostic `Panel`s — **no Streamlit in the package**, unit-testable) → `controller.py` (wire sources → DB → features → models → results) → `signal_model.py` / `evaluation.py` / `features.py` → `db.py` (SQLite) + `sources/` (yahoo, gdelt). Models are **interactive-only, never persisted** — no joblib, no stale-model-on-disk state.

## Techniques & tools in use (established patterns — reuse before adding)
- **Analysis modes = strategy pattern (`analysis.py`).** Predict-a-target modes (Regression, Riktning) live in the `MODES` registry; add a mode = one entry. Target-independent views (asset grouping) are standalone functions, not modes. Modes take a **`params: dict` of hyperparameters** and forward only the keys they understand via `_kw(params, *keys)`; controller/model functions keep explicitly typed args with defaults.
- **Hyperparameter registry (`app.py HYPERPARAMS`).** Each tunable knob is one `HyperParam(key, label, default, options)` entry; `key` == the controller argument name so it forwards straight through. UI = multiselect toggle (several at once) → slider (numeric) or selectbox (categorical). Adding a knob = one registry row + the controller default. Current: `alpha` (Ridge), `c` (logistic reg), `flat_frac` (dead-zone), `test_frac`, `n_groups`, `linkage`.
- **Models (`signal_model`):** regression `train`/`predict` with `model_kind` — "linear" (OLS; `alpha>0` → `StandardScaler`+Ridge — Ridge/logistic are NOT scale-invariant, OLS is) or "forest" (`RandomForestRegressor`, `random_state` pinned; knobs `n_estimators`/`max_depth`); `weights(model, features)` extracts coefficients (linear family) or feature importances (forest) — the evaluation carries them (`EvaluationResult.weights`) so the weight view costs **zero extra fits**. Also: 3-class direction (`train_direction`, labels from `evaluation.direction_labels` on the target's **move** — `_ret` as-is, levels via `.diff()`, never the raw level value) and asset grouping (`cluster_assets`: AgglomerativeClustering on `1 − return-correlation`; correlate **returns**, never trending levels — levels cluster by era).
- **Charts:** all multi-series lines go through app's `line_chart()` — Altair **without pan/zoom** (native `st.line_chart` hijacks the mouse wheel) with tooltips. Categorical series → scatter/heatmap, never a line. Correlation heatmap = `Styler.map(_corr_cell)` (manual red/green rgba — **not** `background_gradient`, which needs matplotlib). Prediction-error line centred on 0 beside overlapping actual/pred lines. Many small charts → behind an opt-in `st.toggle`.
- **Streamlit API (current, not deprecated):** `width="stretch"`, never `use_container_width`. `st.segmented_control` (with `or <default>` for deselect) over radio. `@st.dialog` for rare admin flows (Setup). `@st.fragment(run_every="1m")` for the live panel. `card()` contextmanager = bordered container + heading. Material Symbols icons over emojis.
- **Selection is stream-level, above the chart:** the target picker offers raw columns only (tone/`_close`/`_ret` — predicting a moving average is meaningless) and features are toggled as **stream pills** (multi) in a card above the main view — toggling Silver includes all its columns (close/ret/ma/vol); the target's own stream is auto-excluded. Don't reintroduce a per-column feature multiselect (it was a wall of 5 near-identical chips per asset).
- **Widget state:** per-dependency keys over manual pruning — e.g. the stream pills use `key=f"streams::{target}"` + `default=` so each target keeps its own toggles and options always match. Never mutate a widget's own session key mid-run. Give hyperparam widgets explicit `key=`s so picks survive reruns.
- **Auto-eval:** no "run" button — `cached_eval` (`st.cache_data`) keyed on mode/df/features/target/`tuple(sorted(params.items()))` recomputes only when inputs change.
- **Headless verification:** `streamlit.testing.v1.AppTest` — load `st.session_state['df']`, run, assert no exceptions/warnings and widgets render. This catches real bugs (it caught an unconditional `st.stop()` and a matplotlib crash). Do this before claiming UI work done; visual look still needs a human.
- **Forecast horizon:** `controller.shift_target(df, target, horizon)` turns the nowcast frame into a forecast frame (row t keeps its features, target becomes t+horizon's actual; last rows dropped) — shift happens **before** any split so nothing leaks, and every downstream piece (baseline, direction, live panel) works unchanged. The app's Horisont picker: Idag (0) / Imorgon (1). **The shift runs over TRADING days** (closed rows dropped first): the raw frame has one row per tone date, so a calendar shift hands Friday rows Saturday's carried-forward close — their own value — poisoning ~31% of rows with fake "tomorrow = today" examples. The live panel takes both frames: the shifted one to train, the raw one for live quote/fallback values (the shifted tail is stale).
- **Derived features:** each stream gets `_ma5`/`_ma21` (rolling mean, min_periods=1) and `_vol21` (rolling std of returns, warmup backfilled) computed pre-merge on the stream's own calendar in `build_features`. Suffix set lives in `features.STREAM_SUFFIXES`; the UI defaults to raw `_close`/`_ret` only (derived are opt-in) so the picker isn't a wall of chips.
- **Derive labels/config from one source:** `LABELS` loops `features.TICKERS` × `_SUFFIX_LABELS`; group a stream's columns via `features.stream_of(col)` (split on first `_` — stems never contain underscores).
- **Session-only streams (ticker search):** `yahoo.search_tickers` (yf.Search) + a 5y fetch at add-time; frames live in `st.session_state["extra_streams"]` and flow into `controller.get_data(extra_prices=...)` → `build_features` like any fixed stream — **never written to the DB**, so the table whitelist stays intact. Stems are sanitised to bare alnum (stream_of splits on `_`). Known limits: an extra stream truncates the table to its own history (build_features drops rows any stream lacks) and gets no live intraday quotes (`_fetch_live_closes` maps via features.TICKERS only). `build_features` normalises each frame's `date` to datetime up front — DB frames carry ISO strings, fresh fetches date objects.
- **Evaluation views stay tight:** metrics-vs-baseline table + ONE residual view (histogram). Don't re-add more residual charts — they were cut deliberately as three views of the same information.
- **Baseline:** `naive_baseline` uses `test_target.shift(1)` seeded from last train value; direction baseline = majority class.
- **AI Q&A:** `chatbot.py` (OpenRouter free model + fallback + retry); key injected by caller — module stays Streamlit-free.
- **mypy:** `ignore_missing_imports = true` in `pyproject.toml`. Run `mypy src app.py`; clear `.mypy_cache` if results look stale/nonsensical.

## Security conventions
- SQL: table names validated against the `_TABLES` whitelist (`db._table`) before any f-string interpolation; values go through pandas/params. Never interpolate unvalidated input into SQL.
- All `urllib.request.urlopen` calls have explicit `timeout=` (gdelt 30s, chatbot 60s) — no hang-forever network calls.
- Secrets only via `st.secrets` / `.streamlit/secrets.toml` (git-ignored, as are `data/`, `*.joblib`). Missing key → graceful in-UI error, never a crash or a hardcoded fallback.
- No `eval`/`exec`/`shell=True`/`unsafe_allow_html` anywhere.
- Outbound URLs are fixed constants; query params built with `urllib.parse.urlencode`.

## Domain guidance (skills auto-load when relevant)
- General Python style/structure → `python-best-practices` skill.
- ML: models, pipelines, preprocessing, evaluation → `sklearn` skill.
- Streamlit UI, session state, caching → `streamlit` skill.
