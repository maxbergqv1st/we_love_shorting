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

## Techniques & tools in use (established patterns — reuse before adding)
- **Streamlit API (current, not deprecated):** `width="stretch"` on `st.dataframe`, never `use_container_width=True`. `st.segmented_control` (with `or <default>` fallback for the deselect case) over `st.radio` for span pickers. Live panel is an `@st.fragment(run_every="1m")`.
- **Session-state persistence:** persist widget picks with `key=` + `st.session_state.setdefault(...)`, then prune invalid entries each rerun — don't pass `default=` for state that must survive reruns.
- **Derive labels/config from one source:** `LABELS` is built by looping `features.TICKERS`, not hand-maintained; a new ticker auto-gets labels. Group a stream's columns via `features.stream_of(col)` (`sp500_close`/`sp500_ret` → `sp500`).
- **Baseline:** `naive_baseline` uses `test_target.shift(1)`, seeding row 0 from the last train value — no manual concat.
- **AI Q&A:** `we_love_shorting.chatbot` (OpenRouter free model); key from `st.secrets["OPENROUTER_API_KEY"]`, absent → graceful in-UI error.
- **mypy:** `ignore_missing_imports = true` in `pyproject.toml` (pandas/sklearn/joblib/yfinance ship no stubs). Run `mypy src app.py` before done.

## Domain guidance (skills auto-load when relevant)
- General Python style/structure → `python-best-practices` skill.
- ML: models, pipelines, preprocessing, evaluation → `sklearn` skill.
- Streamlit UI, session state, caching → `streamlit` skill.
