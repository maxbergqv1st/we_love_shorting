---
name: streamlit
description: Streamlit best practices — use when building or editing the Streamlit app, pages, widgets, session state, or caching in this repo. Source: https://medium.com/@jashuamrita360/best-practices-for-streamlit-development-structuring-code-and-managing-session-state-0bdcfb91a745
---

# Streamlit best practices

## Structure — separate UI from logic
- Data loading, features, and model live in `src/` as plain functions.
- `app.py` only wires widgets to those functions. Multi-page via `pages/`.
- The whole script reruns top-to-bottom on every interaction — keep it cheap.

## Session state
- Initialize each key once: `if "k" not in st.session_state: st.session_state.k = ...`.
- Use widget `key=` and `on_change=`/`on_click=` callbacks to mutate state.
- Don't recompute persistent values every rerun — read them from session state.

## Caching (avoid recomputation)
- `@st.cache_data` for data/DataFrames and pure computations.
- `@st.cache_resource` for models/connections — load the joblib Pipeline ONCE.
- Gate expensive work behind `st.button` or `st.form` (form batches inputs, one rerun).

## Config & secrets
- `.streamlit/config.toml` for theme/server settings.
- Secrets in `.streamlit/secrets.toml` (git-ignored) via `st.secrets`. Never hardcode.
