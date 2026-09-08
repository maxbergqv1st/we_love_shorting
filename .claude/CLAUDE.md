# we_love_shorting — conventions

Python 3.14 project (`.venv` present). ML shorting-signal model + Streamlit dashboard.

## Shared conventions
- Layout: `src/` package + `tests/`. `pyproject.toml` is the single source of config.
- Tooling: `ruff` (lint + format), `pytest`, type hints checked with `mypy`.
- Pin dependencies. `logging`, never `print`, in library code.
- `.streamlit/secrets.toml` and real data files stay git-ignored.

## Domain guidance (skills auto-load when relevant)
- General Python style/structure → `python-best-practices` skill.
- ML: models, pipelines, preprocessing, evaluation → `sklearn` skill.
- Streamlit UI, session state, caching → `streamlit` skill.
