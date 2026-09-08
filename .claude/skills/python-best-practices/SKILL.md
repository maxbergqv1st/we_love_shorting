---
name: python-best-practices
description: General Python best practices — use when writing, structuring, refactoring, or reviewing any Python module, function, package layout, or test in this repo. Source: https://realpython.com/tutorials/best-practices/
---

# Python best practices

## Structure & packaging
- `src/` layout: importable package under `src/`, tests under `tests/`.
- `pyproject.toml` is the single source of config (deps, ruff, pytest, mypy). Pin versions.
- Small pure functions. Prefer dependency injection (pass args) over module globals.
- Guard entrypoints with `if __name__ == "__main__":`.

## Style
- PEP 8 via `ruff format` + `ruff check`. Don't hand-format.
- Type hints on public functions; keep `mypy` clean.
- PEP 257 docstrings on public API. f-strings for formatting.

## Idioms
- EAFP (try/except) over LBYL (pre-checking) for expected-to-succeed calls.
- Comprehensions/generators over manual loops that build lists.
- `pathlib.Path` over `os.path`. Context managers (`with`) for files/resources.
- `dataclasses` for records, `enum.Enum` for fixed sets.
- Never use a mutable default arg (`def f(x=[])`) — use `None` + create inside.

## Errors & logging
- Raise/catch specific exceptions, not bare `except:`.
- `logging` (module-level logger), never `print`, in library code.

## Testing
- `pytest`, arrange-act-assert. `@pytest.fixture` for setup, `@pytest.mark.parametrize` for cases.
- Test behavior/edges, not implementation details.
