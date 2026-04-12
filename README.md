# market-structure

Python library for market structure analysis — swings, trends, support/resistance zones, break/retest/flip signals. Designed for use with [Freqtrade](https://www.freqtrade.io/).

> **Status:** early development. API is unstable and the library is not yet published.

## Documentation

- [Freqtrade Column Reference](docs/freqtrade-columns.md) — all 30 `ms_*` columns projected onto the DataFrame, with dtypes, tier descriptions, and strategy examples.

## Install

```bash
pip install market-structure
```

> The PyPI distribution is `market-structure`; the import name is `market_structure`. The GitHub repo is named `pymarket-structure` for historical reasons.

```python
from market_structure import hello

print(hello())
```

## Development

This project uses [`uv`](https://docs.astral.sh/uv/) for everything: Python version management, virtual environment, dependencies, lockfile, and dev tooling. Install `uv` first (one line — see the link above), then:

```bash
git clone git@github.com:fortunato/pymarket-structure.git
cd pymarket-structure
uv sync                            # creates .venv, installs all deps from uv.lock
uv run pre-commit install          # one-time: enable git pre-commit hooks
uv run just                        # list all available development commands
uv run just check                  # run lint + format-check + type + test
```

That's the entire bootstrap. Everything else is described by the project files themselves:

- `pyproject.toml` — package metadata, dependencies, ruff/pyright/pytest configuration
- `Justfile` — development commands (run `uv run just` for the full menu)
- `.pre-commit-config.yaml` — git hook configuration
- `.editorconfig` — editor formatting rules
- `.python-version` — pinned Python version

### Tech stack

| Tool | Purpose |
|---|---|
| [uv](https://docs.astral.sh/uv/) | Package manager, virtual environment, Python version manager |
| [ruff](https://docs.astral.sh/ruff/) | Linter + formatter (replaces black, isort, flake8, pylint) |
| [pyright](https://microsoft.github.io/pyright/) | Strict static type checker |
| [pytest](https://docs.pytest.org/) + [pytest-cov](https://pytest-cov.readthedocs.io/) | Test runner with branch coverage |
| [pre-commit](https://pre-commit.com/) | Git hook framework |
| [just](https://just.systems/) | Task runner |

### Common tasks

Once `uv sync` has run, the day-to-day workflow is driven by `just` recipes. A few highlights:

```bash
uv run just check              # full CI pipeline (lint + format + type + test)
uv run just test               # run all tests
uv run just test -k swing      # run only tests matching a pattern
uv run just lint               # ruff lint with auto-fix
uv run just fmt                # ruff format
uv run just type               # pyright (strict mode)
uv run just hooks              # run all pre-commit hooks against every file
```

> If your shell has the project's venv activated (`source .venv/bin/activate`), you can drop the `uv run` prefix and just type `just check`, `pytest`, `ruff check`, etc. directly. Most editors (PyCharm, VS Code) auto-activate the venv once the interpreter is configured.

## License

See [LICENSE](LICENSE).
