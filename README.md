# market-structure

[![CI](https://github.com/fortunato-geelhoed/pymarket-structure/actions/workflows/ci.yml/badge.svg)](https://github.com/fortunato-geelhoed/pymarket-structure/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/market-structure)](https://pypi.org/project/market-structure/)

Python library for market structure analysis — swings, trends, support/resistance zones, break/retest/flip signals. Designed for use with [Freqtrade](https://www.freqtrade.io/).

> **Status:** early development. API is unstable.

## Backtest: market structure as a strategy filter

To validate the library, we ran an A/B comparison using a TSI signal-line crossover
strategy on 4h Binance futures (BTC, ETH, SOL, XRP, LTC) over 360 days in a **-14.91%
market**. The only difference: the "With MS" variant filters entries and exits through
the `ms_*` columns; the "Without" variant uses the same TSI signal and risk parameters
but skips market structure entirely.

| Metric | With MS filter | Without MS filter |
|---|---|---|
| Total profit | **+39.85%** | +22.59% |
| Profit factor | **1.50** | 1.21 |
| Sharpe | **0.94** | 0.58 |
| Max drawdown | **17.76%** | 21.75% |
| Trades | 164 | 208 |
| Win rate | 36.0% | 31.7% |
| Stop-loss hits | 36 | 46 |

The filter blocked 44 low-quality entries (mostly would-be stop-outs), nearly doubling
profit while cutting drawdown by ~4 percentage points. Backtest data and
[results for all three strategies](refs/freqtrade/README.md) live in
[`refs/freqtrade/`](refs/freqtrade/).

> **Disclaimer:** This is educational software, not financial advice. Past backtest performance does not guarantee future results. Parameters are curve-fit to a specific historical window. Do not trade real capital based on these examples without independent validation and risk assessment.

## Documentation

- [Freqtrade Column Reference](docs/freqtrade-columns.md) — all 63 `ms_*` columns projected onto the DataFrame, with dtypes, tier descriptions, and strategy examples.

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
