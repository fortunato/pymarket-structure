# market-structure

[![CI](https://github.com/fortunato/pymarket-structure/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/fortunato/pymarket-structure/actions/workflows/ci.yml)
[![Deploy to GitHub Pages](https://github.com/fortunato/pymarket-structure/actions/workflows/pages.yml/badge.svg)](https://github.com/fortunato/pymarket-structure/actions/workflows/pages.yml)
[![PyPI](https://img.shields.io/pypi/v/market-structure)](https://pypi.org/project/market-structure/)
[![Python versions](https://img.shields.io/python/required-version-toml?tomlFilePath=https%3A%2F%2Fraw.githubusercontent.com%2Ffortunato%2Fpymarket-structure%2Fmain%2Fpyproject.toml)](https://github.com/fortunato/pymarket-structure)
[![License](https://img.shields.io/github/license/fortunato/pymarket-structure)](LICENSE)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

Python library for market structure analysis — swings, trends, support/resistance zones, break/retest/flip signals. Works with any OHLCV DataFrame. Includes a [Freqtrade](https://www.freqtrade.io/) integration.

**[Live Demo](https://fortunato.github.io/pymarket-structure/)** — interactive chart viewer showing market structure overlays on real price data.

![Market Structure Viewer — SOL/USDT 4h showing support/resistance zones, structure break, bearish divergence, and wave metrics](https://raw.githubusercontent.com/fortunato/pymarket-structure/main/docs/images/market-structure-viewer.png)

## Quick start

```python
import pandas as pd
from market_structure.hydrate import hydrate
from market_structure.tsi import compute_tsi
from market_structure.atr import _compute_atr

# df: any OHLCV DataFrame with open, high, low, close, volume columns
df["tsi_hist"] = compute_tsi(df["close"])["tsi_histogram"]
helper = hydrate(df, histogram_key="tsi_hist")

atr = _compute_atr(df["high"].to_numpy(), df["low"].to_numpy(),
                    df["close"].to_numpy(), period=14)

for z in helper.get_support_zones(atr_arr=atr):
    body_low, body_high = z.range       # body-anchored (accepted price)
    wick_low, wick_high = z.wick_range   # wick extrema (for stop placement)
    print(f"{z.anchor_wave_id}  body=({body_low:.2f}, {body_high:.2f})  "
          f"wick=({wick_low:.2f}, {wick_high:.2f})  double={z.is_double}")
```

### Freqtrade integration

For Freqtrade strategies, a single call projects all 67 `ms_*` columns onto the DataFrame:

```python
from market_structure.freqtrade import attach_market_structure

df = attach_market_structure(df, histogram_key="tsi_hist")
# df now has ms_support_zone_low, ms_wave_side, ms_is_trending_up, etc.
```

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
[results for all three strategies](https://github.com/fortunato/pymarket-structure/blob/main/refs/freqtrade/README.md) live in
[`refs/freqtrade/`](https://github.com/fortunato/pymarket-structure/tree/main/refs/freqtrade/).

> **Disclaimer:** This is educational software, not financial advice. Past backtest performance does not guarantee future results. Parameters are curve-fit to a specific historical window. Do not trade real capital based on these examples without independent validation and risk assessment.

## Documentation

- [Column Reference](https://github.com/fortunato/pymarket-structure/blob/main/docs/freqtrade-columns.md) — all 67 `ms_*` columns projected onto the DataFrame, with dtypes, tier descriptions, and strategy examples.

## Install

```bash
pip install market-structure
```

> The PyPI distribution is `market-structure`; the import name is `market_structure`. The GitHub repo is named `pymarket-structure`.

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

## Support

If this library saves you time or helps your trading, consider buying me a coffee.

[![Buy Me A Coffee](https://img.shields.io/badge/Buy_Me_A_Coffee-FFDD00?style=for-the-badge&logo=buy-me-a-coffee&logoColor=black)](https://buymeacoffee.com/fortunato)

## License

See [LICENSE](https://github.com/fortunato/pymarket-structure/blob/main/LICENSE).
