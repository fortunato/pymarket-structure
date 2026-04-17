# Frontend

This project was generated using [Angular CLI](https://github.com/angular/angular-cli) version 21.2.7.

## Development server

To start a local development server, run:

```bash
ng serve
```

Once the server is running, open your browser and navigate to `http://localhost:4200/`. The application will automatically reload whenever you modify any of the source files.

## Code scaffolding

Angular CLI includes powerful code scaffolding tools. To generate a new component, run:

```bash
ng generate component component-name
```

For a complete list of available schematics (such as `components`, `directives`, or `pipes`), run:

```bash
ng generate --help
```

## Building

To build the project run:

```bash
ng build
```

This will compile your project and store the build artifacts in the `dist/` directory. By default, the production build optimizes your application for performance and speed.

## Running unit tests

To execute unit tests with the [Vitest](https://vitest.dev/) test runner, use the following command:

```bash
ng test
```

## Running end-to-end tests

For end-to-end (e2e) testing, run:

```bash
ng e2e
```

Angular CLI does not come with an end-to-end testing framework by default. You can choose one that suits your needs.

## Chart data fixtures

The viewer reads pre-enriched OHLCV + backtest-trade JSON from
`src/assets/data/`. That directory is **generated** by
`scripts/export_chart_data.py`, not committed by hand.

The script loops every OHLCV file in `../refs/freqtrade/ohlcv/*-4h.json`,
computes TSI, runs `attach_market_structure()` from the Python library,
and writes one enriched JSON per pair. It also transforms every backtest
result in `../refs/freqtrade/backtest_results/*-trades.json` into a
frontend-friendly shape under `src/assets/data/trades/`.

### How to run

From the `pymarket-structure/` directory (the Python project root, so
`uv` picks up the library):

```bash
uv run python frontend/scripts/export_chart_data.py
```

Outputs: `src/assets/data/<PAIR>-4h.json` (one per OHLCV input) and
`src/assets/data/trades/<Strategy>-<PAIR>-4h.json` (one per backtest
input).

### When to re-run

Re-run whenever any of the following changes, otherwise the chart will
display stale zone / wave / trade data:

- The market-structure library behaviour (e.g. zone qualification rules,
  break/retest/flip logic, new `ms_*` columns).
- The TSI computation parameters in `_compute_tsi()` at the top of the
  script.
- The input OHLCV files under `../refs/freqtrade/ohlcv/`.
- The backtest result files under `../refs/freqtrade/backtest_results/`.

Running `ng serve` / `ng build` will NOT regenerate these files — you
must invoke the Python script explicitly.

## Additional Resources

For more information on using the Angular CLI, including detailed command references, visit the [Angular CLI Overview and Command Reference](https://angular.dev/tools/cli) page.
