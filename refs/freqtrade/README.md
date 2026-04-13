# Freqtrade example strategies

Example Freqtrade strategies that consume the `market_structure` library, backtested on 4h Binance futures (BTC, ETH, SOL, XRP, LTC) over 360 days (2025-04-17 to 2026-04-12) in a **-14.91% market**.

Trade results in `backtest_results/` and OHLCV data in `ohlcv/` are used by the frontend to overlay trades on charts.

## Strategies

### MsFilterV6 — TSI crossover + market structure filter

TSI histogram zero-cross entry (long and short) filtered by market structure trend state. Longs are blocked when `ms_is_trending_down`, shorts when `ms_is_trending_up`. RSI guard prevents entries into overbought/oversold conditions.

Parameters loaded from `MsFilterV6.json` (hyperopt-optimized for the backtest window).

### MsFilterV6NoMs — TSI crossover, no market structure (control)

Identical entry signal to MsFilterV6 with the same hyperopt-optimized thresholds, but all `ms_*` filters removed. Exists to measure the value the market structure filter layer adds — compare its trades against MsFilterV6 on the same data.

Parameters are hardcoded in the strategy file to match MsFilterV6.json.

### MsSupportResistanceV1 — S&R proximity + wave confirmation

Bidirectional strategy (long and short) driven by proximity to structural support/resistance levels. Longs enter when price is near the support zone or last confirmed bottom and the wave has turned up. Shorts enter near resistance or on a confirmed lower-high with the wave turning down. Exits target the opposing S&R level, with divergence and structural-break protective exits.

Parameters loaded from `MsSupportResistanceV1.json` (hyperopt-optimized for the backtest window).

## Backtest results

All strategies backtested on 5 pairs × 4h, 2025-04-17 to 2026-04-12 (~360 days after 30-bar startup). 1,000 USDT starting balance, 300 USDT stake per trade, max 3 concurrent positions.

| Strategy | Trades | Win% | Total Profit | Profit Factor | Sharpe | Max Drawdown |
|---|---|---|---|---|---|---|
| MsFilterV6 | 164 | 36.0% | +39.85% | 1.50 | 0.94 | 17.76% |
| MsFilterV6NoMs | 208 | 31.7% | +22.59% | 1.21 | 0.58 | 21.75% |
| MsSupportResistanceV1 | 404 | 42.6% | +70.32% | 1.38 | 2.67 | 18.10% |

The MS filter blocked 44 low-quality entries (mostly would-be stop-outs), nearly doubling profit while cutting drawdown by ~4 percentage points.

Parameters are hyperopt-optimized for this specific window. These results demonstrate the library integration, not production-ready strategy performance.

> **Disclaimer:** This is educational software, not financial advice. Past backtest performance does not guarantee future results. Parameters are curve-fit to a specific historical window. Do not trade real capital based on these examples without independent validation and risk assessment.

## Files

```
config-futures.json                          Freqtrade config for 5-pair futures backtesting
config-fixture.json                          Freqtrade config for fixture-window backtesting (tests)
strategies/
  MsFilterV6.py                              TSI crossover + MS trend filter
  MsFilterV6.json                            hyperopt params
  MsFilterV6NoMs.py                          TSI crossover, no MS (control)
  MsSupportResistanceV1.py                   S&R proximity strategy
  MsSupportResistanceV1.json                 hyperopt params
backtest_results/
  {Strategy}-{PAIR}-4h-trades.json           trade records per strategy × pair (15 files)
ohlcv/
  {PAIR}-4h.json                             OHLCV candle data per pair (5 files)
```

## Reproducing the backtests

The config uses 5 Binance futures pairs. The steps below set up Freqtrade and run the backtest.

1. Set up Freqtrade with the `market-structure` package:
   ```bash
   git clone https://github.com/freqtrade/freqtrade.git
   cd freqtrade
   python -m venv .venv && source .venv/bin/activate
   pip install -e .
   pip install -e /path/to/pymarket-structure
   ```

2. Download the data (or convert from the OHLCV JSON files):
   ```bash
   freqtrade download-data \
     --config /path/to/pymarket-structure/refs/freqtrade/config-futures.json \
     --timerange 20250412-20260412 \
     --timeframe 4h 1h \
     --trading-mode futures
   ```

3. Copy strategies and config:
   ```bash
   cp /path/to/pymarket-structure/refs/freqtrade/strategies/*.py user_data/strategies/
   cp /path/to/pymarket-structure/refs/freqtrade/strategies/*.json user_data/strategies/
   ```

4. Run the backtest:
   ```bash
   freqtrade backtesting \
     --config /path/to/pymarket-structure/refs/freqtrade/config-futures.json \
     --strategy-list MsFilterV6 MsFilterV6NoMs MsSupportResistanceV1 \
     --timerange 20250412-20260412 \
     --export trades
   ```
