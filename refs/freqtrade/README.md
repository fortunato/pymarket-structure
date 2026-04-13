# Freqtrade backtest data

Backtest results and OHLCV data from three strategies built on the `market_structure` library, run on 4h Binance futures (BTC, ETH, SOL, XRP, LTC) over 360 days (2025-04-17 to 2026-04-12) in a **-14.91% market**.

Used by the frontend to overlay trades on charts.

## Strategies

| Strategy | Description |
|---|---|
| MsFilterV6 | TSI crossover entry filtered by `ms_*` trend state |
| MsFilterV6NoMs | Same TSI crossover, no `ms_*` filters (control) |
| MsSupportResistanceV1 | S&R proximity entry with wave-side confirmation |

## Backtest results

All strategies backtested on 5 pairs x 4h, 2025-04-17 to 2026-04-12 (~360 days after 30-bar startup). 1,000 USDT starting balance, 300 USDT stake per trade, max 3 concurrent positions.

| Strategy | Trades | Win% | Total Profit | Profit Factor | Sharpe | Max Drawdown |
|---|---|---|---|---|---|---|
| MsFilterV6 | 164 | 36.0% | +39.85% | 1.50 | 0.94 | 17.76% |
| MsFilterV6NoMs | 208 | 31.7% | +22.59% | 1.21 | 0.58 | 21.75% |
| MsSupportResistanceV1 | 404 | 42.6% | +70.32% | 1.38 | 2.67 | 18.10% |

The MS filter blocked 44 low-quality entries (mostly would-be stop-outs), nearly doubling profit while cutting drawdown by ~4 percentage points.

> **Disclaimer:** This is educational software, not financial advice. Past backtest performance does not guarantee future results. Parameters are curve-fit to a specific historical window. Do not trade real capital based on these examples without independent validation and risk assessment.

## Files

```
backtest_results/
  {Strategy}-{PAIR}-4h-trades.json           trade records per strategy x pair (15 files)
ohlcv/
  {PAIR}-4h.json                             OHLCV candle data per pair (5 files)
```
