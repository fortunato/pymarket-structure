"""Internal ATR (Average True Range) computation using Wilder's smoothing."""

# pyright: reportUnusedFunction=false

from __future__ import annotations

import numpy as np


def _compute_atr(
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    period: int = 14,
) -> np.ndarray:
    """Compute Wilder's ATR over OHLC data.

    Returns an array of the same length as the inputs.  The first
    ``period - 1`` elements are NaN; index ``period - 1`` holds the
    initial SMA seed value.
    """
    n = len(highs)
    if n == 0:
        return np.array([], dtype=np.float64)

    # True Range: max of (H-L, |H-prev_close|, |L-prev_close|)
    tr = np.empty(n, dtype=np.float64)
    tr[0] = highs[0] - lows[0]
    if n > 1:
        hl = highs[1:] - lows[1:]
        hc = np.abs(highs[1:] - closes[:-1])
        lc = np.abs(lows[1:] - closes[:-1])
        tr[1:] = np.maximum(hl, np.maximum(hc, lc))

    # Wilder's smoothing: first ATR is SMA, then EMA with alpha = 1/period
    atr = np.full(n, np.nan, dtype=np.float64)
    if n < period:
        return atr

    atr[period - 1] = np.mean(tr[:period])
    for i in range(period, n):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period

    return atr
