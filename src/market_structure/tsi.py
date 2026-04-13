"""True Strength Index (TSI) computation.

Provides ``compute_tsi`` which returns TSI, signal, and histogram columns
from a close-price Series using double-smoothed EMA of momentum.
"""

import pandas as pd


def compute_tsi(
    close: pd.Series,  # type: ignore[type-arg]
    r: int = 12,
    s: int = 8,
    signal_period: int = 4,
) -> pd.DataFrame:
    """Compute TSI, signal line, and histogram.

    Args:
        close: Close prices.
        r: Long smoothing period (first EMA).  Note: some charting
            platforms use ``r`` for the *short* period — here ``r``
            is the *first* (long) smoothing, matching the original
            Blau formulation.
        s: Short smoothing period (second EMA).
        signal_period: Signal line EMA period.

    Returns:
        DataFrame with columns ``tsi``, ``tsi_signal``, ``tsi_histogram``.
    """
    momentum = close.diff()

    # Double-smoothed momentum.
    ema1 = momentum.ewm(span=r, adjust=False).mean()
    double_smoothed = ema1.ewm(span=s, adjust=False).mean()

    # Double-smoothed absolute momentum.
    abs_ema1 = momentum.abs().ewm(span=r, adjust=False).mean()
    abs_double_smoothed = abs_ema1.ewm(span=s, adjust=False).mean()

    tsi = 100 * double_smoothed / abs_double_smoothed
    tsi_signal = tsi.ewm(span=signal_period, adjust=False).mean()
    tsi_histogram = tsi - tsi_signal

    return pd.DataFrame(
        {
            "tsi": tsi,
            "tsi_signal": tsi_signal,
            "tsi_histogram": tsi_histogram,
        }
    )
