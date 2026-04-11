"""Vectorized bulk-ingest for MarketStructureHelper.

The public entry point is ``hydrate(df)``, which combines a fast numpy
pre-pass (sign-flip detection, wave boundaries) with a Python loop over
waves (not candles) to build the wave registry.  This avoids per-candle
Python overhead while retaining full access to the sequential
wave-registry logic (backward scans, pullback) that cannot be vectorized.
"""

# pyright: reportPrivateUsage=false

import numpy as np
import pandas as pd

from market_structure.helper import MarketStructureHelper
from market_structure.types import Candle, Direction, Wave


def hydrate(
    df: pd.DataFrame,
    *,
    histogram_key: str = "tsi_hist",
    max_waves: int = 200,
) -> MarketStructureHelper:
    """Bulk-construct a ``MarketStructureHelper`` from a complete OHLCV frame.

    Performs a vectorized pre-pass to detect sign-flips and wave
    boundaries, then iterates over waves (not candles) to populate the
    registry.  The resulting helper is in the same state as if every row
    had been fed through ``register_candle`` one at a time.

    The DataFrame must contain columns: ``open_time``, ``open``, ``high``,
    ``low``, ``close``, ``volume``, and the column named by
    ``histogram_key``.

    Args:
        df: OHLCV DataFrame with a histogram column.
        histogram_key: Name of the histogram column.
        max_waves: Maximum confirmed waves to retain.

    Returns:
        A fully populated ``MarketStructureHelper``.
    """
    h = MarketStructureHelper(histogram_key=histogram_key, max_waves=max_waves)

    if df.empty:
        return h

    # Work on a clean 0-based integer index so positional lookups are simple.
    df = df.reset_index(drop=True)
    n = len(df)

    # ------------------------------------------------------------------
    # Extract numpy arrays for fast access
    # ------------------------------------------------------------------
    hist = df[histogram_key].to_numpy(dtype=float)
    opens = df["open"].to_numpy(dtype=float)
    highs = df["high"].to_numpy(dtype=float)
    lows = df["low"].to_numpy(dtype=float)
    closes = df["close"].to_numpy(dtype=float)
    volumes = df["volume"].to_numpy(dtype=float)
    open_times = df["open_time"].to_numpy()
    co_max = np.maximum(closes, opens)
    co_min = np.minimum(closes, opens)

    # ------------------------------------------------------------------
    # Vectorized sign-flip detection
    # ------------------------------------------------------------------
    sign = np.where(hist >= 0, 1, -1)
    flip = np.empty(n, dtype=bool)
    flip[0] = False
    flip[1:] = sign[1:] != sign[:-1]

    # ------------------------------------------------------------------
    # Wave boundaries
    # ------------------------------------------------------------------
    flip_indices = np.flatnonzero(flip)
    n_flips = len(flip_indices)
    n_waves = n_flips + 1

    wave_starts = np.empty(n_waves, dtype=np.intp)
    wave_starts[0] = 0
    if n_flips > 0:
        wave_starts[1:] = flip_indices

    wave_ends = np.empty(n_waves, dtype=np.intp)
    if n_flips > 0:
        wave_ends[:n_flips] = flip_indices - 1
    wave_ends[-1] = n - 1

    # ------------------------------------------------------------------
    # Build all Candle objects in one pass
    # ------------------------------------------------------------------
    all_candles: list[Candle] = [
        Candle(
            open_time=int(open_times[i]),
            open=float(opens[i]),
            high=float(highs[i]),
            low=float(lows[i]),
            close=float(closes[i]),
            volume=float(volumes[i]),
            histogram_value=float(hist[i]),
        )
        for i in range(n)
    ]

    # ------------------------------------------------------------------
    # Process confirmed waves (all but the last group)
    # ------------------------------------------------------------------
    n_confirmed = n_waves - 1

    for i in range(n_confirmed):
        start = int(wave_starts[i])
        end = int(wave_ends[i])
        candles = tuple(all_candles[start : end + 1])

        side: Direction = "up" if float(hist[start]) >= 0 else "down"

        # Per-wave extremes via numpy slices (views, no copy).
        s = slice(start, end + 1)
        high_pos = int(np.argmax(highs[s]))
        low_pos = int(np.argmin(lows[s]))
        highest_close_pos = int(np.argmax(closes[s]))
        lowest_close_pos = int(np.argmin(closes[s]))
        hco_pos = int(np.argmax(co_max[s]))
        lco_pos = int(np.argmin(co_min[s]))

        hco_candle = candles[hco_pos]
        lco_candle = candles[lco_pos]

        # formation_bar_index = index of the flip candle (first of next wave).
        formation_bar_index = int(wave_starts[i + 1])

        # Backward scans — read _wave_registry (populated by prior _push_wave calls).
        high_since = h._determine_high_since(hco_candle, hco_pos) if side == "up" else 0
        low_since = h._determine_low_since(lco_candle, lco_pos) if side == "down" else 0

        if side == "up":
            pullback = h._determine_pullback_from_bottom(hco_candle, hco_pos)
        else:
            pullback = h._determine_pullback_from_top(lco_candle, lco_pos)

        wave = Wave(
            id=f"w-{h._next_wave_id}",
            side=side,
            formation_bar_index=formation_bar_index,
            high=candles[high_pos],
            low=candles[low_pos],
            highest_close=candles[highest_close_pos],
            lowest_close=candles[lowest_close_pos],
            highest_close_or_open=hco_candle,
            lowest_close_or_open=lco_candle,
            high_idx=start + high_pos,
            low_idx=start + low_pos,
            highest_close_or_open_idx=start + hco_pos,
            lowest_close_or_open_idx=start + lco_pos,
            high_since=high_since,
            low_since=low_since,
            pullback=pullback,
            candles=candles,
        )

        h._next_wave_id += 1
        h._push_wave(wave)

    # ------------------------------------------------------------------
    # Set up forming wave state
    # ------------------------------------------------------------------
    forming_start = int(wave_starts[-1])
    forming_end = int(wave_ends[-1])
    h._wave_candles = list(all_candles[forming_start : forming_end + 1])
    h._wave_start_index = forming_start
    h._total_candles_registered = n
    h._previous_histogram_value = float(hist[-1])
    h._last_registered_open_time = int(open_times[-1])

    return h
