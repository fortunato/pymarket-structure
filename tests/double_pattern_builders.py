"""Synthetic fixture builders for the double-patterns suite.

Each builder constructs a ``MarketStructureHelper`` with a known wave
structure by feeding synthetic candles through ``register_candle``. The
candles' OHLC and ``histogram_value`` fields are crafted so that
histogram sign-flips land at precisely the bars that produce the target
registry.

Every builder returns a tuple ``(helper, atr_arr)`` where ``atr_arr`` is
aligned 1:1 with the DataFrame / candle sequence that was registered
(same length, same indexing). That lets the caller pass ``atr_arr``
straight into ``get_support_zones(atr_arr=...)`` /
``get_resistance_zones(atr_arr=...)``.

All wave-building helpers use single-candle waves where possible so that
``wave.low == wave.lowest_close_or_open`` (support) and
``wave.high == wave.highest_close_or_open`` (resistance) — simplifying
the relationship between bar OHLC and the wick-range geometry produced
by ``get_bottom_range`` / ``get_top_range``.

"""

# pyright: reportPrivateUsage=false

import numpy as np

from market_structure import MarketStructureHelper
from market_structure.types import Candle


def _c(
    open_time: int,
    *,
    open: float,
    high: float,
    low: float,
    close: float,
    histogram_value: float,
    volume: float = 1.0,
) -> Candle:
    """Compact constructor used throughout the builders."""
    return Candle(
        open_time=open_time,
        open=open,
        high=high,
        low=low,
        close=close,
        volume=volume,
        histogram_value=histogram_value,
    )


# ---------------------------------------------------------------------------
# Double-bottom fixtures (support side)
# ---------------------------------------------------------------------------


def build_nan_atr_bottoms() -> tuple[MarketStructureHelper, np.ndarray]:
    """ATR array is all-NaN. Tolerance must fall through to the
    percentage fallback via the ``np.isfinite(atr_val)`` guard in
    ``_double_pattern_tolerance``. Uses the same wave layout as
    ``build_tight_disjoint_bottoms`` (lows 0.15 apart). Percentage
    fallback: 0.004 x 100.15 = 0.4006 > 0.15 gap → qualifies.
    """
    h = MarketStructureHelper()
    h.register_candle(
        _c(1000, open=100.05, high=100.10, low=100.00, close=100.05, histogram_value=-0.5)
    )
    h.register_candle(
        _c(2000, open=105.00, high=110.00, low=104.00, close=109.00, histogram_value=0.3)
    )
    h.register_candle(
        _c(3000, open=109.00, high=112.00, low=108.00, close=111.00, histogram_value=0.2)
    )
    h.register_candle(
        _c(4000, open=102.00, high=102.00, low=100.15, close=102.00, histogram_value=-0.4)
    )
    h.register_candle(
        _c(5000, open=105.00, high=108.00, low=104.00, close=107.00, histogram_value=0.5)
    )
    atr_arr = np.full(5, np.nan)
    return h, atr_arr


def build_negative_atr_bottoms() -> tuple[MarketStructureHelper, np.ndarray]:
    """ATR array contains negative values. Tolerance must fall
    through to the percentage fallback via the ``atr_val > 0`` guard in
    ``_double_pattern_tolerance``. Same wave layout as
    ``build_tight_disjoint_bottoms``. Percentage fallback:
    0.004 x 100.15 = 0.4006 > 0.15 gap → qualifies.
    """
    h = MarketStructureHelper()
    h.register_candle(
        _c(1000, open=100.05, high=100.10, low=100.00, close=100.05, histogram_value=-0.5)
    )
    h.register_candle(
        _c(2000, open=105.00, high=110.00, low=104.00, close=109.00, histogram_value=0.3)
    )
    h.register_candle(
        _c(3000, open=109.00, high=112.00, low=108.00, close=111.00, histogram_value=0.2)
    )
    h.register_candle(
        _c(4000, open=102.00, high=102.00, low=100.15, close=102.00, histogram_value=-0.4)
    )
    h.register_candle(
        _c(5000, open=105.00, high=108.00, low=104.00, close=107.00, histogram_value=0.5)
    )
    atr_arr = np.array([-1.0, -0.5, -1.0, -0.5, -1.0])
    return h, atr_arr


def build_tight_disjoint_bottoms() -> tuple[MarketStructureHelper, np.ndarray]:
    """Two lows 0.15 apart, wick ranges disjoint.

    Wave layout (down-up-down):
        bar 0: down-wave-0 — single candle; low=100.00, body_bottom=100.05
        bar 1: flip up
        bar 2: up continuation
        bar 3: flip down — down-wave-2 single candle; low=100.15, body_bottom=102.00
        bar 4: flip up — confirms wave-2

    Wick range of wave-0 = (100.00, 100.05); wave-2 = (100.15, 102.00).
    These are DISJOINT (100.05 < 100.15) so the old wick-overlap
    predicate rejects the pair. Lows are 0.15 apart, within 0.3 x ATR=1.0
    tolerance, so the new predicate qualifies.
    """
    h = MarketStructureHelper()
    h.register_candle(
        _c(1000, open=100.05, high=100.10, low=100.00, close=100.05, histogram_value=-0.5)
    )
    h.register_candle(
        _c(2000, open=105.00, high=110.00, low=104.00, close=109.00, histogram_value=0.3)
    )
    h.register_candle(
        _c(3000, open=109.00, high=112.00, low=108.00, close=111.00, histogram_value=0.2)
    )
    h.register_candle(
        _c(4000, open=102.00, high=102.00, low=100.15, close=102.00, histogram_value=-0.4)
    )
    h.register_candle(
        _c(5000, open=105.00, high=108.00, low=104.00, close=107.00, histogram_value=0.5)
    )
    atr_arr = np.ones(5)
    return h, atr_arr


def build_wide_far_bottoms() -> tuple[MarketStructureHelper, np.ndarray]:
    """Two lows 2.0 apart with wide overlapping wicks.

    Wave layout (down-up-down):
        bar 0: down-wave-0 single candle; low=100.00, body_bottom=103.00
        bar 1: flip up
        bar 2: up continuation
        bar 3: flip down — down-wave-2 single candle; low=102.00, body_bottom=105.00
        bar 4: flip up — confirms wave-2

    Wick range of wave-0 = (100.00, 103.00); wave-2 = (102.00, 105.00).
    These OVERLAP at (102, 103) — the old predicate labels is_double=True.
    Lows are 2.0 apart >> 0.3 x ATR=1.0, so the new predicate rejects.
    """
    h = MarketStructureHelper()
    h.register_candle(
        _c(1000, open=103.00, high=103.00, low=100.00, close=103.00, histogram_value=-0.5)
    )
    h.register_candle(
        _c(2000, open=107.00, high=110.00, low=106.00, close=109.00, histogram_value=0.3)
    )
    h.register_candle(
        _c(3000, open=109.00, high=112.00, low=108.00, close=111.00, histogram_value=0.2)
    )
    h.register_candle(
        _c(4000, open=105.00, high=105.00, low=102.00, close=105.00, histogram_value=-0.4)
    )
    h.register_candle(
        _c(5000, open=108.00, high=111.00, low=107.00, close=110.00, histogram_value=0.5)
    )
    atr_arr = np.ones(5)
    return h, atr_arr


def build_exact_tie_bottoms() -> tuple[MarketStructureHelper, np.ndarray]:
    """Two swing lows at the same price. Tolerance is inclusive at 0.

    Wave layout (down-up-down):
        bar 0: down-wave-0 single candle; low=100.00, body_bottom=101.00
        bar 3: down-wave-2 single candle; low=100.00, body_bottom=102.00
    """
    h = MarketStructureHelper()
    h.register_candle(
        _c(1000, open=101.00, high=101.00, low=100.00, close=101.00, histogram_value=-0.5)
    )
    h.register_candle(
        _c(2000, open=105.00, high=110.00, low=104.00, close=109.00, histogram_value=0.3)
    )
    h.register_candle(
        _c(3000, open=109.00, high=112.00, low=108.00, close=111.00, histogram_value=0.2)
    )
    h.register_candle(
        _c(4000, open=102.00, high=102.00, low=100.00, close=102.00, histogram_value=-0.4)
    )
    h.register_candle(
        _c(5000, open=105.00, high=108.00, low=104.00, close=107.00, histogram_value=0.5)
    )
    atr_arr = np.ones(5)
    return h, atr_arr


def build_zero_atr_bottoms() -> tuple[MarketStructureHelper, np.ndarray]:
    """Flat-market: ATR is zero everywhere, lows are near-equal.

    Two lows 0.2 apart. Tolerance under zero-ATR falls through to
    ``tolerance_pct_fallback * anchor.low.low`` ≈ 0.4 → pair qualifies.

    Wave layout (down-up-down):
        bar 0: down-wave-0 — low=100.00, body_bottom=100.05
        bar 3: down-wave-2 — low=100.20, body_bottom=101.00
    """
    h = MarketStructureHelper()
    h.register_candle(
        _c(1000, open=100.05, high=100.10, low=100.00, close=100.05, histogram_value=-0.5)
    )
    h.register_candle(
        _c(2000, open=105.00, high=110.00, low=104.00, close=109.00, histogram_value=0.3)
    )
    h.register_candle(
        _c(3000, open=109.00, high=112.00, low=108.00, close=111.00, histogram_value=0.2)
    )
    h.register_candle(
        _c(4000, open=101.00, high=101.00, low=100.20, close=101.00, histogram_value=-0.4)
    )
    h.register_candle(
        _c(5000, open=105.00, high=108.00, low=104.00, close=107.00, histogram_value=0.5)
    )
    atr_arr = np.zeros(5)
    return h, atr_arr


# ---------------------------------------------------------------------------
# Double-top fixtures (resistance side — mirrors of the four bottom fixtures)
# ---------------------------------------------------------------------------


def build_tight_disjoint_tops() -> tuple[MarketStructureHelper, np.ndarray]:
    """Mirror of ``build_tight_disjoint_bottoms`` for double-tops.

    Wave layout (up-down-up):
        bar 0: up-wave-0 single candle; high=110.00, body_top=109.95
        bar 3: up-wave-2 single candle; high=109.85, body_top=108.00

    Wick range of wave-0 = (109.95, 110.00); wave-2 = (108.00, 109.85).
    DISJOINT (109.85 < 109.95). Highs are 0.15 apart, within 0.3 x ATR=1.0.
    """
    h = MarketStructureHelper()
    h.register_candle(
        _c(1000, open=109.95, high=110.00, low=109.90, close=109.95, histogram_value=0.5)
    )
    h.register_candle(
        _c(2000, open=95.00, high=95.00, low=90.00, close=92.00, histogram_value=-0.3)
    )
    h.register_candle(
        _c(3000, open=92.00, high=93.00, low=88.00, close=90.00, histogram_value=-0.2)
    )
    h.register_candle(
        _c(4000, open=108.00, high=109.85, low=108.00, close=108.00, histogram_value=0.4)
    )
    h.register_candle(
        _c(5000, open=95.00, high=95.00, low=91.00, close=92.00, histogram_value=-0.5)
    )
    atr_arr = np.ones(5)
    return h, atr_arr


def build_wide_far_tops() -> tuple[MarketStructureHelper, np.ndarray]:
    """Mirror of ``build_wide_far_bottoms``: highs 2.0 apart, wide overlap.

    Wave layout (up-down-up):
        bar 0: up-wave-0 — high=110.00, body_top=107.00
        bar 3: up-wave-2 — high=108.00, body_top=105.00

    Wick ranges (107, 110) and (105, 108) overlap at (107, 108). Under
    the OLD wick-overlap predicate this pair is_double=True. Under the
    new predicate 2.0 > 0.3 x 1.0 → rejected.
    """
    h = MarketStructureHelper()
    h.register_candle(
        _c(1000, open=107.00, high=110.00, low=107.00, close=107.00, histogram_value=0.5)
    )
    h.register_candle(
        _c(2000, open=95.00, high=95.00, low=90.00, close=92.00, histogram_value=-0.3)
    )
    h.register_candle(
        _c(3000, open=92.00, high=93.00, low=88.00, close=90.00, histogram_value=-0.2)
    )
    h.register_candle(
        _c(4000, open=105.00, high=108.00, low=105.00, close=105.00, histogram_value=0.4)
    )
    h.register_candle(
        _c(5000, open=95.00, high=95.00, low=91.00, close=92.00, histogram_value=-0.5)
    )
    atr_arr = np.ones(5)
    return h, atr_arr


def build_exact_tie_tops() -> tuple[MarketStructureHelper, np.ndarray]:
    """Mirror of ``build_exact_tie_bottoms``: two identical highs."""
    h = MarketStructureHelper()
    h.register_candle(
        _c(1000, open=109.00, high=110.00, low=109.00, close=109.00, histogram_value=0.5)
    )
    h.register_candle(
        _c(2000, open=95.00, high=95.00, low=90.00, close=92.00, histogram_value=-0.3)
    )
    h.register_candle(
        _c(3000, open=92.00, high=93.00, low=88.00, close=90.00, histogram_value=-0.2)
    )
    h.register_candle(
        _c(4000, open=108.00, high=110.00, low=108.00, close=108.00, histogram_value=0.4)
    )
    h.register_candle(
        _c(5000, open=95.00, high=95.00, low=91.00, close=92.00, histogram_value=-0.5)
    )
    atr_arr = np.ones(5)
    return h, atr_arr


def build_zero_atr_tops() -> tuple[MarketStructureHelper, np.ndarray]:
    """Mirror of ``build_zero_atr_bottoms``: ATR=0, percentage fallback path."""
    h = MarketStructureHelper()
    h.register_candle(
        _c(1000, open=109.95, high=110.00, low=109.90, close=109.95, histogram_value=0.5)
    )
    h.register_candle(
        _c(2000, open=95.00, high=95.00, low=90.00, close=92.00, histogram_value=-0.3)
    )
    h.register_candle(
        _c(3000, open=92.00, high=93.00, low=88.00, close=90.00, histogram_value=-0.2)
    )
    h.register_candle(
        _c(4000, open=108.90, high=109.80, low=108.90, close=108.90, histogram_value=0.4)
    )
    h.register_candle(
        _c(5000, open=95.00, high=95.00, low=91.00, close=92.00, histogram_value=-0.5)
    )
    atr_arr = np.zeros(5)
    return h, atr_arr


# ---------------------------------------------------------------------------
# Default-proximity bump (1 → 2) fixtures
# ---------------------------------------------------------------------------


def build_w_pattern_with_intermediate() -> tuple[MarketStructureHelper, np.ndarray]:
    """Classical W-pattern with one intermediate non-violating low.

    Three down-waves: L1 at 100, intermediate L2 at 100.5 (higher than L1),
    L3 at 100.1 (matches L1 within tolerance). Under proximity=1 the
    anchor only looks one step back (at L2) and does NOT find L1, so
    is_double=False. Under proximity=2 the anchor can reach L1 → double.
    """
    h = MarketStructureHelper()
    # wave-0 down: bar 0 (low=100)
    h.register_candle(
        _c(1000, open=100.05, high=100.10, low=100.00, close=100.05, histogram_value=-0.5)
    )
    # wave-1 up: bars 1-2
    h.register_candle(
        _c(2000, open=104.00, high=108.00, low=103.00, close=107.00, histogram_value=0.3)
    )
    h.register_candle(
        _c(3000, open=107.00, high=110.00, low=106.00, close=109.00, histogram_value=0.2)
    )
    # wave-2 down: bar 3 (intermediate higher low=100.5)
    h.register_candle(
        _c(4000, open=101.00, high=101.00, low=100.50, close=101.00, histogram_value=-0.4)
    )
    # wave-3 up: bars 4-5
    h.register_candle(
        _c(5000, open=104.00, high=108.00, low=103.00, close=107.00, histogram_value=0.4)
    )
    h.register_candle(
        _c(6000, open=107.00, high=110.00, low=106.00, close=109.00, histogram_value=0.3)
    )
    # wave-4 down: bar 6 (anchor low=100.1)
    h.register_candle(
        _c(7000, open=101.00, high=101.00, low=100.10, close=101.00, histogram_value=-0.3)
    )
    # confirm wave-4 with a flip up
    h.register_candle(
        _c(8000, open=105.00, high=108.00, low=104.00, close=107.00, histogram_value=0.5)
    )
    atr_arr = np.ones(8)
    return h, atr_arr


def build_m_pattern_with_intermediate() -> tuple[MarketStructureHelper, np.ndarray]:
    """Mirror of ``build_w_pattern_with_intermediate`` for resistance."""
    h = MarketStructureHelper()
    # wave-0 up: bar 0 (high=110)
    h.register_candle(
        _c(1000, open=109.95, high=110.00, low=109.90, close=109.95, histogram_value=0.5)
    )
    # wave-1 down: bars 1-2
    h.register_candle(
        _c(2000, open=94.00, high=95.00, low=90.00, close=92.00, histogram_value=-0.3)
    )
    h.register_candle(
        _c(3000, open=92.00, high=93.00, low=88.00, close=90.00, histogram_value=-0.2)
    )
    # wave-2 up: bar 3 (intermediate lower high=109.5)
    h.register_candle(
        _c(4000, open=109.00, high=109.50, low=109.00, close=109.00, histogram_value=0.4)
    )
    # wave-3 down: bars 4-5
    h.register_candle(
        _c(5000, open=94.00, high=95.00, low=90.00, close=92.00, histogram_value=-0.4)
    )
    h.register_candle(
        _c(6000, open=92.00, high=93.00, low=88.00, close=90.00, histogram_value=-0.3)
    )
    # wave-4 up: bar 6 (anchor high=109.9)
    h.register_candle(
        _c(7000, open=109.00, high=109.90, low=109.00, close=109.00, histogram_value=0.3)
    )
    # confirm wave-4 with a flip down
    h.register_candle(
        _c(8000, open=95.00, high=95.00, low=91.00, close=92.00, histogram_value=-0.5)
    )
    atr_arr = np.ones(8)
    return h, atr_arr


# ---------------------------------------------------------------------------
# Edge-case fixtures
# ---------------------------------------------------------------------------


def build_first_swing_only() -> tuple[MarketStructureHelper, np.ndarray]:
    """Registry holds exactly one down-wave. No preceding same-side
    wave exists, so the double-pattern body must short-circuit without crashing
    and produce a zone with ``is_double=False``.
    """
    h = MarketStructureHelper()
    h.register_candle(
        _c(1000, open=101.00, high=101.00, low=100.00, close=101.00, histogram_value=-0.5)
    )
    # confirm wave-0 with a flip up
    h.register_candle(
        _c(2000, open=105.00, high=108.00, low=104.00, close=107.00, histogram_value=0.5)
    )
    atr_arr = np.ones(2)
    return h, atr_arr


def build_short_atr_array() -> tuple[MarketStructureHelper, np.ndarray]:
    """``atr_arr`` shorter than the DataFrame. Out-of-bounds indices
    must fall through to the percentage fallback without raising.

    Uses the same wave layout as ``build_tight_disjoint_bottoms`` (lows at 100.00
    and 100.15 → gap of 0.15). The anchor's ``low_idx`` is 3, but ``atr_arr``
    has length 1, so the lookup is out-of-bounds → percentage fallback:
    0.004 x 100.15 ≈ 0.40 > 0.15 → qualifies.
    """
    h = MarketStructureHelper()
    h.register_candle(
        _c(1000, open=100.05, high=100.10, low=100.00, close=100.05, histogram_value=-0.5)
    )
    h.register_candle(
        _c(2000, open=105.00, high=110.00, low=104.00, close=109.00, histogram_value=0.3)
    )
    h.register_candle(
        _c(3000, open=109.00, high=112.00, low=108.00, close=111.00, histogram_value=0.2)
    )
    h.register_candle(
        _c(4000, open=102.00, high=102.00, low=100.15, close=102.00, histogram_value=-0.4)
    )
    h.register_candle(
        _c(5000, open=105.00, high=108.00, low=104.00, close=107.00, histogram_value=0.5)
    )
    # Intentionally too short (length 1 vs 5 bars registered).
    atr_arr = np.array([1.0])
    return h, atr_arr


def build_regime_shift_atr() -> tuple[MarketStructureHelper, np.ndarray]:
    """Anchor's ``low_idx`` ATR ≈ 5x preceding's ``low_idx`` ATR.

    Lows at 100.00 (wave-0, bar 0) and 100.10 (wave-2, bar 3).
    ATR values: bar 0 = 0.05 (low-vol regime); bar 3 = 0.5 (high-vol regime).
    Under the anchor's current-regime ATR: 0.1 ≤ 0.3 x 0.5 = 0.15 → qualifies.
    Under the preceding's older-regime ATR: 0.1 > 0.3 x 0.05 = 0.015 → reject.

    A correct implementation reads ``atr_arr[anchor.low_idx]`` → qualifies.
    A bug that used ``atr_arr[preceding.low_idx]`` or
    ``atr_arr[anchor.formation_bar_index]`` would reject — this fixture
    pins the design intent.
    """
    h = MarketStructureHelper()
    h.register_candle(
        _c(1000, open=100.02, high=100.05, low=100.00, close=100.02, histogram_value=-0.5)
    )
    h.register_candle(
        _c(2000, open=105.00, high=110.00, low=104.00, close=109.00, histogram_value=0.3)
    )
    h.register_candle(
        _c(3000, open=109.00, high=112.00, low=108.00, close=111.00, histogram_value=0.2)
    )
    h.register_candle(
        _c(4000, open=101.00, high=101.00, low=100.10, close=101.00, histogram_value=-0.4)
    )
    h.register_candle(
        _c(5000, open=105.00, high=108.00, low=104.00, close=107.00, histogram_value=0.5)
    )
    # bar 0 sits in a quiet regime; bar 3 (anchor's low_idx) enters a
    # high-vol regime. Bar 4 (anchor's formation_bar_index, the flip
    # candle) swings BACK to low-vol — so a buggy implementation that
    # keyed on ``formation_bar_index`` instead of ``low_idx`` would
    # reject the pair. Pinning the design intent.
    atr_arr = np.array([0.05, 0.05, 0.50, 0.50, 0.05])
    return h, atr_arr


def build_adjacent_wicks() -> tuple[MarketStructureHelper, np.ndarray]:
    """Two bottom wick ranges share an edge exactly.

    Wave-0 range ends at 100.20; wave-2 range starts at 100.20. Inclusive
    ``range_overlaps`` returns True. The deeper-wick extension should fire
    and the zone should extend down to 100.00.
    """
    h = MarketStructureHelper()
    h.register_candle(
        _c(1000, open=100.20, high=100.25, low=100.00, close=100.20, histogram_value=-0.5)
    )
    h.register_candle(
        _c(2000, open=105.00, high=110.00, low=104.00, close=109.00, histogram_value=0.3)
    )
    h.register_candle(
        _c(3000, open=109.00, high=112.00, low=108.00, close=111.00, histogram_value=0.2)
    )
    h.register_candle(
        _c(4000, open=101.00, high=101.00, low=100.20, close=101.00, histogram_value=-0.4)
    )
    h.register_candle(
        _c(5000, open=105.00, high=108.00, low=104.00, close=107.00, histogram_value=0.5)
    )
    atr_arr = np.ones(5)
    return h, atr_arr


def build_nearest_not_deepest() -> tuple[MarketStructureHelper, np.ndarray]:
    """Three down-waves, both slot-0 and slot-1 qualify.

    Wave-0 (oldest; slot 1 from anchor): low=99.90 — "deepest" (lowest)
    Wave-2 (middle; slot 0 from anchor): low=100.05
    Wave-4 (newest; anchor):             low=100.00

    All three lows sit within 0.3 of each other (tolerance at ATR=1.0),
    and the intervening non-matching waves are well above all three lows
    so ``made_lower_low_between`` returns False for both slots. All three
    wick ranges overlap, so ``overlapping_low_wave_ids`` should contain
    BOTH wave-2 and wave-0. The "nearest-not-deepest" guard asserts that
    slot 0 (wave-2, the closer match) is present — protection against a
    hypothetical regression that might short-circuit and keep only the
    price-deepest match (wave-0 here).
    """
    h = MarketStructureHelper()
    # wave-0 down (oldest, price-deepest)
    h.register_candle(
        _c(1000, open=101.00, high=101.00, low=99.90, close=101.00, histogram_value=-0.5)
    )
    # wave-1 up
    h.register_candle(
        _c(2000, open=104.00, high=108.00, low=103.00, close=107.00, histogram_value=0.3)
    )
    h.register_candle(
        _c(3000, open=107.00, high=110.00, low=106.00, close=109.00, histogram_value=0.2)
    )
    # wave-2 down (middle, nearer match, narrow wick)
    h.register_candle(
        _c(4000, open=100.20, high=100.25, low=100.05, close=100.20, histogram_value=-0.4)
    )
    # wave-3 up
    h.register_candle(
        _c(5000, open=104.00, high=108.00, low=103.00, close=107.00, histogram_value=0.4)
    )
    h.register_candle(
        _c(6000, open=107.00, high=110.00, low=106.00, close=109.00, histogram_value=0.3)
    )
    # wave-4 down (anchor)
    h.register_candle(
        _c(7000, open=101.00, high=101.00, low=100.00, close=101.00, histogram_value=-0.3)
    )
    # confirm wave-4 with a flip up
    h.register_candle(
        _c(8000, open=105.00, high=108.00, low=104.00, close=107.00, histogram_value=0.5)
    )
    atr_arr = np.ones(8)
    return h, atr_arr


# ---------------------------------------------------------------------------
# Body-anchored zone geometry fixtures
# ---------------------------------------------------------------------------


def build_long_lower_wick_anchor() -> tuple[MarketStructureHelper, np.ndarray]:
    """Support anchor with a body far above the wick extreme.

    Wave layout (down-up-down-up):
        bar 0: down-wave-0 — low=90.00 (extreme wick), close=98.00, open=99.00
                body = (98.00, 99.00), wick extends 8 points below body
        bar 1-2: up-wave
        bar 3: down-wave-2 — unrelated, moderate depth
        bar 4: up — confirms wave-2

    After body-anchoring: zone.range should be (98.00, 99.00)
    Wick range: (90.00, 98.00)
    """
    h = MarketStructureHelper()
    h.register_candle(
        _c(1000, open=99.00, high=100.00, low=90.00, close=98.00, histogram_value=-0.5)
    )
    h.register_candle(
        _c(2000, open=105.00, high=110.00, low=104.00, close=109.00, histogram_value=0.3)
    )
    h.register_candle(
        _c(3000, open=109.00, high=112.00, low=108.00, close=111.00, histogram_value=0.2)
    )
    h.register_candle(
        _c(4000, open=103.00, high=103.00, low=101.00, close=103.00, histogram_value=-0.4)
    )
    h.register_candle(
        _c(5000, open=105.00, high=108.00, low=104.00, close=107.00, histogram_value=0.5)
    )
    atr_arr = np.ones(5)
    return h, atr_arr


def build_long_upper_wick_anchor() -> tuple[MarketStructureHelper, np.ndarray]:
    """Resistance anchor with a body far below the wick extreme.

    Wave layout (up-down-up-down):
        bar 0: up-wave-0 — high=110.00 (extreme wick), close=102.00, open=101.00
                body = (101.00, 102.00), wick extends 8 points above body
        bar 1-2: down-wave
        bar 3: up-wave-2 — unrelated, moderate height
        bar 4: down — confirms wave-2

    After body-anchoring: zone.range should be (101.00, 102.00)
    Wick range: (102.00, 110.00)
    """
    h = MarketStructureHelper()
    h.register_candle(
        _c(1000, open=101.00, high=110.00, low=100.00, close=102.00, histogram_value=0.5)
    )
    h.register_candle(
        _c(2000, open=95.00, high=96.00, low=90.00, close=91.00, histogram_value=-0.3)
    )
    h.register_candle(
        _c(3000, open=91.00, high=92.00, low=88.00, close=89.00, histogram_value=-0.2)
    )
    h.register_candle(_c(4000, open=97.00, high=97.00, low=96.00, close=97.00, histogram_value=0.4))
    h.register_candle(
        _c(5000, open=93.00, high=94.00, low=92.00, close=93.00, histogram_value=-0.5)
    )
    atr_arr = np.ones(5)
    return h, atr_arr


def build_doji_anchor() -> tuple[MarketStructureHelper, np.ndarray]:
    """Support anchor whose lowest-close candle has open == close (doji).

    body = (100.00, 100.00), zero width. Wick = (95.00, 100.00).
    """
    h = MarketStructureHelper()
    h.register_candle(
        _c(1000, open=100.00, high=102.00, low=95.00, close=100.00, histogram_value=-0.5)
    )
    h.register_candle(
        _c(2000, open=105.00, high=110.00, low=104.00, close=109.00, histogram_value=0.3)
    )
    h.register_candle(
        _c(3000, open=109.00, high=112.00, low=108.00, close=111.00, histogram_value=0.2)
    )
    h.register_candle(
        _c(4000, open=103.00, high=103.00, low=101.00, close=103.00, histogram_value=-0.4)
    )
    h.register_candle(
        _c(5000, open=105.00, high=108.00, low=104.00, close=107.00, histogram_value=0.5)
    )
    atr_arr = np.ones(5)
    return h, atr_arr


def build_single_candle_wave() -> tuple[MarketStructureHelper, np.ndarray]:
    """Wave with a single candle. Zone must equal that candle's body.

    Wave-0 is a single down candle: open=100.00, close=97.00, low=95.00, high=101.00.
    body = (97.00, 100.00). Wick extends to 95.00.
    The immediate flip to up after one bar means wave-0 has exactly one candle.
    """
    h = MarketStructureHelper()
    h.register_candle(
        _c(1000, open=100.00, high=101.00, low=95.00, close=97.00, histogram_value=-0.5)
    )
    h.register_candle(
        _c(2000, open=105.00, high=110.00, low=104.00, close=109.00, histogram_value=0.3)
    )
    h.register_candle(
        _c(3000, open=109.00, high=112.00, low=108.00, close=111.00, histogram_value=0.2)
    )
    h.register_candle(
        _c(4000, open=103.00, high=103.00, low=101.00, close=103.00, histogram_value=-0.4)
    )
    h.register_candle(
        _c(5000, open=105.00, high=108.00, low=104.00, close=107.00, histogram_value=0.5)
    )
    atr_arr = np.ones(5)
    return h, atr_arr


def build_marubozu_anchor() -> tuple[MarketStructureHelper, np.ndarray]:
    """Support anchor whose lowest-close candle is a pure bear marubozu.

    open == high and close == low -> body == full range.
    body = (95.00, 100.00), wick = (95.00, 100.00) — identical.
    """
    h = MarketStructureHelper()
    h.register_candle(
        _c(1000, open=100.00, high=100.00, low=95.00, close=95.00, histogram_value=-0.5)
    )
    h.register_candle(
        _c(2000, open=105.00, high=110.00, low=104.00, close=109.00, histogram_value=0.3)
    )
    h.register_candle(
        _c(3000, open=109.00, high=112.00, low=108.00, close=111.00, histogram_value=0.2)
    )
    h.register_candle(
        _c(4000, open=103.00, high=103.00, low=101.00, close=103.00, histogram_value=-0.4)
    )
    h.register_candle(
        _c(5000, open=105.00, high=108.00, low=104.00, close=107.00, histogram_value=0.5)
    )
    atr_arr = np.ones(5)
    return h, atr_arr


def build_tie_break_lowest_close() -> tuple[MarketStructureHelper, np.ndarray]:
    """Two candles in the same down-wave tie on lowest close.

    bar 0: open=101.00, close=98.00, low=95.00 (first candle, close=98.00)
    bar 1: open=99.00, close=98.00, low=96.00 (second candle, same close=98.00)
    Both in same down-wave. The earlier candle (bar 0) should be the anchor.
    body of bar 0 = (98.00, 101.00).
    """
    h = MarketStructureHelper()
    # Two candles in the down-wave, both with close=98.00
    h.register_candle(
        _c(1000, open=101.00, high=102.00, low=95.00, close=98.00, histogram_value=-0.5)
    )
    h.register_candle(
        _c(2000, open=99.00, high=100.00, low=96.00, close=98.00, histogram_value=-0.3)
    )
    # Flip to up
    h.register_candle(
        _c(3000, open=105.00, high=110.00, low=104.00, close=109.00, histogram_value=0.3)
    )
    h.register_candle(
        _c(4000, open=109.00, high=112.00, low=108.00, close=111.00, histogram_value=0.2)
    )
    # Another down wave to provide context
    h.register_candle(
        _c(5000, open=103.00, high=103.00, low=101.00, close=103.00, histogram_value=-0.4)
    )
    h.register_candle(
        _c(6000, open=105.00, high=108.00, low=104.00, close=107.00, histogram_value=0.5)
    )
    atr_arr = np.ones(6)
    return h, atr_arr


def build_body_overlap_disjoint_wicks_bottoms() -> tuple[MarketStructureHelper, np.ndarray]:
    """Two bottom anchors whose body ranges overlap but wick ranges are disjoint.

    Wave layout (down-up-down-up):
        bar 0: down-wave-0 — open=101, close=98.1, low=98.0, high=101.5
               body = (98.1, 101.0); wick_range = (98.0, 98.1)
        bar 1-2: up-wave
        bar 3: down-wave-2 — open=102, close=99, low=98.2, high=102.5
               body = (99.0, 102.0); wick_range = (98.2, 99.0)
        bar 4: up — confirms wave-2

    Bodies overlap at (99.0, 101.0).
    Wick ranges (98.0, 98.1) vs (98.2, 99.0) — disjoint (98.1 < 98.2).
    Wave lows: 98.0 vs 98.2, diff=0.2 ≤ 0.3 ATR tolerance.
    """
    h = MarketStructureHelper()
    h.register_candle(
        _c(1000, open=101.00, high=101.50, low=98.00, close=98.10, histogram_value=-0.5)
    )
    h.register_candle(
        _c(2000, open=105.00, high=110.00, low=104.00, close=109.00, histogram_value=0.3)
    )
    h.register_candle(
        _c(3000, open=109.00, high=112.00, low=108.00, close=111.00, histogram_value=0.2)
    )
    h.register_candle(
        _c(4000, open=102.00, high=102.50, low=98.20, close=99.00, histogram_value=-0.4)
    )
    h.register_candle(
        _c(5000, open=105.00, high=108.00, low=104.00, close=107.00, histogram_value=0.5)
    )
    atr_arr = np.ones(5)
    return h, atr_arr


def build_body_overlap_disjoint_wicks_tops() -> tuple[MarketStructureHelper, np.ndarray]:
    """Two top anchors whose body ranges overlap but wick ranges are disjoint.

    Wave layout (up-down-up-down):
        bar 0: up-wave-0 — open=99, close=101.9, high=102.0, low=98.5
               body = (99.0, 101.9); wick_range = (101.9, 102.0)
        bar 1-2: down-wave
        bar 3: up-wave-2 — open=98, close=101, high=101.8, low=97.5
               body = (98.0, 101.0); wick_range = (101.0, 101.8)
        bar 4: down — confirms wave-2

    Bodies overlap at (99.0, 101.0).
    Wick ranges (101.9, 102.0) vs (101.0, 101.8) — disjoint (101.8 < 101.9).
    Wave highs: 102.0 vs 101.8, diff=0.2 ≤ 0.3 ATR tolerance.
    """
    h = MarketStructureHelper()
    h.register_candle(
        _c(1000, open=99.00, high=102.00, low=98.50, close=101.90, histogram_value=0.5)
    )
    h.register_candle(
        _c(2000, open=95.00, high=96.00, low=90.00, close=91.00, histogram_value=-0.3)
    )
    h.register_candle(
        _c(3000, open=91.00, high=92.00, low=88.00, close=89.00, histogram_value=-0.2)
    )
    h.register_candle(
        _c(4000, open=98.00, high=101.80, low=97.50, close=101.00, histogram_value=0.4)
    )
    h.register_candle(
        _c(5000, open=93.00, high=94.00, low=92.00, close=93.00, histogram_value=-0.5)
    )
    atr_arr = np.ones(5)
    return h, atr_arr


def build_body_disjoint_wick_overlap_bottoms() -> tuple[MarketStructureHelper, np.ndarray]:
    """Two bottom anchors whose body ranges are disjoint but wick ranges overlap.

    Wave layout (down-up-down-up):
        bar 0: down-wave-0 — open=96, close=95, low=93, high=97
               body = (95.0, 96.0); wick_range = (93.0, 95.0)
        bar 1-2: up-wave
        bar 3: down-wave-2 — open=98, close=97, low=94, high=99
               body = (97.0, 98.0); wick_range = (94.0, 97.0)
        bar 4: up — confirms wave-2

    Bodies (95,96) vs (97,98) — disjoint (96 < 97).
    Wick ranges (93,95) vs (94,97) — overlap at (94,95).
    Wave lows: 93 vs 94, diff=1.0. ATR=5, tolerance=1.5 → qualifies.
    """
    h = MarketStructureHelper()
    h.register_candle(
        _c(1000, open=96.00, high=97.00, low=93.00, close=95.00, histogram_value=-0.5)
    )
    h.register_candle(
        _c(2000, open=105.00, high=110.00, low=104.00, close=109.00, histogram_value=0.3)
    )
    h.register_candle(
        _c(3000, open=109.00, high=112.00, low=108.00, close=111.00, histogram_value=0.2)
    )
    h.register_candle(
        _c(4000, open=98.00, high=99.00, low=94.00, close=97.00, histogram_value=-0.4)
    )
    h.register_candle(
        _c(5000, open=105.00, high=108.00, low=104.00, close=107.00, histogram_value=0.5)
    )
    atr_arr = np.full(5, 5.0)
    return h, atr_arr


def build_body_disjoint_wick_overlap_tops() -> tuple[MarketStructureHelper, np.ndarray]:
    """Two top anchors whose body ranges are disjoint but wick ranges overlap.

    Wave layout (up-down-up-down):
        bar 0: up-wave-0 — open=103, close=104, high=107, low=102
               body = (103.0, 104.0); wick_range = (104.0, 107.0)
        bar 1-2: down-wave
        bar 3: up-wave-2 — open=101, close=102, high=106, low=100
               body = (101.0, 102.0); wick_range = (102.0, 106.0)
        bar 4: down — confirms wave-2

    Bodies (103,104) vs (101,102) — disjoint (102 < 103).
    Wick ranges (104,107) vs (102,106) — overlap at (104,106).
    Wave highs: 107 vs 106, diff=1.0. ATR=5, tolerance=1.5 → qualifies.
    """
    h = MarketStructureHelper()
    h.register_candle(
        _c(1000, open=103.00, high=107.00, low=102.00, close=104.00, histogram_value=0.5)
    )
    h.register_candle(
        _c(2000, open=95.00, high=96.00, low=90.00, close=91.00, histogram_value=-0.3)
    )
    h.register_candle(
        _c(3000, open=91.00, high=92.00, low=88.00, close=89.00, histogram_value=-0.2)
    )
    h.register_candle(
        _c(4000, open=101.00, high=106.00, low=100.00, close=102.00, histogram_value=0.4)
    )
    h.register_candle(
        _c(5000, open=93.00, high=94.00, low=92.00, close=93.00, histogram_value=-0.5)
    )
    atr_arr = np.full(5, 5.0)
    return h, atr_arr


def build_body_touching_bottoms() -> tuple[MarketStructureHelper, np.ndarray]:
    """Two bottom anchors whose body ranges exactly touch (closed interval).

    Wave layout (down-up-down-up):
        bar 0: down-wave-0 — open=102, close=100, low=97, high=103
               body = (100.0, 102.0)
        bar 1-2: up-wave
        bar 3: down-wave-2 — open=100, close=98, low=95, high=101
               body = (98.0, 100.0)
        bar 4: up — confirms wave-2

    Bodies: wave-0 body_bottom=100 == wave-2 body_top=100 → touching.
    range_overlaps uses <= so this qualifies as overlap.
    Wave lows: 97 vs 95, diff=2.0. ATR=10, tolerance=3.0 → qualifies.
    """
    h = MarketStructureHelper()
    h.register_candle(
        _c(1000, open=102.00, high=103.00, low=97.00, close=100.00, histogram_value=-0.5)
    )
    h.register_candle(
        _c(2000, open=108.00, high=115.00, low=107.00, close=114.00, histogram_value=0.3)
    )
    h.register_candle(
        _c(3000, open=114.00, high=118.00, low=113.00, close=117.00, histogram_value=0.2)
    )
    h.register_candle(
        _c(4000, open=100.00, high=101.00, low=95.00, close=98.00, histogram_value=-0.4)
    )
    h.register_candle(
        _c(5000, open=105.00, high=108.00, low=104.00, close=107.00, histogram_value=0.5)
    )
    atr_arr = np.full(5, 10.0)
    return h, atr_arr


def build_body_touching_tops() -> tuple[MarketStructureHelper, np.ndarray]:
    """Two top anchors whose body ranges exactly touch (closed interval).

    Wave layout (up-down-up-down):
        bar 0: up-wave-0 — open=98, close=100, high=103, low=97
               body = (98.0, 100.0)
        bar 1-2: down-wave
        bar 3: up-wave-2 — open=100, close=102, high=105, low=99
               body = (100.0, 102.0)
        bar 4: down — confirms wave-2

    Bodies: wave-0 body_top=100 == wave-2 body_bottom=100 → touching.
    range_overlaps uses <= so this qualifies as overlap.
    Wave highs: 103 vs 105, diff=2.0. ATR=10, tolerance=3.0 → qualifies.
    """
    h = MarketStructureHelper()
    h.register_candle(
        _c(1000, open=98.00, high=103.00, low=97.00, close=100.00, histogram_value=0.5)
    )
    h.register_candle(
        _c(2000, open=92.00, high=93.00, low=85.00, close=86.00, histogram_value=-0.3)
    )
    h.register_candle(
        _c(3000, open=86.00, high=87.00, low=82.00, close=83.00, histogram_value=-0.2)
    )
    h.register_candle(
        _c(4000, open=100.00, high=105.00, low=99.00, close=102.00, histogram_value=0.4)
    )
    h.register_candle(
        _c(5000, open=93.00, high=94.00, low=92.00, close=93.00, histogram_value=-0.5)
    )
    atr_arr = np.full(5, 10.0)
    return h, atr_arr


def build_tie_break_highest_close() -> tuple[MarketStructureHelper, np.ndarray]:
    """Two candles in the same up-wave tie on highest close.

    bar 0: open=99.00, close=102.00, high=105.00 (first candle, close=102.00)
    bar 1: open=101.00, close=102.00, high=106.00 (second candle, same close=102.00)
    Both in same up-wave. The earlier candle (bar 0) should be the anchor.
    body of bar 0 = (99.00, 102.00).
    """
    h = MarketStructureHelper()
    # Two candles in the up-wave, both with close=102.00
    h.register_candle(
        _c(1000, open=99.00, high=105.00, low=98.00, close=102.00, histogram_value=0.5)
    )
    h.register_candle(
        _c(2000, open=101.00, high=106.00, low=100.00, close=102.00, histogram_value=0.3)
    )
    # Flip to down
    h.register_candle(
        _c(3000, open=95.00, high=96.00, low=90.00, close=91.00, histogram_value=-0.3)
    )
    h.register_candle(
        _c(4000, open=91.00, high=92.00, low=88.00, close=89.00, histogram_value=-0.2)
    )
    # Another up wave
    h.register_candle(_c(5000, open=97.00, high=97.00, low=96.00, close=97.00, histogram_value=0.4))
    h.register_candle(
        _c(6000, open=93.00, high=94.00, low=92.00, close=93.00, histogram_value=-0.5)
    )
    atr_arr = np.ones(6)
    return h, atr_arr
