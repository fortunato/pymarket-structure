"""Synthetic fixture builders for the 004-robust-double-patterns suite.

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
    """CR-007: ATR array is all-NaN. Tolerance must fall through to the
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
    """CR-008: ATR array contains negative values. Tolerance must fall
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
    """D-8 item 1: two lows 0.15 apart, wick ranges disjoint.

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
    """D-8 item 2: two lows 2.0 apart with wide overlapping wicks.

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
    """D-8 item 3: two swing lows at the same price. Tolerance is inclusive at 0.

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
    """D-8 item 5 (flat-market): ATR is zero everywhere, lows are near-equal.

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
# US2 fixtures — default-proximity bump (1 → 2)
# ---------------------------------------------------------------------------


def build_w_pattern_with_intermediate() -> tuple[MarketStructureHelper, np.ndarray]:
    """D-8 item 4: classical W-pattern with one intermediate non-violating low.

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
# Edge-case fixtures (D-8 items 7-11)
# ---------------------------------------------------------------------------


def build_first_swing_only() -> tuple[MarketStructureHelper, np.ndarray]:
    """D-8 item 7: registry holds exactly one down-wave. No preceding same-side
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
    """D-8 item 8: ``atr_arr`` shorter than the DataFrame. Out-of-bounds indices
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
    """D-8 item 9: anchor's ``low_idx`` ATR ≈ 5x preceding's ``low_idx`` ATR.

    Lows at 100.00 (wave-0, bar 0) and 100.10 (wave-2, bar 3).
    ATR values: bar 0 = 0.05 (low-vol regime); bar 3 = 0.5 (high-vol regime).
    Under the anchor's current-regime ATR: 0.1 ≤ 0.3 x 0.5 = 0.15 → qualifies.
    Under the preceding's older-regime ATR: 0.1 > 0.3 x 0.05 = 0.015 → reject.

    A correct implementation reads ``atr_arr[anchor.low_idx]`` → qualifies.
    A bug that used ``atr_arr[preceding.low_idx]`` or
    ``atr_arr[anchor.formation_bar_index]`` would reject — this fixture
    pins the D-2 design intent.
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
    # reject the pair. Pinning D-2's design intent.
    atr_arr = np.array([0.05, 0.05, 0.50, 0.50, 0.05])
    return h, atr_arr


def build_adjacent_wicks() -> tuple[MarketStructureHelper, np.ndarray]:
    """D-8 item 10: two bottom wick ranges share an edge exactly.

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
    """D-8 item 11: three down-waves, both slot-0 and slot-1 qualify.

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
