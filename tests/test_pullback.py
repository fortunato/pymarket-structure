"""Stage 7 tests for pullback computation.

Pullback measures the retracement from a prior opposite-direction wave.
For up-waves, ``_determine_pullback_from_bottom`` looks at the last
confirmed bottom; for down-waves, ``_determine_pullback_from_top``
looks at the last confirmed top.

``correction_factor`` expresses the current move as a fraction of the
prior run (previous same-side wave → opposite wave). It requires at
least three confirmed waves to compute.
"""

import pytest

from market_structure import MarketStructureHelper
from market_structure.types import Candle

# ---------------------------------------------------------------------------
# Test-local candle factory
# ---------------------------------------------------------------------------


def _candle(
    open_time: int = 1_000,
    *,
    open: float = 100.0,
    high: float = 101.0,
    low: float = 99.0,
    close: float = 100.5,
    volume: float = 1.0,
    histogram_value: float = 0.0,
) -> Candle:
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
# Pullback from bottom (up-waves)
# ---------------------------------------------------------------------------


class TestPullbackFromBottom:
    """Pullback for up-waves, computed from the last confirmed bottom."""

    def test_none_when_no_prior_bottom(self) -> None:
        """First up-wave has no bottom to pull back from → None."""
        h = MarketStructureHelper()
        h.register_candle(_candle(open_time=1_000, histogram_value=0.5))
        h.register_candle(_candle(open_time=2_000, histogram_value=-0.3))  # flip → w-0 (up)
        assert h.wave_registry[0].side == "up"
        assert h.wave_registry[0].pullback is None

    def test_basic_pullback_metrics(self) -> None:
        """Up-wave after a down-wave: pullback has length, breakout, price_diff."""
        h = MarketStructureHelper()
        # Wave 0 (up): single candle, defaults.
        h.register_candle(_candle(open_time=1_000, histogram_value=0.5))
        h.register_candle(
            _candle(open_time=2_000, open=88, close=85, histogram_value=-0.3),  # flip → w-0 (up)
        )
        # Wave 1 (down): candle t=2000, LCO = min(88, 85) = 85.
        h.register_candle(
            _candle(open_time=3_000, open=105, close=110, histogram_value=0.4),  # flip → w-1 (down)
        )
        # Wave 2 (up): candle t=3000, HCO = max(105, 110) = 110.
        h.register_candle(
            _candle(open_time=4_000, histogram_value=-0.1),  # flip → w-2 (up)
        )

        w2 = h.wave_registry[2]
        assert w2.side == "up"
        assert w2.pullback is not None
        assert w2.pullback.breakout_level == 85.0
        assert w2.pullback.price_diff == pytest.approx(25.0)  # 110 - 85
        assert w2.pullback.atr_factor is None

    def test_correction_factor_none_when_bottom_is_first_wave(self) -> None:
        """When the bottom is the first wave, there is no top before it."""
        h = MarketStructureHelper()
        # Wave 0 (down): LCO = min(88, 85) = 85.
        h.register_candle(
            _candle(open_time=1_000, open=88, close=85, histogram_value=-0.5),
        )
        h.register_candle(
            _candle(open_time=2_000, open=105, close=110, histogram_value=0.3),  # flip → w-0 (down)
        )
        # Wave 1 (up): HCO = max(105, 110) = 110.
        h.register_candle(
            _candle(open_time=3_000, histogram_value=-0.2),  # flip → w-1 (up)
        )

        w1 = h.wave_registry[1]
        assert w1.side == "up"
        assert w1.pullback is not None
        # getTopBefore(w-0): w-0 is at idx 0, side=down, topIdx = -1 → None.
        assert w1.pullback.correction_factor is None

    def test_correction_factor_computed(self) -> None:
        """Three waves (top → bottom → up): correction_factor is the
        retracement ratio of the prior run."""
        h = MarketStructureHelper()
        # Wave 0 (up): HCO = max(120, 118) = 120.
        h.register_candle(
            _candle(open_time=1_000, open=120, close=118, histogram_value=0.5),
        )
        h.register_candle(
            _candle(open_time=2_000, open=88, close=85, histogram_value=-0.3),  # flip → w-0 (up)
        )
        # Wave 1 (down): candle t=2000, LCO = min(88, 85) = 85.
        h.register_candle(
            _candle(open_time=3_000, open=105, close=110, histogram_value=0.4),  # flip → w-1 (down)
        )
        # Wave 2 (up): candle t=3000, HCO = max(105, 110) = 110.
        h.register_candle(
            _candle(open_time=4_000, histogram_value=-0.1),  # flip → w-2 (up)
        )

        w2 = h.wave_registry[2]
        assert w2.side == "up"
        assert w2.pullback is not None
        # getTopBefore(w-1 at idx 1, side=down) → idx 0 → w-0 (up).
        # previousTopHigh = 120, bottomCloseOrOpen = 85.
        # correction_factor = (110 - 85) / (120 - 85) = 25/35
        assert w2.pullback.correction_factor == pytest.approx(25.0 / 35.0)

    def test_length_with_multi_candle_waves(self) -> None:
        """Length = bar distance from bottom's LCO to the up-wave's HCO."""
        h = MarketStructureHelper()
        # Wave 0 (down): 3 candles.
        h.register_candle(
            _candle(open_time=1_000, open=100, close=95, histogram_value=-0.5),
        )
        h.register_candle(
            _candle(
                open_time=2_000, open=92, close=90, histogram_value=-0.3
            ),  # LCO: min(92,90)=90 at pos 1
        )
        h.register_candle(
            _candle(open_time=3_000, open=93, close=95, histogram_value=-0.1),  # pos 2 (after LCO)
        )
        h.register_candle(
            _candle(
                open_time=4_000, open=98, close=100, histogram_value=0.4
            ),  # flip → w-0 (down, 3 candles)
        )
        # Wave 1 (up): 3 candles.
        h.register_candle(
            _candle(
                open_time=5_000, open=105, close=110, histogram_value=0.6
            ),  # HCO: max(105,110)=110 at pos 1
        )
        h.register_candle(
            _candle(open_time=6_000, open=107, close=108, histogram_value=0.2),
        )
        h.register_candle(
            _candle(open_time=7_000, histogram_value=-0.1),  # flip → w-1 (up, 3 candles)
        )

        w1 = h.wave_registry[1]
        assert w1.side == "up"
        assert w1.pullback is not None
        # bottom_candle_distance = 3 - 1 - 1 = 1 (one candle after LCO in w-0)
        # hco_pos = 1 (one candle before HCO in w-1)
        # length = 1 + 1 = 2
        assert w1.pullback.length == 2

    def test_price_diff_is_positive(self) -> None:
        """Up-wave pullback price_diff is positive (price rose)."""
        h = MarketStructureHelper()
        h.register_candle(
            _candle(open_time=1_000, open=88, close=85, histogram_value=-0.5),
        )
        h.register_candle(
            _candle(open_time=2_000, open=105, close=110, histogram_value=0.3),  # flip → w-0 (down)
        )
        h.register_candle(
            _candle(open_time=3_000, histogram_value=-0.2),  # flip → w-1 (up)
        )

        w1 = h.wave_registry[1]
        assert w1.pullback is not None
        assert w1.pullback.price_diff > 0


# ---------------------------------------------------------------------------
# Pullback from top (down-waves)
# ---------------------------------------------------------------------------


class TestPullbackFromTop:
    """Pullback for down-waves, computed from the last confirmed top."""

    def test_none_when_no_prior_top(self) -> None:
        """First down-wave has no top to pull back from → None."""
        h = MarketStructureHelper()
        h.register_candle(_candle(open_time=1_000, histogram_value=-0.5))
        h.register_candle(_candle(open_time=2_000, histogram_value=0.3))  # flip → w-0 (down)
        assert h.wave_registry[0].side == "down"
        assert h.wave_registry[0].pullback is None

    def test_basic_pullback_metrics(self) -> None:
        """Down-wave after an up-wave: pullback has breakout, price_diff."""
        h = MarketStructureHelper()
        # Wave 0 (down): single candle.
        h.register_candle(_candle(open_time=1_000, histogram_value=-0.5))
        h.register_candle(
            _candle(open_time=2_000, open=115, close=118, histogram_value=0.3),  # flip → w-0 (down)
        )
        # Wave 1 (up): candle t=2000, HCO = max(115, 118) = 118.
        h.register_candle(
            _candle(open_time=3_000, open=95, close=90, histogram_value=-0.4),  # flip → w-1 (up)
        )
        # Wave 2 (down): candle t=3000, LCO = min(95, 90) = 90.
        h.register_candle(
            _candle(open_time=4_000, histogram_value=0.1),  # flip → w-2 (down)
        )

        w2 = h.wave_registry[2]
        assert w2.side == "down"
        assert w2.pullback is not None
        assert w2.pullback.breakout_level == 118.0
        assert w2.pullback.price_diff == pytest.approx(-28.0)  # 90 - 118
        assert w2.pullback.atr_factor is None

    def test_correction_factor_computed(self) -> None:
        """Three waves (bottom → top → down): correction_factor is computed."""
        h = MarketStructureHelper()
        # Wave 0 (down): LCO = min(82, 80) = 80.
        h.register_candle(
            _candle(open_time=1_000, open=82, close=80, histogram_value=-0.5),
        )
        h.register_candle(
            _candle(open_time=2_000, open=115, close=118, histogram_value=0.3),  # flip → w-0 (down)
        )
        # Wave 1 (up): candle t=2000, HCO = max(115, 118) = 118.
        h.register_candle(
            _candle(open_time=3_000, open=95, close=90, histogram_value=-0.4),  # flip → w-1 (up)
        )
        # Wave 2 (down): candle t=3000, LCO = min(95, 90) = 90.
        h.register_candle(
            _candle(open_time=4_000, histogram_value=0.1),  # flip → w-2 (down)
        )

        w2 = h.wave_registry[2]
        assert w2.side == "down"
        assert w2.pullback is not None
        # getBottomBefore(w-1 at idx 1, side=up) → idx 0 → w-0 (down).
        # previousBottomLow = 80, topCloseOrOpen = 118.
        # correction_factor = (118 - 90) / (118 - 80) = 28/38
        assert w2.pullback.correction_factor == pytest.approx(28.0 / 38.0)

    def test_price_diff_is_negative(self) -> None:
        """Down-wave pullback price_diff is negative (price fell)."""
        h = MarketStructureHelper()
        h.register_candle(
            _candle(open_time=1_000, open=115, close=118, histogram_value=0.5),
        )
        h.register_candle(
            _candle(open_time=2_000, open=95, close=90, histogram_value=-0.3),  # flip → w-0 (up)
        )
        h.register_candle(
            _candle(open_time=3_000, histogram_value=0.2),  # flip → w-1 (down)
        )

        w1 = h.wave_registry[1]
        assert w1.pullback is not None
        assert w1.pullback.price_diff < 0

    def test_length_with_multi_candle_waves(self) -> None:
        """Length = bar distance from top's HCO to the down-wave's LCO."""
        h = MarketStructureHelper()
        # Wave 0 (up): 3 candles.
        h.register_candle(
            _candle(open_time=1_000, open=100, close=105, histogram_value=0.5),
        )
        h.register_candle(
            _candle(
                open_time=2_000, open=108, close=115, histogram_value=0.3
            ),  # HCO: max(108,115)=115 at pos 1
        )
        h.register_candle(
            _candle(open_time=3_000, open=110, close=112, histogram_value=0.1),  # pos 2 (after HCO)
        )
        h.register_candle(
            _candle(
                open_time=4_000, open=100, close=98, histogram_value=-0.4
            ),  # flip → w-0 (up, 3 candles)
        )
        # Wave 1 (down): 3 candles.
        h.register_candle(
            _candle(
                open_time=5_000, open=88, close=85, histogram_value=-0.3
            ),  # LCO: min(88,85)=85 at pos 1
        )
        h.register_candle(
            _candle(open_time=6_000, open=90, close=92, histogram_value=-0.1),
        )
        h.register_candle(
            _candle(open_time=7_000, histogram_value=0.2),  # flip → w-1 (down, 3 candles)
        )

        w1 = h.wave_registry[1]
        assert w1.side == "down"
        assert w1.pullback is not None
        # top_candle_distance = 3 - 1 - 1 = 1 (one candle after HCO in w-0)
        # lco_pos = 1 (one candle before LCO in w-1)
        # length = 1 + 1 = 2
        assert w1.pullback.length == 2


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestPullbackEdgeCases:
    """Boundary conditions and degenerate inputs."""

    def test_atr_factor_always_none(self) -> None:
        """Candle has no ``atr`` field → atr_factor stays None."""
        h = MarketStructureHelper()
        h.register_candle(
            _candle(open_time=1_000, open=88, close=85, histogram_value=-0.5),
        )
        h.register_candle(
            _candle(open_time=2_000, open=105, close=110, histogram_value=0.3),
        )
        h.register_candle(
            _candle(open_time=3_000, histogram_value=-0.2),
        )

        w1 = h.wave_registry[1]
        assert w1.pullback is not None
        assert w1.pullback.atr_factor is None

    def test_forming_wave_has_pullback(self) -> None:
        """``get_current_wave()`` also computes pullback."""
        h = MarketStructureHelper()
        # Wave 0 (down): LCO = min(88, 85) = 85.
        h.register_candle(
            _candle(open_time=1_000, open=88, close=85, histogram_value=-0.5),
        )
        h.register_candle(
            _candle(open_time=2_000, open=105, close=110, histogram_value=0.3),  # flip → w-0 (down)
        )
        # Forming up wave: candle t=2000, HCO = max(105, 110) = 110.

        wave = h.get_current_wave()
        assert wave is not None
        assert wave.side == "up"
        assert wave.pullback is not None
        assert wave.pullback.breakout_level == 85.0
        assert wave.pullback.price_diff == pytest.approx(25.0)

    def test_correction_factor_none_on_zero_denominator(self) -> None:
        """When previous top equals bottom level, correction_factor is None
        (guards against ZeroDivisionError that JS would silently produce as Infinity)."""
        h = MarketStructureHelper()
        # Wave 0 (up): HCO = max(100, 100) = 100 — same as bottom level below.
        h.register_candle(
            _candle(open_time=1_000, open=100, close=100, histogram_value=0.5),
        )
        h.register_candle(
            _candle(open_time=2_000, open=100, close=100, histogram_value=-0.3),  # flip → w-0 (up)
        )
        # Wave 1 (down): LCO = min(100, 100) = 100 — same as previous top.
        h.register_candle(
            _candle(open_time=3_000, open=105, close=110, histogram_value=0.4),  # flip → w-1 (down)
        )
        # Wave 2 (up).
        h.register_candle(
            _candle(open_time=4_000, histogram_value=-0.1),  # flip → w-2 (up)
        )

        w2 = h.wave_registry[2]
        assert w2.pullback is not None
        # denominator = previousTopHigh - bottomCloseOrOpen = 100 - 100 = 0
        assert w2.pullback.correction_factor is None
