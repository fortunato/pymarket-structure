"""Stage 6 tests for backward scans — ``_determine_high_since`` / ``_determine_low_since``.

These methods walk ``_wave_registry`` backward to find how many bars ago
the current wave's extreme was last exceeded. They power ``high_since``
(for up waves) and ``low_since`` (for down waves) on the ``Wave`` object,
which Stage 11 uses to identify long-term swing extremes.
"""

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
# high_since (up waves)
# ---------------------------------------------------------------------------


class TestHighSince:
    """``high_since`` counts bars backward from the HCO candle to the last
    time ``max(close, open)`` exceeded that level."""

    def test_single_wave_hco_at_first_candle(self) -> None:
        """HCO at position 0, no prior waves → high_since = 0."""
        h = MarketStructureHelper()
        h.register_candle(
            _candle(
                open_time=1_000, open=115.0, close=112.0, histogram_value=0.5
            ),  # max(c,o)=115 ← HCO
        )
        h.register_candle(
            _candle(open_time=2_000, open=100.0, close=105.0, histogram_value=0.3),  # max(c,o)=105
        )
        h.register_candle(
            _candle(open_time=3_000, histogram_value=-0.2),  # flip
        )

        w = h.wave_registry[0]
        assert w.side == "up"
        assert w.high_since == 0

    def test_single_wave_hco_in_middle(self) -> None:
        """HCO at position 1 of 3 candles, no prior waves → high_since = 1."""
        h = MarketStructureHelper()
        h.register_candle(
            _candle(open_time=1_000, open=100.0, close=103.0, histogram_value=0.5),  # max(c,o)=103
        )
        h.register_candle(
            _candle(
                open_time=2_000, open=102.0, close=108.0, histogram_value=0.3
            ),  # max(c,o)=108 ← HCO
        )
        h.register_candle(
            _candle(open_time=3_000, open=107.0, close=106.0, histogram_value=0.1),  # max(c,o)=107
        )
        h.register_candle(
            _candle(open_time=4_000, histogram_value=-0.2),  # flip
        )

        w = h.wave_registry[0]
        assert w.high_since == 1

    def test_prior_wave_exceeds(self) -> None:
        """An older wave had a higher max(c,o) → high_since stops there."""
        h = MarketStructureHelper()
        # Wave 0 (up): [t=1000, max(c,o)=120].
        h.register_candle(
            _candle(open_time=1_000, open=120.0, close=118.0, histogram_value=0.5),
        )
        h.register_candle(
            _candle(open_time=2_000, histogram_value=-0.3),  # flip → w-0
        )
        # Wave 1 (down): [t=2000, defaults].
        # The flip candle carries the OHLC for the *next* wave's first candle.
        h.register_candle(
            _candle(
                open_time=3_000, open=110.0, close=108.0, histogram_value=0.4
            ),  # flip → w-1 (down)
        )
        # Wave 2 (up): [t=3000, max(c,o)=110]. HCO at pos 0.
        h.register_candle(
            _candle(open_time=4_000, histogram_value=-0.1),  # flip → w-2
        )

        w2 = h.wave_registry[2]
        assert w2.side == "up"
        # hco_pos=0. Scan:
        #   w-1 (1 candle): max(c,o)=100.5, not > 110 → count=1
        #   w-0 (1 candle): max(c,o)=120 > 110 → found!
        #     local_idx=0, return 1 + 1 - 1 - 0 = 1
        assert w2.high_since == 1

    def test_no_prior_wave_exceeds(self) -> None:
        """No older wave exceeded → high_since = total distance to start of history."""
        h = MarketStructureHelper()
        # Wave 0 (up): [t=1000, max(c,o)=105].
        h.register_candle(
            _candle(open_time=1_000, open=105.0, close=103.0, histogram_value=0.5),
        )
        h.register_candle(
            _candle(open_time=2_000, histogram_value=-0.3),  # flip → w-0
        )
        # Wave 1 (down): [t=2000, defaults].
        h.register_candle(
            _candle(open_time=3_000, open=115.0, close=112.0, histogram_value=0.4),  # flip → w-1
        )
        # Wave 2 (up): [t=3000, max(c,o)=115] — new all-time high.
        h.register_candle(
            _candle(open_time=4_000, histogram_value=-0.1),  # flip → w-2
        )

        w2 = h.wave_registry[2]
        # hco_pos=0, w-1 max(c,o)=100.5 (count=1), w-0 max(c,o)=105 (count=2).
        # No more waves → return 2.
        assert w2.high_since == 2

    def test_not_computed_for_down_waves(self) -> None:
        """Down waves keep the default ``high_since = 0``."""
        h = MarketStructureHelper()
        h.register_candle(_candle(open_time=1_000, histogram_value=-0.5))
        h.register_candle(_candle(open_time=2_000, histogram_value=0.3))  # flip → down wave
        assert h.wave_registry[0].side == "down"
        assert h.wave_registry[0].high_since == 0

    def test_multiple_waves_scanned_before_exceeding(self) -> None:
        """Scan passes through several non-exceeding waves before finding one."""
        h = MarketStructureHelper()
        # Wave 0 (up): [t=1000, max(c,o)=130] — the one that will exceed.
        h.register_candle(
            _candle(open_time=1_000, open=130.0, close=128.0, histogram_value=0.5),
        )
        h.register_candle(_candle(open_time=2_000, histogram_value=-0.3))  # flip → w-0
        # Wave 1 (down): [t=2000, defaults].
        h.register_candle(_candle(open_time=3_000, histogram_value=0.4))  # flip → w-1
        # Wave 2 (up): [t=3000, defaults, max(c,o)=100.5].
        h.register_candle(_candle(open_time=4_000, histogram_value=-0.2))  # flip → w-2
        # Wave 3 (down): [t=4000, defaults].
        # Flip candle carries the OHLC for wave 4's first candle.
        h.register_candle(
            _candle(open_time=5_000, open=120.0, close=118.0, histogram_value=0.1),  # flip → w-3
        )
        # Wave 4 (up): [t=5000, max(c,o)=120] — the target. HCO at pos 0.
        h.register_candle(_candle(open_time=6_000, histogram_value=-0.1))  # flip → w-4

        w4 = h.wave_registry[4]
        assert w4.side == "up"
        # hco_pos=0. Scan backward:
        #   w-3 (1 candle): max(c,o)=100.5, not > 120 → count=1
        #   w-2 (1 candle): max(c,o)=100.5, not > 120 → count=2
        #   w-1 (1 candle): max(c,o)=100.5, not > 120 → count=3
        #   w-0 (1 candle): max(c,o)=130 > 120 → found!
        #     local_idx=0, return 3 + 1 - 1 - 0 = 3
        assert w4.high_since == 3


# ---------------------------------------------------------------------------
# low_since (down waves)
# ---------------------------------------------------------------------------


class TestLowSince:
    """``low_since`` counts bars backward from the LCO candle to the last
    time ``min(close, open)`` went lower than that level."""

    def test_single_wave_lco_at_first_candle(self) -> None:
        h = MarketStructureHelper()
        h.register_candle(
            _candle(
                open_time=1_000, open=85.0, close=88.0, histogram_value=-0.5
            ),  # min(c,o)=85 ← LCO
        )
        h.register_candle(
            _candle(open_time=2_000, open=90.0, close=95.0, histogram_value=-0.3),  # min(c,o)=90
        )
        h.register_candle(
            _candle(open_time=3_000, histogram_value=0.2),  # flip
        )

        w = h.wave_registry[0]
        assert w.side == "down"
        assert w.low_since == 0

    def test_single_wave_lco_in_middle(self) -> None:
        h = MarketStructureHelper()
        h.register_candle(
            _candle(open_time=1_000, open=95.0, close=93.0, histogram_value=-0.5),  # min(c,o)=93
        )
        h.register_candle(
            _candle(
                open_time=2_000, open=88.0, close=85.0, histogram_value=-0.3
            ),  # min(c,o)=85 ← LCO
        )
        h.register_candle(
            _candle(open_time=3_000, open=90.0, close=92.0, histogram_value=-0.1),  # min(c,o)=90
        )
        h.register_candle(
            _candle(open_time=4_000, histogram_value=0.2),  # flip
        )

        w = h.wave_registry[0]
        assert w.low_since == 1

    def test_prior_wave_exceeds(self) -> None:
        """An older wave had a lower min(c,o) → low_since stops there."""
        h = MarketStructureHelper()
        # Wave 0 (down): [t=1000, min(c,o)=80] — lower than the target.
        h.register_candle(
            _candle(open_time=1_000, open=80.0, close=82.0, histogram_value=-0.5),
        )
        h.register_candle(
            _candle(open_time=2_000, histogram_value=0.3),  # flip → w-0 (down)
        )
        # Wave 1 (up): [t=2000, defaults].
        # Flip candle carries OHLC for next wave's first candle.
        h.register_candle(
            _candle(
                open_time=3_000, open=88.0, close=90.0, histogram_value=-0.4
            ),  # flip → w-1 (up)
        )
        # Wave 2 (down): [t=3000, min(c,o)=88]. LCO at pos 0.
        h.register_candle(
            _candle(open_time=4_000, histogram_value=0.1),  # flip → w-2 (down)
        )

        w2 = h.wave_registry[2]
        assert w2.side == "down"
        # lco_pos=0. Scan:
        #   w-1 (1 candle): min(c,o)=100, not < 88 → count=1
        #   w-0 (1 candle): min(c,o)=80 < 88 → found!
        #     local_idx=0, return 1 + 1 - 1 - 0 = 1
        assert w2.low_since == 1

    def test_no_prior_wave_exceeds(self) -> None:
        """All-time low → low_since = total distance to start of history."""
        h = MarketStructureHelper()
        # Wave 0 (down): [t=1000, min(c,o)=92].
        h.register_candle(
            _candle(open_time=1_000, open=95.0, close=92.0, histogram_value=-0.5),
        )
        h.register_candle(
            _candle(open_time=2_000, histogram_value=0.3),  # flip → w-0
        )
        # Wave 1 (up): [t=2000, defaults].
        h.register_candle(
            _candle(open_time=3_000, open=85.0, close=87.0, histogram_value=-0.4),  # flip → w-1
        )
        # Wave 2 (down): [t=3000, min(c,o)=85] — new all-time low.
        h.register_candle(
            _candle(open_time=4_000, histogram_value=0.1),  # flip → w-2
        )

        w2 = h.wave_registry[2]
        # lco_pos=0, w-1 min(c,o)=100 (count=1), w-0 min(c,o)=92 (count=2).
        # No more → return 2.
        assert w2.low_since == 2

    def test_not_computed_for_up_waves(self) -> None:
        h = MarketStructureHelper()
        h.register_candle(_candle(open_time=1_000, histogram_value=0.5))
        h.register_candle(_candle(open_time=2_000, histogram_value=-0.3))  # flip → up wave
        assert h.wave_registry[0].side == "up"
        assert h.wave_registry[0].low_since == 0


# ---------------------------------------------------------------------------
# Forming wave
# ---------------------------------------------------------------------------


class TestFormingWaveSince:
    """``get_current_wave()`` also computes ``high_since`` / ``low_since``."""

    def test_forming_up_wave_has_high_since(self) -> None:
        h = MarketStructureHelper()
        h.register_candle(
            _candle(open_time=1_000, open=100.0, close=103.0, histogram_value=0.5),
        )
        h.register_candle(
            _candle(open_time=2_000, open=110.0, close=108.0, histogram_value=0.3),  # HCO at pos 1
        )
        wave = h.get_current_wave()
        assert wave is not None
        assert wave.side == "up"
        assert wave.high_since == 1  # hco_pos=1, no prior waves

    def test_forming_down_wave_has_low_since(self) -> None:
        h = MarketStructureHelper()
        h.register_candle(
            _candle(open_time=1_000, open=95.0, close=93.0, histogram_value=-0.5),
        )
        h.register_candle(
            _candle(open_time=2_000, open=85.0, close=88.0, histogram_value=-0.3),  # LCO at pos 1
        )
        wave = h.get_current_wave()
        assert wave is not None
        assert wave.side == "down"
        assert wave.low_since == 1

    def test_forming_wave_scans_confirmed_registry(self) -> None:
        """The forming wave's backward scan looks through confirmed waves."""
        h = MarketStructureHelper()
        # Wave 0 (up): [t=1000, max(c,o)=120].
        h.register_candle(
            _candle(open_time=1_000, open=120.0, close=118.0, histogram_value=0.5),
        )
        h.register_candle(
            _candle(open_time=2_000, histogram_value=-0.3),  # flip → w-0
        )
        # Wave 1 (down): [t=2000, defaults].
        # Flip candle carries OHLC for the forming wave.
        h.register_candle(
            _candle(open_time=3_000, open=110.0, close=108.0, histogram_value=0.4),  # flip → w-1
        )
        # Forming up wave: [t=3000, max(c,o)=110]. HCO at pos 0.

        wave = h.get_current_wave()
        assert wave is not None
        assert wave.side == "up"
        # hco_pos=0. Scan:
        #   w-1 (1 candle): max(c,o)=100.5, not > 110 → count=1
        #   w-0 (1 candle): max(c,o)=120 > 110 → found!
        #     local_idx=0, return 1 + 1 - 1 - 0 = 1
        assert wave.high_since == 1
