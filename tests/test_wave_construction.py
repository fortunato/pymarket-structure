"""Stage 4 tests for wave construction — turning the ``_wave_candles``
buffer into ``Wave`` objects when a histogram sign-flip occurs.

``reportPrivateUsage`` is disabled for the same reason as in
``test_register_candle.py``: we need to inspect ``_wave_candles`` and
other internal state to verify the construction machinery.
"""

# pyright: reportPrivateUsage=false

import pytest

from market_structure import MarketStructureHelper
from market_structure.types import Candle, Wave

# ---------------------------------------------------------------------------
# Test-local candle factories
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


def _trigger_one_up_wave(h: MarketStructureHelper) -> None:
    """Register three positive-histogram candles, then flip to negative.

    After this call, ``h.wave_registry`` contains exactly one ``"up"``
    wave built from the three candles.
    """
    h.register_candle(
        _candle(
            open_time=1_000, open=100.0, high=105.0, low=98.0, close=103.0, histogram_value=0.5
        ),
    )
    h.register_candle(
        _candle(
            open_time=2_000, open=102.0, high=110.0, low=100.0, close=108.0, histogram_value=0.3
        ),
    )
    h.register_candle(
        _candle(
            open_time=3_000, open=107.0, high=109.0, low=101.0, close=106.0, histogram_value=0.1
        ),
    )
    # Flip → wave constructed from the three candles above.
    h.register_candle(
        _candle(
            open_time=4_000, open=104.0, high=106.0, low=97.0, close=98.0, histogram_value=-0.2
        ),
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestWaveConstruction:
    """Verify that a sign-flip creates a correctly populated Wave."""

    def test_single_flip_creates_one_wave(self) -> None:
        h = MarketStructureHelper()
        _trigger_one_up_wave(h)
        assert len(h.wave_registry) == 1

    def test_wave_side_from_histogram_sign(self) -> None:
        h = MarketStructureHelper()
        _trigger_one_up_wave(h)
        assert h.wave_registry[0].side == "up"

    def test_down_wave_side(self) -> None:
        """Negative histogram candles produce a ``"down"`` wave."""
        h = MarketStructureHelper()
        h.register_candle(_candle(open_time=1_000, histogram_value=-0.5))
        h.register_candle(_candle(open_time=2_000, histogram_value=-0.3))
        h.register_candle(_candle(open_time=3_000, histogram_value=0.2))  # flip
        assert h.wave_registry[0].side == "down"

    def test_wave_high_is_candle_with_highest_high(self) -> None:
        h = MarketStructureHelper()
        _trigger_one_up_wave(h)
        w = h.wave_registry[0]
        # Candle at t=2000 has high=110 — the highest of the three.
        assert w.high.high == 110.0
        assert w.high.open_time == 2_000

    def test_wave_low_is_candle_with_lowest_low(self) -> None:
        h = MarketStructureHelper()
        _trigger_one_up_wave(h)
        w = h.wave_registry[0]
        # Candle at t=1000 has low=98 — the lowest of the three.
        assert w.low.low == 98.0
        assert w.low.open_time == 1_000

    def test_wave_highest_close(self) -> None:
        h = MarketStructureHelper()
        _trigger_one_up_wave(h)
        w = h.wave_registry[0]
        assert w.highest_close.close == 108.0
        assert w.highest_close.open_time == 2_000

    def test_wave_lowest_close(self) -> None:
        h = MarketStructureHelper()
        _trigger_one_up_wave(h)
        w = h.wave_registry[0]
        assert w.lowest_close.close == 103.0
        assert w.lowest_close.open_time == 1_000

    def test_wave_highest_close_or_open(self) -> None:
        """max(close, open) per candle: t=1000→103, t=2000→108, t=3000→107."""
        h = MarketStructureHelper()
        _trigger_one_up_wave(h)
        w = h.wave_registry[0]
        assert w.highest_close_or_open.open_time == 2_000

    def test_wave_lowest_close_or_open(self) -> None:
        """min(close, open) per candle: t=1000→100, t=2000→102, t=3000→106."""
        h = MarketStructureHelper()
        _trigger_one_up_wave(h)
        w = h.wave_registry[0]
        assert w.lowest_close_or_open.open_time == 1_000

    def test_wave_candles_stored_as_tuple(self) -> None:
        h = MarketStructureHelper()
        _trigger_one_up_wave(h)
        w = h.wave_registry[0]
        assert len(w.candles) == 3
        assert w.candles[0].open_time == 1_000
        assert w.candles[2].open_time == 3_000

    def test_wave_id_is_deterministic(self) -> None:
        h = MarketStructureHelper()
        _trigger_one_up_wave(h)
        assert h.wave_registry[0].id == "w-0"

    def test_wave_ids_are_unique(self) -> None:
        """Two successive waves get different IDs."""
        h = MarketStructureHelper()
        h.register_candle(_candle(open_time=1_000, histogram_value=0.5))
        h.register_candle(_candle(open_time=2_000, histogram_value=-0.3))  # flip
        h.register_candle(_candle(open_time=3_000, histogram_value=0.4))  # flip
        assert len(h.wave_registry) == 2
        assert h.wave_registry[0].id != h.wave_registry[1].id
        assert h.wave_registry[1].id == "w-1"

    def test_formation_bar_index(self) -> None:
        """formationBarIndex = index of the candle that triggered the flip.

        Four candles registered (indices 0-3), flip happens at the 4th
        (index 3).
        """
        h = MarketStructureHelper()
        _trigger_one_up_wave(h)
        assert h.wave_registry[0].formation_bar_index == 3


class TestExtremumIndices:
    """Verify the stored row indices for extremum candles.

    These indices avoid re-scanning wave candles during backward scans
    (``_determine_high_since`` etc. in Stage 6).
    """

    def test_high_idx_and_low_idx(self) -> None:
        h = MarketStructureHelper()
        _trigger_one_up_wave(h)
        w = h.wave_registry[0]
        # Wave candles are at global indices 0, 1, 2.
        # high (t=2000) is at index 1, low (t=1000) is at index 0.
        assert w.high_idx == 1
        assert w.low_idx == 0

    def test_close_or_open_indices(self) -> None:
        h = MarketStructureHelper()
        _trigger_one_up_wave(h)
        w = h.wave_registry[0]
        assert w.highest_close_or_open_idx == 1  # t=2000
        assert w.lowest_close_or_open_idx == 0  # t=1000

    def test_indices_offset_by_prior_waves(self) -> None:
        """The second wave's indices are offset by the first wave's length."""
        h = MarketStructureHelper()
        # Wave 1: two candles (indices 0, 1), then flip at index 2.
        h.register_candle(_candle(open_time=1_000, histogram_value=0.5))
        h.register_candle(_candle(open_time=2_000, histogram_value=0.3))
        h.register_candle(_candle(open_time=3_000, histogram_value=-0.2))  # flip
        # Wave 2: one candle (index 2), then flip at index 3.
        h.register_candle(_candle(open_time=4_000, histogram_value=0.4))  # flip

        w2 = h.wave_registry[1]
        # The single candle of wave 2 is at global index 2 (3rd candle, 0-based).
        assert w2.high_idx == 2
        assert w2.low_idx == 2

    def test_single_candle_wave(self) -> None:
        """A wave with exactly one candle: all extremes point at that candle."""
        h = MarketStructureHelper()
        c = _candle(
            open_time=1_000, open=100.0, high=105.0, low=95.0, close=102.0, histogram_value=0.5
        )
        h.register_candle(c)
        h.register_candle(_candle(open_time=2_000, histogram_value=-0.3))  # flip

        w = h.wave_registry[0]
        assert w.high is c
        assert w.low is c
        assert w.highest_close is c
        assert w.lowest_close is c
        assert w.highest_close_or_open is c
        assert w.lowest_close_or_open is c
        assert w.candles == (c,)


class TestDirectionalArrays:
    """Constructed waves are partitioned into ``_top_waves`` / ``_bottom_waves``."""

    def test_up_wave_accessible_via_get_last_top(self) -> None:
        h = MarketStructureHelper()
        _trigger_one_up_wave(h)
        assert h.get_last_top() is not None
        assert h.get_last_top() is h.wave_registry[0]
        assert h.get_last_bottom() is None

    def test_down_wave_accessible_via_get_last_bottom(self) -> None:
        h = MarketStructureHelper()
        h.register_candle(_candle(open_time=1_000, histogram_value=-0.5))
        h.register_candle(_candle(open_time=2_000, histogram_value=0.3))  # flip
        assert h.get_last_bottom() is not None
        assert h.get_last_bottom() is h.wave_registry[0]
        assert h.get_last_top() is None

    def test_alternating_waves_partition_correctly(self) -> None:
        """up → down → up produces one top and one bottom (the first up
        wave is in the registry, the forming wave is not)."""
        h = MarketStructureHelper()
        h.register_candle(_candle(open_time=1_000, histogram_value=0.5))
        h.register_candle(_candle(open_time=2_000, histogram_value=-0.3))  # flip → up wave
        h.register_candle(_candle(open_time=3_000, histogram_value=0.2))  # flip → down wave

        assert len(h.wave_registry) == 2
        assert h.get_last_top() is h.wave_registry[0]
        assert h.get_last_bottom() is h.wave_registry[1]


class TestEviction:
    """Waves beyond ``max_waves`` are evicted FIFO."""

    def test_registry_does_not_exceed_max_waves(self) -> None:
        h = MarketStructureHelper(max_waves=3)

        # Generate 6 sign-flips → 6 waves pushed, but max is 3.
        sign = 0.5
        for i in range(7):
            h.register_candle(
                _candle(open_time=1_000 * (i + 1), histogram_value=sign),
            )
            sign = -sign

        assert len(h.wave_registry) <= 3

    def test_evicted_wave_removed_from_directional_array(self) -> None:
        """When a top wave is evicted, ``_top_waves`` shrinks too."""
        h = MarketStructureHelper(max_waves=2)

        # Generate 4 flips → 4 waves, cap at 2.
        sign = 0.5
        for i in range(5):
            h.register_candle(
                _candle(open_time=1_000 * (i + 1), histogram_value=sign),
            )
            sign = -sign

        assert len(h.wave_registry) == 2
        # Directional arrays should not contain evicted waves.
        all_wave_ids = {w.id for w in h.wave_registry}
        for w in h._top_waves:
            assert w.id in all_wave_ids
        for w in h._bottom_waves:
            assert w.id in all_wave_ids

    def test_default_max_waves_is_200(self) -> None:
        h = MarketStructureHelper()
        assert h.max_waves == 200


class TestMultipleWaves:
    """End-to-end scenarios with several waves."""

    def test_multi_flip_builds_correct_count(self) -> None:
        """7 candles with 3 sign-flips → 3 waves in the registry."""
        h = MarketStructureHelper()
        h.register_candle(_candle(open_time=1_000, histogram_value=0.4))
        h.register_candle(_candle(open_time=2_000, histogram_value=0.2))
        h.register_candle(_candle(open_time=3_000, histogram_value=-0.1))  # flip 1
        h.register_candle(_candle(open_time=4_000, histogram_value=-0.3))
        h.register_candle(_candle(open_time=5_000, histogram_value=0.2))  # flip 2
        h.register_candle(_candle(open_time=6_000, histogram_value=0.5))
        h.register_candle(_candle(open_time=7_000, histogram_value=-0.4))  # flip 3

        assert len(h.wave_registry) == 3
        assert h.wave_registry[0].side == "up"  # first wave: positive histogram
        assert h.wave_registry[1].side == "down"  # second wave: negative
        assert h.wave_registry[2].side == "up"  # third wave: positive again

    def test_wave_candle_counts_match_flip_boundaries(self) -> None:
        """Each wave's candle tuple has the right number of candles."""
        h = MarketStructureHelper()
        h.register_candle(_candle(open_time=1_000, histogram_value=0.4))
        h.register_candle(_candle(open_time=2_000, histogram_value=0.2))
        h.register_candle(_candle(open_time=3_000, histogram_value=-0.1))  # flip
        h.register_candle(_candle(open_time=4_000, histogram_value=-0.3))
        h.register_candle(_candle(open_time=5_000, histogram_value=0.2))  # flip

        assert len(h.wave_registry[0].candles) == 2  # t=1000, t=2000
        assert len(h.wave_registry[1].candles) == 2  # t=3000, t=4000

    def test_no_wave_without_flip(self) -> None:
        """Candles with the same histogram sign produce no waves."""
        h = MarketStructureHelper()
        for i in range(10):
            h.register_candle(
                _candle(open_time=1_000 * (i + 1), histogram_value=0.3),
            )
        assert h.wave_registry == ()

    def test_high_since_computed_for_up_wave(self) -> None:
        """high_since reflects the backward distance from the HCO candle."""
        h = MarketStructureHelper()
        _trigger_one_up_wave(h)
        # HCO (t=2000, max(close,open)=108) is at position 1 in the wave.
        # No prior waves → high_since = 1.
        assert h.wave_registry[0].high_since == 1
        assert h.wave_registry[0].low_since == 0  # not computed for up waves

    def test_pullback_none_for_first_wave(self) -> None:
        """First wave has no prior opposite wave → pullback is None."""
        h = MarketStructureHelper()
        _trigger_one_up_wave(h)
        assert h.wave_registry[0].pullback is None


# ---------------------------------------------------------------------------
# Alternation invariant — runtime assert guards the double-pattern body
# from malformed registries.
# ---------------------------------------------------------------------------


def _minimal_wave(wave_id: str, side: str, low: float, high: float, fbi: int) -> Wave:
    """Build a Wave with the minimum fields the zone methods read.

    The double-pattern path uses ``wave.id``, ``wave.side``, ``wave.low``,
    ``wave.high``, ``wave.low_idx``, ``wave.high_idx`` and
    ``wave.formation_bar_index``. Everything else can be a stub.
    """
    c = Candle(
        open_time=fbi * 1000,
        open=low,
        high=high,
        low=low,
        close=high,
        volume=1.0,
        histogram_value=-0.5 if side == "down" else 0.5,
    )
    return Wave(
        id=wave_id,
        side=side,  # type: ignore[arg-type]
        formation_bar_index=fbi,
        high=c,
        low=c,
        highest_close=c,
        lowest_close=c,
        highest_close_or_open=c,
        lowest_close_or_open=c,
        high_idx=fbi,
        low_idx=fbi,
        highest_close_or_open_idx=fbi,
        lowest_close_or_open_idx=fbi,
        candles=(c,),
    )


def test_alternation_assertion_fires_on_adjacent_same_side_waves() -> None:
    """Two adjacent same-side waves in ``_wave_registry`` (no
    opposite-side wave between them) must trigger ``AssertionError`` when
    the double-pattern body runs.
    """
    h = MarketStructureHelper()
    w0 = _minimal_wave("w-0", "down", low=100.0, high=101.0, fbi=0)
    w1 = _minimal_wave("w-1", "down", low=100.1, high=101.0, fbi=1)
    # Deliberately malformed: two adjacent down-waves with no up-wave
    # between them. Under TSI-driven emission this is impossible.
    h._wave_registry = [w0, w1]
    h._bottom_waves = [w0, w1]
    with pytest.raises(AssertionError):
        h.get_support_zones()


def test_alternation_assertion_fires_on_adjacent_same_side_waves_top() -> None:
    """Mirror of the above test for the resistance path."""
    h = MarketStructureHelper()
    w0 = _minimal_wave("w-0", "up", low=99.0, high=110.0, fbi=0)
    w1 = _minimal_wave("w-1", "up", low=99.0, high=109.9, fbi=1)
    h._wave_registry = [w0, w1]
    h._top_waves = [w0, w1]
    with pytest.raises(AssertionError):
        h.get_resistance_zones()
