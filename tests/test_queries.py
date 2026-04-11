"""Stage 5 tests for trivial query methods.

Covers ``get_current_wave`` (forming wave from the candle buffer),
``get_previous_top`` / ``get_previous_bottom``, and the
``include_forming_wave`` parameter on all four directional getters.
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
# get_current_wave
# ---------------------------------------------------------------------------


class TestGetCurrentWave:
    """``get_current_wave`` builds a forming wave from the candle buffer."""

    def test_returns_none_when_no_candles(self) -> None:
        h = MarketStructureHelper()
        assert h.get_current_wave() is None

    def test_returns_forming_wave_after_first_candle(self) -> None:
        h = MarketStructureHelper()
        h.register_candle(_candle(open_time=1_000, high=105.0, histogram_value=0.5))
        wave = h.get_current_wave()
        assert wave is not None
        assert wave.side == "up"
        assert wave.high.high == 105.0

    def test_forming_wave_side_tracks_histogram(self) -> None:
        """Negative histogram → forming wave is ``"down"``."""
        h = MarketStructureHelper()
        h.register_candle(_candle(open_time=1_000, histogram_value=-0.3))
        wave = h.get_current_wave()
        assert wave is not None
        assert wave.side == "down"

    def test_forming_wave_id_uses_forming_prefix(self) -> None:
        h = MarketStructureHelper()
        h.register_candle(_candle(open_time=1_000, histogram_value=0.5))
        wave = h.get_current_wave()
        assert wave is not None
        assert wave.id == "forming-0"

    def test_forming_wave_id_tracks_next_wave_id(self) -> None:
        """After one confirmed wave, the forming wave's ID reflects the
        next counter value."""
        h = MarketStructureHelper()
        h.register_candle(_candle(open_time=1_000, histogram_value=0.5))
        h.register_candle(_candle(open_time=2_000, histogram_value=-0.3))  # flip → w-0
        wave = h.get_current_wave()
        assert wave is not None
        assert wave.id == "forming-1"

    def test_does_not_increment_wave_counter(self) -> None:
        """Calling ``get_current_wave`` multiple times doesn't consume IDs."""
        h = MarketStructureHelper()
        h.register_candle(_candle(open_time=1_000, histogram_value=0.5))
        h.get_current_wave()
        h.get_current_wave()
        h.get_current_wave()
        # Next confirmed wave should still be w-0.
        h.register_candle(_candle(open_time=2_000, histogram_value=-0.3))  # flip
        assert h.wave_registry[0].id == "w-0"

    def test_forming_wave_updates_as_candles_arrive(self) -> None:
        h = MarketStructureHelper()
        h.register_candle(_candle(open_time=1_000, high=105.0, histogram_value=0.5))
        w1 = h.get_current_wave()
        assert w1 is not None
        assert w1.high.high == 105.0

        h.register_candle(_candle(open_time=2_000, high=110.0, histogram_value=0.3))
        w2 = h.get_current_wave()
        assert w2 is not None
        assert w2.high.high == 110.0

    def test_forming_wave_candles_match_buffer(self) -> None:
        h = MarketStructureHelper()
        h.register_candle(_candle(open_time=1_000, histogram_value=0.5))
        h.register_candle(_candle(open_time=2_000, histogram_value=0.3))
        wave = h.get_current_wave()
        assert wave is not None
        assert len(wave.candles) == 2
        assert wave.candles[0].open_time == 1_000
        assert wave.candles[1].open_time == 2_000

    def test_forming_wave_after_flip_contains_only_new_candles(self) -> None:
        h = MarketStructureHelper()
        h.register_candle(_candle(open_time=1_000, histogram_value=0.5))
        h.register_candle(_candle(open_time=2_000, histogram_value=-0.3))  # flip
        wave = h.get_current_wave()
        assert wave is not None
        assert len(wave.candles) == 1
        assert wave.candles[0].open_time == 2_000

    def test_zero_histogram_is_up_side(self) -> None:
        """``0.0`` sits on the "up" side (``>= 0``), consistent with
        ``_sign_flipped``."""
        h = MarketStructureHelper()
        h.register_candle(_candle(open_time=1_000, histogram_value=0.0))
        wave = h.get_current_wave()
        assert wave is not None
        assert wave.side == "up"


# ---------------------------------------------------------------------------
# get_previous_top / get_previous_bottom
# ---------------------------------------------------------------------------


class TestGetPreviousTop:
    """``get_previous_top`` returns the second-to-last confirmed up-wave."""

    def test_returns_none_with_no_waves(self) -> None:
        h = MarketStructureHelper()
        assert h.get_previous_top() is None

    def test_returns_none_with_one_top(self) -> None:
        h = MarketStructureHelper()
        h.register_candle(_candle(open_time=1_000, histogram_value=0.5))
        h.register_candle(_candle(open_time=2_000, histogram_value=-0.3))  # 1 up wave
        assert h.get_previous_top() is None

    def test_returns_first_top_when_two_exist(self) -> None:
        h = MarketStructureHelper()
        # up → down → up → (need another flip to confirm the second up)
        h.register_candle(_candle(open_time=1_000, histogram_value=0.5))
        h.register_candle(_candle(open_time=2_000, histogram_value=-0.3))  # up wave w-0
        h.register_candle(_candle(open_time=3_000, histogram_value=0.4))  # down wave w-1
        h.register_candle(_candle(open_time=4_000, histogram_value=-0.2))  # up wave w-2

        prev = h.get_previous_top()
        assert prev is not None
        assert prev is h.wave_registry[0]  # w-0, the first up wave


class TestGetPreviousBottom:
    """``get_previous_bottom`` returns the second-to-last confirmed down-wave."""

    def test_returns_none_with_no_waves(self) -> None:
        h = MarketStructureHelper()
        assert h.get_previous_bottom() is None

    def test_returns_none_with_one_bottom(self) -> None:
        h = MarketStructureHelper()
        h.register_candle(_candle(open_time=1_000, histogram_value=-0.5))
        h.register_candle(_candle(open_time=2_000, histogram_value=0.3))  # 1 down wave
        assert h.get_previous_bottom() is None

    def test_returns_first_bottom_when_two_exist(self) -> None:
        h = MarketStructureHelper()
        # down → up → down → (flip to confirm)
        h.register_candle(_candle(open_time=1_000, histogram_value=-0.5))
        h.register_candle(_candle(open_time=2_000, histogram_value=0.3))  # down wave w-0
        h.register_candle(_candle(open_time=3_000, histogram_value=-0.4))  # up wave w-1
        h.register_candle(_candle(open_time=4_000, histogram_value=0.2))  # down wave w-2

        prev = h.get_previous_bottom()
        assert prev is not None
        assert prev is h.wave_registry[0]  # w-0, the first down wave


# ---------------------------------------------------------------------------
# include_forming_wave parameter
# ---------------------------------------------------------------------------


class TestIncludeFormingWave:
    """The ``include_forming_wave`` kwarg on ``get_last_*`` / ``get_previous_*``."""

    # -- get_last_top --

    def test_get_last_top_includes_forming_up_wave(self) -> None:
        h = MarketStructureHelper()
        h.register_candle(_candle(open_time=1_000, histogram_value=0.5))
        # No confirmed waves yet, but forming wave is "up".
        assert h.get_last_top() is None  # default: confirmed only
        wave = h.get_last_top(include_forming_wave=True)
        assert wave is not None
        assert wave.side == "up"
        assert wave.id.startswith("forming-")

    def test_get_last_top_ignores_forming_down_wave(self) -> None:
        h = MarketStructureHelper()
        h.register_candle(_candle(open_time=1_000, histogram_value=-0.5))
        # Forming wave is "down" → get_last_top should still return None.
        assert h.get_last_top(include_forming_wave=True) is None

    # -- get_last_bottom --

    def test_get_last_bottom_includes_forming_down_wave(self) -> None:
        h = MarketStructureHelper()
        h.register_candle(_candle(open_time=1_000, histogram_value=-0.5))
        assert h.get_last_bottom() is None
        wave = h.get_last_bottom(include_forming_wave=True)
        assert wave is not None
        assert wave.side == "down"

    def test_get_last_bottom_ignores_forming_up_wave(self) -> None:
        h = MarketStructureHelper()
        h.register_candle(_candle(open_time=1_000, histogram_value=0.5))
        assert h.get_last_bottom(include_forming_wave=True) is None

    # -- get_previous_top --

    def test_get_previous_top_shifts_when_forming_matches(self) -> None:
        """When forming wave is ``"up"`` and ``include_forming_wave=True``,
        the forming wave becomes "last" — so "previous" returns the most
        recent *confirmed* up-wave."""
        h = MarketStructureHelper()
        h.register_candle(_candle(open_time=1_000, histogram_value=0.5))
        h.register_candle(_candle(open_time=2_000, histogram_value=-0.3))  # up wave w-0
        h.register_candle(_candle(open_time=3_000, histogram_value=0.4))  # down wave w-1
        # Forming wave is "up" (histogram = 0.4).

        # Without include_forming_wave: only 1 confirmed up wave → None.
        assert h.get_previous_top() is None

        # With include_forming_wave: forming up wave is "last", w-0 is "previous".
        prev = h.get_previous_top(include_forming_wave=True)
        assert prev is not None
        assert prev.id == "w-0"

    def test_get_previous_top_unaffected_when_forming_is_other_side(self) -> None:
        """Forming wave is ``"down"`` — ``include_forming_wave`` doesn't change
        the result for ``get_previous_top``."""
        h = MarketStructureHelper()
        h.register_candle(_candle(open_time=1_000, histogram_value=0.5))
        h.register_candle(_candle(open_time=2_000, histogram_value=-0.3))  # up wave w-0
        # Forming wave is "down" (histogram = -0.3).
        assert h.get_previous_top(include_forming_wave=True) is None

    # -- get_previous_bottom --

    def test_get_previous_bottom_shifts_when_forming_matches(self) -> None:
        h = MarketStructureHelper()
        h.register_candle(_candle(open_time=1_000, histogram_value=-0.5))
        h.register_candle(_candle(open_time=2_000, histogram_value=0.3))  # down wave w-0
        h.register_candle(_candle(open_time=3_000, histogram_value=-0.4))  # up wave w-1
        # Forming wave is "down" (histogram = -0.4).

        assert h.get_previous_bottom() is None

        prev = h.get_previous_bottom(include_forming_wave=True)
        assert prev is not None
        assert prev.id == "w-0"

    def test_get_previous_bottom_unaffected_when_forming_is_other_side(self) -> None:
        h = MarketStructureHelper()
        h.register_candle(_candle(open_time=1_000, histogram_value=-0.5))
        h.register_candle(_candle(open_time=2_000, histogram_value=0.3))  # down wave w-0
        # Forming wave is "up" (histogram = 0.3).
        assert h.get_previous_bottom(include_forming_wave=True) is None
