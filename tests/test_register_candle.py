"""Stage 3 tests for ``MarketStructureHelper.register_candle``.

Covers the state machine around the ``_wave_candles`` buffer:

- Idempotency: duplicate ``open_time`` values are silently dropped.
- Sign-flip detection: histogram crossing zero clears the buffer.
- Wave construction is still deferred to Stage 4 — ``get_current_wave()``
  must keep returning ``None`` even as the buffer grows.

``reportPrivateUsage`` is intentionally disabled for this module: the
whole point of these tests is to verify the shape of ``_wave_candles``
and the behavior of ``_sign_flipped`` — both deliberately private in
the public API, both load-bearing in the state machine under test.
"""

# pyright: reportPrivateUsage=false

import pytest

from market_structure import MarketStructureHelper
from market_structure.types import Candle


def _candle(
    open_time: int = 1_000, close: float = 100.5, *, histogram_value: float = 0.0
) -> Candle:
    """Build a minimal valid Candle with the fields this stage cares about.

    OHLV values are placeholders — Stage 3 only ingests candles and
    buffers them; it never inspects any field other than ``open_time``.
    """
    return Candle(
        open_time=open_time,
        open=100.0,
        high=101.0,
        low=99.0,
        close=close,
        volume=1.0,
        histogram_value=histogram_value,
    )


class TestDedup:
    """``register_candle`` is idempotent on ``open_time``.

    Freqtrade re-emits the current (forming) candle on every tick while
    it is still open. The helper must treat repeated ``open_time`` values
    as a single logical ingest, not as new candles stacking up.
    """

    def test_first_candle_counts_once(self) -> None:
        h = MarketStructureHelper()
        h.register_candle(_candle(open_time=1_000, histogram_value=0.5))
        assert h.total_candles_registered == 1
        assert len(h._wave_candles) == 1

    def test_duplicate_open_time_is_ignored(self) -> None:
        h = MarketStructureHelper()
        h.register_candle(_candle(open_time=1_000, histogram_value=0.5))
        h.register_candle(_candle(open_time=1_000, close=999.0, histogram_value=0.7))

        assert h.total_candles_registered == 1
        assert len(h._wave_candles) == 1

    def test_distinct_open_times_both_count(self) -> None:
        h = MarketStructureHelper()
        h.register_candle(_candle(open_time=1_000, histogram_value=0.5))
        h.register_candle(_candle(open_time=2_000, histogram_value=0.6))

        assert h.total_candles_registered == 2
        assert len(h._wave_candles) == 2

    def test_duplicate_then_new_still_counts_two(self) -> None:
        """Three calls (1_000, 1_000, 2_000) => counter == 2."""
        h = MarketStructureHelper()
        h.register_candle(_candle(open_time=1_000, histogram_value=0.5))
        h.register_candle(_candle(open_time=1_000, histogram_value=0.5))
        h.register_candle(_candle(open_time=2_000, histogram_value=0.6))

        assert h.total_candles_registered == 2


class TestSignFlip:
    """The ``_wave_candles`` buffer clears when the histogram crosses zero."""

    def test_first_candle_enters_buffer_without_flip(self) -> None:
        """With no prior value, there is nothing to flip *from* — just append."""
        h = MarketStructureHelper()
        h.register_candle(_candle(open_time=1_000, histogram_value=-0.4))
        assert len(h._wave_candles) == 1

    def test_same_side_values_accumulate(self) -> None:
        h = MarketStructureHelper()
        h.register_candle(_candle(open_time=1_000, histogram_value=0.5))
        h.register_candle(_candle(open_time=2_000, histogram_value=0.6))
        h.register_candle(_candle(open_time=3_000, histogram_value=0.3))
        assert len(h._wave_candles) == 3

    def test_up_to_down_flip_clears_buffer_to_current_only(self) -> None:
        h = MarketStructureHelper()
        h.register_candle(_candle(open_time=1_000, histogram_value=0.5))
        h.register_candle(_candle(open_time=2_000, histogram_value=0.3))
        # Flip: prior > 0, current < 0.
        h.register_candle(_candle(open_time=3_000, histogram_value=-0.2))

        assert len(h._wave_candles) == 1
        assert h._wave_candles[0].open_time == 3_000

    def test_down_to_up_flip_clears_buffer_to_current_only(self) -> None:
        h = MarketStructureHelper()
        h.register_candle(_candle(open_time=1_000, histogram_value=-0.4))
        h.register_candle(_candle(open_time=2_000, histogram_value=-0.1))
        h.register_candle(_candle(open_time=3_000, histogram_value=0.2))

        assert len(h._wave_candles) == 1
        assert h._wave_candles[0].open_time == 3_000

    def test_zero_is_up_side(self) -> None:
        """Exact zero is classified as 'up', so 0.0 → -0.1 is a flip.

        This is the behavior locked in by ``_sign_flipped``: ``prev >= 0``
        evaluates True at exactly zero. Guards against a subtle off-by-one
        that the TS original already gets right.
        """
        h = MarketStructureHelper()
        h.register_candle(_candle(open_time=1_000, histogram_value=0.0))
        h.register_candle(_candle(open_time=2_000, histogram_value=-0.1))

        assert len(h._wave_candles) == 1
        assert h._wave_candles[0].open_time == 2_000

    def test_zero_to_positive_is_not_a_flip(self) -> None:
        """0.0 and 0.5 are both 'up' side, so the buffer accumulates."""
        h = MarketStructureHelper()
        h.register_candle(_candle(open_time=1_000, histogram_value=0.0))
        h.register_candle(_candle(open_time=2_000, histogram_value=0.5))

        assert len(h._wave_candles) == 2

    def test_multi_flip_sequence_stays_consistent(self) -> None:
        """Walk through up → down → up → down; buffer always contains
        only the candles since the most recent flip."""
        h = MarketStructureHelper()
        h.register_candle(_candle(open_time=1_000, histogram_value=0.4))
        h.register_candle(_candle(open_time=2_000, histogram_value=0.2))
        h.register_candle(_candle(open_time=3_000, histogram_value=-0.1))  # flip
        h.register_candle(_candle(open_time=4_000, histogram_value=-0.3))
        h.register_candle(_candle(open_time=5_000, histogram_value=0.2))  # flip
        h.register_candle(_candle(open_time=6_000, histogram_value=0.5))
        h.register_candle(_candle(open_time=7_000, histogram_value=-0.4))  # flip

        assert len(h._wave_candles) == 1
        assert h._wave_candles[0].open_time == 7_000
        assert h.total_candles_registered == 7


@pytest.mark.parametrize(
    ("prev", "curr", "expected"),
    [
        (0.5, 0.3, False),  # both up, no flip
        (-0.5, -0.3, False),  # both down, no flip
        (0.5, -0.3, True),  # up -> down
        (-0.5, 0.3, True),  # down -> up
        (0.0, -0.1, True),  # zero (up) -> negative = flip
        (0.0, 0.1, False),  # zero (up) -> positive = not a flip
        (-0.1, 0.0, True),  # negative -> zero (up) = flip
    ],
)
def test_sign_flipped_static_method(prev: float, curr: float, expected: bool) -> None:
    """Direct coverage of the ``_sign_flipped`` staticmethod.

    A staticmethod can be called on either the class or an instance —
    calling on the class here documents that it does not touch ``self``.
    """
    assert MarketStructureHelper._sign_flipped(prev, curr) is expected
