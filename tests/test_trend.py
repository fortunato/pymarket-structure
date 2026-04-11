"""Stage 10 — Trend state, comparisons, and divergence.

Spot-checks against the TS test suite using the LTC/USDT 4h fixture.
Each assertion maps to a named TS spec test case with the same date range.
"""

import json
from pathlib import Path

import pandas as pd
import pytest

from market_structure import MarketStructureHelper
from market_structure.types import Candle

# ---------------------------------------------------------------------------
# Fixture loading (mirrors test_parity.py helpers)
# ---------------------------------------------------------------------------

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "ms-LTCUSDT-4h.json"
HISTOGRAM_KEY = "tsi_histogram"


def _load_raw() -> list[dict[str, object]]:
    with FIXTURE_PATH.open() as f:
        return json.load(f)


def _filter_range(
    raw: list[dict[str, object]], from_str: str, to_str: str
) -> list[dict[str, object]]:
    from_ts = pd.Timestamp(from_str, tz="UTC")
    to_ts = pd.Timestamp(to_str, tz="UTC")
    return [
        row
        for row in raw
        if pd.Timestamp(str(row["openTime"])) >= from_ts
        and pd.Timestamp(str(row["closeTime"])) < to_ts
    ]


def _make_candle(row: dict[str, object]) -> Candle:
    return Candle(
        open_time=int(pd.Timestamp(str(row["openTime"]), tz="UTC").value // 10**6),
        open=float(row["open"]),  # type: ignore[arg-type]
        high=float(row["high"]),  # type: ignore[arg-type]
        low=float(row["low"]),  # type: ignore[arg-type]
        close=float(row["close"]),  # type: ignore[arg-type]
        volume=float(row["volume"]),  # type: ignore[arg-type]
        histogram_value=float(row[HISTOGRAM_KEY]),  # type: ignore[arg-type]
    )


def _make_helper(from_str: str, to_str: str) -> MarketStructureHelper:
    raw = _load_raw()
    rows = _filter_range(raw, from_str, to_str)
    h = MarketStructureHelper(histogram_key=HISTOGRAM_KEY)
    for row in rows:
        h.register_candle(_make_candle(row))
    return h


# ---------------------------------------------------------------------------
# Trend up — TS: "Determines confirmed up-trend based on tops and bottoms"
# ---------------------------------------------------------------------------


class TestTrendUp:
    def test_uptrend_confirmed(self) -> None:
        """TS: 'At 2020-07-23 00:00:00 an up-trend is confirmed'"""
        h = _make_helper("2020-07-01 00:00:00", "2020-07-23 00:00:00")
        assert h.is_trending_up() is True

    def test_uptrend_not_yet_confirmed(self) -> None:
        """TS: 'At 2020-07-22 20:00:00 an up-trend is NOT YET confirmed'"""
        h = _make_helper("2020-07-01 00:00:00", "2020-07-22 20:00:00")
        assert h.is_trending_up() is False

    def test_uptrend_still_going(self) -> None:
        """TS: 'At 2020-07-25 04:00:00 an up-trend is still going'"""
        h = _make_helper("2020-07-01 00:00:00", "2020-07-25 04:00:00")
        assert h.is_trending_up() is True

    def test_uptrend_broken(self) -> None:
        """TS: 'Absence of trend after market structure was broken'"""
        h = _make_helper("2020-07-01 00:00:00", "2020-08-02 08:00:00")
        assert h.is_trending_up() is False


# ---------------------------------------------------------------------------
# Trend down — TS: "Determines confirmed down-trend based on tops and bottoms"
# ---------------------------------------------------------------------------


class TestTrendDown:
    def test_downtrend_confirmed(self) -> None:
        """TS: 'At 2020-07-15 04:00:00 a down-trend is confirmed'"""
        h = _make_helper("2020-07-01 00:00:00", "2020-07-15 04:00:00")
        assert h.is_trending_down() is True

    def test_downtrend_not_yet_confirmed(self) -> None:
        """TS: 'At 2020-07-15 00:00:00 a down-trend is NOT YET confirmed'"""
        h = _make_helper("2020-07-01 00:00:00", "2020-07-15 00:00:00")
        assert h.is_trending_down() is False

    def test_downtrend_still_going(self) -> None:
        """TS: 'At 2020-07-17 00:00:00 a down-trend is still going'"""
        h = _make_helper("2020-07-01 00:00:00", "2020-07-17 00:00:00")
        assert h.is_trending_down() is True

    def test_downtrend_broken(self) -> None:
        """TS: 'Absence of down-trend after market structure was broken'"""
        h = _make_helper("2020-07-01 00:00:00", "2020-07-19 20:00:00")
        assert h.is_trending_down() is False


# ---------------------------------------------------------------------------
# Divergence — TS: "Determines divergence based on highest/lowest close and histogram"
# ---------------------------------------------------------------------------


class TestDivergence:
    def test_tops_diverging(self) -> None:
        """TS: 'Tops with divergence'"""
        h = _make_helper("2020-06-28 04:00:00", "2020-07-09 12:00:00")
        last_top = h.get_last_top()
        prev_top = h.get_previous_top()
        assert last_top is not None
        assert prev_top is not None
        assert h.is_diverging(last_top, prev_top) is True

    def test_tops_not_diverging(self) -> None:
        """TS: 'Tops without divergence'"""
        h = _make_helper("2020-06-28 04:00:00", "2020-07-07 16:00:00")
        last_top = h.get_last_top()
        prev_top = h.get_previous_top()
        assert last_top is not None
        assert prev_top is not None
        assert h.is_diverging(last_top, prev_top) is False

    def test_bottoms_diverging(self) -> None:
        """TS: 'Bottoms with divergence'"""
        h = _make_helper("2020-08-01 00:00:00", "2020-08-10 12:00:00")
        last_bottom = h.get_last_bottom()
        prev_bottom = h.get_previous_bottom()
        assert last_bottom is not None
        assert prev_bottom is not None
        assert h.is_diverging(last_bottom, prev_bottom) is True

    def test_bottoms_not_diverging(self) -> None:
        """TS: 'Bottoms without divergence'"""
        h = _make_helper("2020-08-01 00:00:00", "2020-08-08 20:00:00")
        last_bottom = h.get_last_bottom()
        prev_bottom = h.get_previous_bottom()
        assert last_bottom is not None
        assert prev_bottom is not None
        assert h.is_diverging(last_bottom, prev_bottom) is False


# ---------------------------------------------------------------------------
# Direct comparison method tests
# ---------------------------------------------------------------------------


class TestComparisons:
    """Direct tests for made_higher_high / made_higher_low / made_lower_*."""

    @pytest.fixture()
    def h(self) -> MarketStructureHelper:
        return _make_helper("2020-07-01 00:00:00", "2020-07-23 00:00:00")

    def test_made_higher_high_in_uptrend(self, h: MarketStructureHelper) -> None:
        last_top = h.get_last_top()
        prev_top = h.get_previous_top()
        assert last_top is not None and prev_top is not None
        assert h.made_higher_high(last_top, prev_top) is True

    def test_made_higher_low_in_uptrend(self, h: MarketStructureHelper) -> None:
        last_bottom = h.get_last_bottom()
        prev_bottom = h.get_previous_bottom()
        assert last_bottom is not None and prev_bottom is not None
        assert h.made_higher_low(last_bottom, prev_bottom) is True

    def test_made_lower_high_in_downtrend(self) -> None:
        h = _make_helper("2020-07-01 00:00:00", "2020-07-15 04:00:00")
        last_top = h.get_last_top()
        prev_top = h.get_previous_top()
        assert last_top is not None and prev_top is not None
        assert h.made_lower_high(last_top, prev_top) is True

    def test_made_lower_low_in_downtrend(self) -> None:
        h = _make_helper("2020-07-01 00:00:00", "2020-07-15 04:00:00")
        last_bottom = h.get_last_bottom()
        prev_bottom = h.get_previous_bottom()
        assert last_bottom is not None and prev_bottom is not None
        assert h.made_lower_low(last_bottom, prev_bottom) is True


# ---------------------------------------------------------------------------
# Between scans
# ---------------------------------------------------------------------------


class TestBetweenScans:
    """Tests for made_lower_low_between / made_higher_high_between.

    These scan ALL registry entries between two waves (not just same-side),
    so an up-wave sitting between two down-waves can trigger a "lower low"
    if its low undercuts either endpoint.
    """

    def test_between_scans_same_wave_returns_false(self) -> None:
        """Scanning between a wave and itself finds nothing."""
        h = _make_helper("2020-07-01 00:00:00", "2020-07-23 00:00:00")
        last_top = h.get_last_top()
        assert last_top is not None
        assert h.made_higher_high_between(last_top, last_top) is False
        assert h.made_lower_low_between(last_top, last_top) is False

    def test_returns_false_for_unknown_wave(self) -> None:
        """Waves not in the registry return False gracefully."""
        h = _make_helper("2020-07-01 00:00:00", "2020-07-23 00:00:00")
        last_top = h.get_last_top()
        assert last_top is not None
        fake = Candle(open_time=0, open=1, high=2, low=0.5, close=1.5, volume=10)
        from market_structure.types import Wave

        orphan = Wave(
            id="orphan",
            side="up",
            formation_bar_index=0,
            high=fake,
            low=fake,
            highest_close=fake,
            lowest_close=fake,
            highest_close_or_open=fake,
            lowest_close_or_open=fake,
            high_idx=0,
            low_idx=0,
            highest_close_or_open_idx=0,
            lowest_close_or_open_idx=0,
        )
        assert h.made_higher_high_between(last_top, orphan) is False
        assert h.made_lower_low_between(last_top, orphan) is False


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestTrendEdgeCases:
    def test_trending_up_returns_false_when_no_waves(self) -> None:
        h = MarketStructureHelper()
        assert h.is_trending_up() is False

    def test_trending_down_returns_false_when_no_waves(self) -> None:
        h = MarketStructureHelper()
        assert h.is_trending_down() is False

    def test_trending_returns_false_with_insufficient_waves(self) -> None:
        """Need at least 4 confirmed waves (2 tops + 2 bottoms)."""
        h = _make_helper("2020-07-01 00:00:00", "2020-07-10 00:00:00")
        # With few candles we may not have enough waves for trend detection
        if len(h.wave_registry) < 4:
            assert h.is_trending_up() is False
            assert h.is_trending_down() is False
