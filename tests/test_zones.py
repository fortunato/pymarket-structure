"""Stage 11 tests for zone detection, long-term swing pickers, and helpers.

Spot-checks against the TS test suite using the LTC/USDT 4h fixture.
Each assertion maps to a named TS spec test case with the same date range.

``reportPrivateUsage`` is disabled because zone cache tests inspect
``_zone_cache_support`` to verify invalidation behaviour.
"""

# pyright: reportPrivateUsage=false

import json
from pathlib import Path

import pandas as pd
import pytest

from market_structure import MarketStructureHelper
from market_structure.types import Candle

# ---------------------------------------------------------------------------
# Fixture loading (mirrors test_trend.py / test_parity.py helpers)
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
# Synthetic candle factory
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
# range_overlaps
# ---------------------------------------------------------------------------


class TestRangeOverlaps:
    """Parametrized tests for the static ``range_overlaps`` utility."""

    @pytest.mark.parametrize(
        ("a", "b", "expected"),
        [
            # a[0] falls within b
            ((5.0, 10.0), (3.0, 7.0), True),
            # a[1] falls within b
            ((1.0, 5.0), (3.0, 7.0), True),
            # a completely contains b
            ((1.0, 10.0), (3.0, 7.0), True),
            # b completely contains a
            ((4.0, 6.0), (3.0, 7.0), True),
            # no overlap — a above b
            ((8.0, 10.0), (3.0, 7.0), False),
            # no overlap — a below b
            ((1.0, 2.0), (3.0, 7.0), False),
            # touching at a single point
            ((7.0, 10.0), (3.0, 7.0), True),
            # identical ranges
            ((3.0, 7.0), (3.0, 7.0), True),
        ],
    )
    def test_range_overlaps(
        self,
        a: tuple[float, float],
        b: tuple[float, float],
        expected: bool,
    ) -> None:
        assert MarketStructureHelper.range_overlaps(a, b) is expected


# ---------------------------------------------------------------------------
# get_bottom_range / get_top_range — fixture-based
# ---------------------------------------------------------------------------


class TestBottomRange:
    """TS: 'Determines zone from top/bottom candle wicks' (bottoms)."""

    def test_based_purely_on_lowest_close_or_open(self) -> None:
        h = _make_helper("2020-07-01 00:00:00", "2020-07-27 20:00:00")
        last_bottom = h.get_last_bottom()
        assert last_bottom is not None
        r = h.get_bottom_range(last_bottom)
        assert r == pytest.approx((46.85, 47.61))

    def test_based_on_lowest_close_or_open_plus_lowest_low(self) -> None:
        h = _make_helper("2020-07-01 00:00:00", "2020-07-25 04:00:00")
        last_bottom = h.get_last_bottom()
        assert last_bottom is not None
        r = h.get_bottom_range(last_bottom)
        assert r == pytest.approx((43.93, 44.11))


class TestTopRange:
    """TS: 'Determines zone from top/bottom candle wicks' (tops)."""

    def test_based_purely_on_highest_close_or_open(self) -> None:
        h = _make_helper("2020-07-01 00:00:00", "2020-08-02 08:00:00")
        last_top = h.get_last_top()
        assert last_top is not None
        r = h.get_top_range(last_top)
        assert r == pytest.approx((64.91, 65.17))

    def test_based_on_highest_close_or_open_plus_highest_high(self) -> None:
        h = _make_helper("2020-07-01 00:00:00", "2020-07-29 08:00:00")
        last_top = h.get_last_top()
        assert last_top is not None
        r = h.get_top_range(last_top)
        assert r == pytest.approx((56.88, 57.92))


# ---------------------------------------------------------------------------
# get_support_zones — fixture-based
# ---------------------------------------------------------------------------


class TestSupportZones:
    """TS: 'Determines significant levels — Based on bottoms'."""

    def test_picks_double_bottom_in_the_past(self) -> None:
        h = _make_helper("2020-07-01 00:00:00", "2020-08-05 12:00:00")
        zones = h.get_support_zones(include_forming_wave=False)
        z0 = zones[0]
        assert z0.range == pytest.approx((51.79, 57.31))
        assert z0.is_double is True
        assert len(z0.overlapping_low_wave_ids) == 3
        assert len(z0.overlapping_high_wave_ids) == 1

    def test_picks_double_bottom_in_slightly_more_distant_past(self) -> None:
        h = _make_helper("2020-07-01 00:00:00", "2020-08-07 12:00:00")
        zones = h.get_support_zones(include_forming_wave=False)
        z1 = zones[1]
        assert z1.range == pytest.approx((51.79, 57.31))
        assert z1.is_double is True
        assert len(z1.overlapping_low_wave_ids) == 3
        assert len(z1.overlapping_high_wave_ids) == 1

    def test_picks_double_bottom_with_current_wave(self) -> None:
        h = _make_helper("2020-07-01 00:00:00", "2020-08-04 16:00:00")
        zones = h.get_support_zones(include_forming_wave=True)
        z0 = zones[0]
        assert z0.range == pytest.approx((51.79, 57.31))
        assert z0.is_double is True
        assert len(z0.overlapping_low_wave_ids) == 3
        assert len(z0.overlapping_high_wave_ids) == 1

    def test_can_pick_double_bottom_over_longer_period(self) -> None:
        h = _make_helper("2020-06-25 00:00:00", "2020-07-17 04:00:00")
        zones = h.get_support_zones(include_forming_wave=False, double_bottom_proximity=6)
        z0 = zones[0]
        assert z0.range == pytest.approx((40.75, 41.72))
        assert z0.is_double is True
        assert len(z0.overlapping_low_wave_ids) == 5
        assert len(z0.overlapping_high_wave_ids) == 1

    def test_double_bottom_with_lower_bottom_between_is_discarded(self) -> None:
        h = _make_helper("2020-06-29 00:00:00", "2020-07-04 04:00:00")
        zones = h.get_support_zones()
        z0 = zones[0]
        assert z0.range == pytest.approx((40.96, 41.14))
        assert z0.is_double is False
        assert len(z0.overlapping_low_wave_ids) == 1
        assert len(z0.overlapping_high_wave_ids) == 0


# ---------------------------------------------------------------------------
# get_resistance_zones — fixture-based
# ---------------------------------------------------------------------------


class TestResistanceZones:
    """TS: 'Determines significant levels — Based on tops'."""

    def test_picks_double_top_in_the_past(self) -> None:
        h = _make_helper("2020-07-25 00:00:00", "2020-07-31 04:00:00")
        zones = h.get_resistance_zones()
        z0 = zones[0]
        assert z0.range == pytest.approx((57.57, 58.61))
        assert z0.is_double is True
        assert len(z0.overlapping_low_wave_ids) == 0
        assert len(z0.overlapping_high_wave_ids) == 1

    def test_picks_double_top_in_slightly_more_distant_past(self) -> None:
        h = _make_helper("2020-07-25 00:00:00", "2020-08-05 00:00:00")
        zones = h.get_resistance_zones()
        z2 = zones[2]
        assert z2.range == pytest.approx((57.57, 58.61))
        assert z2.is_double is True
        assert len(z2.overlapping_low_wave_ids) == 0
        assert len(z2.overlapping_high_wave_ids) == 1

    def test_picks_double_top_with_current_wave(self) -> None:
        h = _make_helper("2020-07-25 00:00:00", "2020-07-30 20:00:00")
        zones = h.get_resistance_zones(include_forming_wave=True)
        z0 = zones[0]
        assert z0.range == pytest.approx((57.57, 58.61))
        assert z0.is_double is True
        assert len(z0.overlapping_low_wave_ids) == 0
        assert len(z0.overlapping_high_wave_ids) == 1

    def test_can_pick_double_top_over_longer_period(self) -> None:
        h = _make_helper("2020-07-12 00:00:00", "2020-07-22 08:00:00")
        zones = h.get_resistance_zones(include_forming_wave=False, double_top_proximity=3)
        z0 = zones[0]
        assert z0.range == pytest.approx((43.91, 44.11))
        assert z0.is_double is True
        assert len(z0.overlapping_low_wave_ids) == 1
        assert len(z0.overlapping_high_wave_ids) == 1

    def test_double_top_with_higher_top_between_is_discarded(self) -> None:
        h = _make_helper("2020-06-29 00:00:00", "2020-07-15 04:00:00")
        zones = h.get_resistance_zones(include_forming_wave=False, double_top_proximity=10)
        z0 = zones[0]
        assert z0.range == pytest.approx((43.88, 44.09))
        assert z0.is_double is False
        assert len(z0.overlapping_low_wave_ids) == 3
        assert len(z0.overlapping_high_wave_ids) == 1


# ---------------------------------------------------------------------------
# pick_long_term_top — fixture-based
# ---------------------------------------------------------------------------


class TestPickLongTermTop:
    """TS: 'Picks a long term top if it meets given criteria'."""

    def test_found_on_2020_07_24(self) -> None:
        """On 2020-07-24, a long term high occurred @ 2020-07-23 looking back 60 bars."""
        h = _make_helper("2020-07-01 00:00:00", "2020-07-24 16:00:00")
        result = h.pick_long_term_top(high_since=60, max_age=10)
        assert result is not None
        assert result.wave.highest_close_or_open.open_time == int(
            pd.Timestamp("2020-07-23T20:00:00Z").value // 10**6
        )
        assert result.age == 4

    def test_not_found_when_exceeds_max_age(self) -> None:
        """On 2020-07-24, a long term high occurred > maxAge looking back 100 bars."""
        h = _make_helper("2020-07-01 00:00:00", "2020-07-24 16:00:00")
        result = h.pick_long_term_top(high_since=100, max_age=50)
        assert result is None

    def test_found_on_2020_08_07(self) -> None:
        """On 2020-08-07, a long term high occurred @ 2020-08-02 looking back 150 bars."""
        h = _make_helper("2020-06-28 04:00:00", "2020-08-07 16:00:00")
        result = h.pick_long_term_top(high_since=200, max_age=100)
        assert result is not None
        assert result.wave.highest_close_or_open.open_time == int(
            pd.Timestamp("2020-08-02T00:00:00Z").value // 10**6
        )
        assert result.age == 33


# ---------------------------------------------------------------------------
# pick_long_term_bottom — fixture-based
# ---------------------------------------------------------------------------


class TestPickLongTermBottom:
    """TS: 'Picks a long term bottom if it meets given criteria'."""

    def test_found_on_2020_08_17(self) -> None:
        """On 2020-08-17, a long term low occurred @ 2020-08-12 looking back 91 bars."""
        h = _make_helper("2020-07-01 00:00:00", "2020-08-17 00:00:00")
        result = h.pick_long_term_bottom(low_since=91, max_age=50)
        assert result is not None
        assert result.wave.lowest_close_or_open.open_time == int(
            pd.Timestamp("2020-08-12T00:00:00Z").value // 10**6
        )
        assert result.age == 29

    def test_not_found_at_92_bars(self) -> None:
        """On 2020-08-17, a long term low DID NOT occur looking back 92 bars."""
        h = _make_helper("2020-07-01 00:00:00", "2020-08-17 00:00:00")
        result = h.pick_long_term_bottom(low_since=92, max_age=50)
        assert result is None


# ---------------------------------------------------------------------------
# Zone cache
# ---------------------------------------------------------------------------


class TestZoneCache:
    """Zone results are cached and invalidated on wave push."""

    def test_cached_result_returned_on_second_call(self) -> None:
        h = _make_helper("2020-07-01 00:00:00", "2020-08-05 12:00:00")
        z1 = h.get_support_zones()
        z2 = h.get_support_zones()
        assert z1 is z2  # same list object — cached

    def test_cache_invalidated_on_new_wave(self) -> None:
        h = _make_helper("2020-07-01 00:00:00", "2020-08-05 12:00:00")
        _ = h.get_support_zones()
        assert h._zone_cache_support is not None
        # Register a candle that flips → pushes a new wave → invalidates.
        # We need to trigger a flip, so pass a candle with opposite histogram sign.
        current = h.get_current_wave()
        assert current is not None
        new_sign = 1.0 if current.side == "down" else -1.0
        h.register_candle(
            _candle(open_time=99_999_000, histogram_value=new_sign),
        )
        assert h._zone_cache_support is None


# ---------------------------------------------------------------------------
# get_wave_by_id
# ---------------------------------------------------------------------------


class TestGetWaveById:
    def test_returns_wave_for_known_id(self) -> None:
        h = _make_helper("2020-07-01 00:00:00", "2020-07-10 00:00:00")
        assert len(h.wave_registry) > 0
        first = h.wave_registry[0]
        assert h.get_wave_by_id(first.id) is first

    def test_returns_none_for_unknown_id(self) -> None:
        h = _make_helper("2020-07-01 00:00:00", "2020-07-10 00:00:00")
        assert h.get_wave_by_id("nonexistent") is None
