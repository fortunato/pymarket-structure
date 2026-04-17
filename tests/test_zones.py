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
from tests.double_pattern_builders import (
    build_adjacent_wicks,
    build_exact_tie_bottoms,
    build_exact_tie_tops,
    build_first_swing_only,
    build_m_pattern_with_intermediate,
    build_nan_atr_bottoms,
    build_nearest_not_deepest,
    build_negative_atr_bottoms,
    build_regime_shift_atr,
    build_short_atr_array,
    build_tight_disjoint_bottoms,
    build_tight_disjoint_tops,
    build_w_pattern_with_intermediate,
    build_wide_far_bottoms,
    build_wide_far_tops,
    build_zero_atr_bottoms,
    build_zero_atr_tops,
)

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
        """Under the new tolerance predicate, the w-34 / w-36 pair no
        longer qualifies as a double (lows 4.47 apart ≫ 0.004 x 56.26
        pct-fallback tolerance of ~0.225). The old expectation
        ``range=(51.79, 57.31) is_double=True`` was the wick-overlap
        FP this feature intentionally corrects.
        """
        h = _make_helper("2020-07-01 00:00:00", "2020-08-05 12:00:00")
        zones = h.get_support_zones(include_forming_wave=False)
        z0 = zones[0]
        assert z0.range == pytest.approx((56.26, 57.31))
        assert z0.is_double is False
        assert len(z0.overlapping_low_wave_ids) == 1
        assert len(z0.overlapping_high_wave_ids) == 1

    def test_picks_double_bottom_in_slightly_more_distant_past(self) -> None:
        """See :meth:`test_picks_double_bottom_in_the_past` — same pair,
        different window; new predicate rejects it uniformly.
        """
        h = _make_helper("2020-07-01 00:00:00", "2020-08-07 12:00:00")
        zones = h.get_support_zones(include_forming_wave=False)
        z1 = zones[1]
        assert z1.range == pytest.approx((56.26, 57.31))
        assert z1.is_double is False
        assert len(z1.overlapping_low_wave_ids) == 1
        assert len(z1.overlapping_high_wave_ids) == 1

    def test_picks_double_bottom_with_current_wave(self) -> None:
        """See :meth:`test_picks_double_bottom_in_the_past` — same pair,
        with the forming wave included; new predicate rejects it.
        """
        h = _make_helper("2020-07-01 00:00:00", "2020-08-04 16:00:00")
        zones = h.get_support_zones(include_forming_wave=True)
        z0 = zones[0]
        assert z0.range == pytest.approx((56.26, 57.31))
        assert z0.is_double is False
        assert len(z0.overlapping_low_wave_ids) == 1
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
        """Under the new tolerance predicate, the w-2 / w-4 pair no
        longer qualifies as a double-top (highs too far apart for the
        0.004 x high pct-fallback tolerance). The old expectation
        ``is_double=True`` was the wick-overlap FP this feature
        intentionally corrects.
        """
        h = _make_helper("2020-07-25 00:00:00", "2020-07-31 04:00:00")
        zones = h.get_resistance_zones()
        z0 = zones[0]
        assert z0.range == pytest.approx((57.57, 58.61))
        assert z0.is_double is False
        assert len(z0.overlapping_low_wave_ids) == 0
        assert len(z0.overlapping_high_wave_ids) == 1

    def test_picks_double_top_in_slightly_more_distant_past(self) -> None:
        """See :meth:`test_picks_double_top_in_the_past`."""
        h = _make_helper("2020-07-25 00:00:00", "2020-08-05 00:00:00")
        zones = h.get_resistance_zones()
        z2 = zones[2]
        assert z2.range == pytest.approx((57.57, 58.61))
        assert z2.is_double is False
        assert len(z2.overlapping_low_wave_ids) == 0
        assert len(z2.overlapping_high_wave_ids) == 1

    def test_picks_double_top_with_current_wave(self) -> None:
        """See :meth:`test_picks_double_top_in_the_past`."""
        h = _make_helper("2020-07-25 00:00:00", "2020-07-30 20:00:00")
        zones = h.get_resistance_zones(include_forming_wave=True)
        z0 = zones[0]
        assert z0.range == pytest.approx((57.57, 58.61))
        assert z0.is_double is False
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


# ---------------------------------------------------------------------------
# ATR-driven double-bottom tolerance predicate
# ---------------------------------------------------------------------------


class TestDoubleBottomTolerance:
    """ATR-driven price-proximity tolerance for double bottoms.

    Each fixture is a synthetic wave registry constructed in
    ``tests/double_pattern_builders.py``. The newest down wave (``w-2``)
    is the anchor; the older down wave (``w-0``) is the candidate pair.
    The tolerance predicate replaces the old wick-overlap test.
    """

    def test_double_bottom_tight_disjoint_qualifies(self) -> None:
        """T012 — lows 0.15 apart, wick ranges disjoint: the NEW predicate
        qualifies this pair as a double bottom even though wicks don't overlap.
        """
        h, atr_arr = build_tight_disjoint_bottoms()
        zones = h.get_support_zones(atr_arr=atr_arr)
        assert zones[0].anchor_wave_id == "w-2"
        assert zones[0].is_double is True

    def test_double_bottom_wide_far_rejects(self) -> None:
        """T013 — lows 2.0 apart with overlapping wicks: the NEW predicate
        REJECTS this pair because the price distance exceeds tolerance, even
        though the wick ranges overlap.
        """
        h, atr_arr = build_wide_far_bottoms()
        zones = h.get_support_zones(atr_arr=atr_arr)
        assert zones[0].anchor_wave_id == "w-2"
        assert zones[0].is_double is False

    def test_double_bottom_exact_tie_qualifies(self) -> None:
        """T014 — lows at identical prices. Tolerance is inclusive at zero
        distance, so the pair qualifies.
        """
        h, atr_arr = build_exact_tie_bottoms()
        zones = h.get_support_zones(atr_arr=atr_arr)
        assert zones[0].anchor_wave_id == "w-2"
        assert zones[0].is_double is True

    def test_double_bottom_zero_atr_fallback(self) -> None:
        """T015 — ATR array is zero everywhere. Tolerance falls through to
        ``tolerance_pct_fallback`` and the near-equal lows still qualify.
        """
        h, atr_arr = build_zero_atr_bottoms()
        zones = h.get_support_zones(atr_arr=atr_arr)
        assert zones[0].anchor_wave_id == "w-2"
        assert zones[0].is_double is True

    def test_double_bottom_percentage_override(self) -> None:
        """T016 — with ``atr_arr=None``, the caller's ``tolerance_pct_fallback``
        is used. A strict override (0.001 = 0.1 %) rejects the 0.15 gap;
        a loose override (0.01 = 1 %) accepts it.
        """
        h, _ = build_tight_disjoint_bottoms()
        # Strict percentage: 0.001 * 100.15 ≈ 0.1002 < 0.15 gap → rejected
        strict = h.get_support_zones(atr_arr=None, tolerance_pct_fallback=0.001)
        anchor_strict = next(z for z in strict if z.anchor_wave_id == "w-2")
        assert anchor_strict.is_double is False

        # Invalidate cache before second call (different kwarg value).
        h2, _ = build_tight_disjoint_bottoms()
        loose = h2.get_support_zones(atr_arr=None, tolerance_pct_fallback=0.01)
        anchor_loose = next(z for z in loose if z.anchor_wave_id == "w-2")
        assert anchor_loose.is_double is True

    def test_double_bottom_tolerance_boundary_inclusive(self) -> None:
        """T017 — lows exactly at the tolerance boundary. Inclusive ``<=``
        check means the pair qualifies when gap == tolerance.
        """
        h, atr_arr = build_tight_disjoint_bottoms()
        # Use the actual FP gap between the two lows so tolerance == gap
        # exactly (``100.15 - 100.0`` is not precisely 0.15 in double-precision
        # FP — setting multiple to the computed gap sidesteps that).
        gap = abs(100.15 - 100.0)
        zones = h.get_support_zones(atr_arr=atr_arr, tolerance_atr_multiple=gap)
        anchor = next(z for z in zones if z.anchor_wave_id == "w-2")
        assert anchor.is_double is True

    def test_zone_geometry_not_bridged_for_disjoint_wicks(self) -> None:
        """T018 — FR-013: a qualified pair with DISJOINT wicks must not
        extend the zone to bridge the gap between the two waves. The zone
        range stays at the anchor's own wick range (100.15, 102.00).
        """
        h, atr_arr = build_tight_disjoint_bottoms()
        zones = h.get_support_zones(atr_arr=atr_arr)
        anchor = next(z for z in zones if z.anchor_wave_id == "w-2")
        assert anchor.is_double is True
        assert anchor.range == pytest.approx((100.15, 102.00))


# ---------------------------------------------------------------------------
# ATR-driven double-top tolerance predicate (resistance mirror)
# ---------------------------------------------------------------------------


class TestDoubleTopTolerance:
    """Mirror of ``TestDoubleBottomTolerance`` for the resistance path."""

    def test_double_top_tight_disjoint_qualifies(self) -> None:
        h, atr_arr = build_tight_disjoint_tops()
        zones = h.get_resistance_zones(atr_arr=atr_arr)
        anchor = next(z for z in zones if z.anchor_wave_id == "w-2")
        assert anchor.is_double is True

    def test_double_top_wide_far_rejects(self) -> None:
        h, atr_arr = build_wide_far_tops()
        zones = h.get_resistance_zones(atr_arr=atr_arr)
        anchor = next(z for z in zones if z.anchor_wave_id == "w-2")
        assert anchor.is_double is False

    def test_double_top_exact_tie_qualifies(self) -> None:
        h, atr_arr = build_exact_tie_tops()
        zones = h.get_resistance_zones(atr_arr=atr_arr)
        anchor = next(z for z in zones if z.anchor_wave_id == "w-2")
        assert anchor.is_double is True

    def test_double_top_zero_atr_fallback(self) -> None:
        h, atr_arr = build_zero_atr_tops()
        zones = h.get_resistance_zones(atr_arr=atr_arr)
        anchor = next(z for z in zones if z.anchor_wave_id == "w-2")
        assert anchor.is_double is True

    def test_double_top_percentage_override(self) -> None:
        h, _ = build_tight_disjoint_tops()
        strict = h.get_resistance_zones(atr_arr=None, tolerance_pct_fallback=0.001)
        anchor_strict = next(z for z in strict if z.anchor_wave_id == "w-2")
        assert anchor_strict.is_double is False

        h2, _ = build_tight_disjoint_tops()
        loose = h2.get_resistance_zones(atr_arr=None, tolerance_pct_fallback=0.01)
        anchor_loose = next(z for z in loose if z.anchor_wave_id == "w-2")
        assert anchor_loose.is_double is True

    def test_double_top_tolerance_boundary_inclusive(self) -> None:
        h, atr_arr = build_tight_disjoint_tops()
        # See bottom-side twin for the FP rationale.
        gap = abs(110.0 - 109.85)
        zones = h.get_resistance_zones(atr_arr=atr_arr, tolerance_atr_multiple=gap)
        anchor = next(z for z in zones if z.anchor_wave_id == "w-2")
        assert anchor.is_double is True

    def test_zone_geometry_not_bridged_for_disjoint_wicks_top(self) -> None:
        """Mirror of T018: disjoint wicks → zone stays at anchor's own
        ``get_top_range`` ``(108.00, 109.85)``.
        """
        h, atr_arr = build_tight_disjoint_tops()
        zones = h.get_resistance_zones(atr_arr=atr_arr)
        anchor = next(z for z in zones if z.anchor_wave_id == "w-2")
        assert anchor.is_double is True
        assert anchor.range == pytest.approx((108.00, 109.85))


# ---------------------------------------------------------------------------
# Double-pattern edge cases (first swing, short ATR, regime shift, etc.)
# ---------------------------------------------------------------------------


class TestDoublePatternEdgeCases:
    """Edge-case coverage for the new tolerance predicate."""

    def test_first_swing_no_double_label(self) -> None:
        """T019a — anchor is the only same-side wave in the registry. The
        double-pattern loop body has no preceding wave to pair with, must
        not raise, and must yield ``is_double=False``.
        """
        h, atr_arr = build_first_swing_only()
        zones = h.get_support_zones(atr_arr=atr_arr)
        assert len(zones) == 1
        assert zones[0].is_double is False
        assert zones[0].overlapping_low_wave_ids == ()

    def test_short_atr_array_falls_back_gracefully(self) -> None:
        """T019b — ``atr_arr`` shorter than the DataFrame. Out-of-bounds
        indices fall through to the percentage fallback without raising.
        """
        h, atr_arr = build_short_atr_array()
        assert len(atr_arr) < len(h.wave_registry[-1].candles) + 5  # sanity
        zones = h.get_support_zones(atr_arr=atr_arr)
        anchor = next(z for z in zones if z.anchor_wave_id == "w-2")
        # anchor.low_idx=3 is OOB in a length-1 atr_arr → pct fallback path.
        # Percentage fallback: 0.004 x 100.15 ≈ 0.4006 > 0.15 gap → qualifies.
        assert anchor.is_double is True

    def test_regime_shift_uses_anchor_low_idx_atr(self) -> None:
        """T019c — ATR at anchor's ``low_idx`` (high-vol) is 10x the ATR at
        preceding's ``low_idx`` (low-vol). A correct implementation uses
        ``atr_arr[anchor.low_idx]`` and qualifies the pair. A buggy
        implementation that used ``preceding.low_idx`` or
        ``anchor.formation_bar_index`` would reject.
        """
        h, atr_arr = build_regime_shift_atr()
        zones = h.get_support_zones(atr_arr=atr_arr)
        anchor = next(z for z in zones if z.anchor_wave_id == "w-2")
        # Expected tolerance: 0.3 * atr_arr[3] = 0.3 * 0.5 = 0.15
        # Gap: |100.10 - 100.00| = 0.10 ≤ 0.15 → qualifies.
        assert anchor.is_double is True

    def test_nan_atr_array_falls_back_gracefully(self) -> None:
        """CR-007 — ``atr_arr`` is all-NaN. ``np.isfinite`` guard triggers
        for every anchor, and the percentage fallback qualifies the 0.15
        gap (0.004 x 100.15 = 0.4006 > 0.15).
        """
        h, atr_arr = build_nan_atr_bottoms()
        zones = h.get_support_zones(atr_arr=atr_arr)
        anchor = next(z for z in zones if z.anchor_wave_id == "w-2")
        assert anchor.is_double is True

    def test_negative_atr_array_falls_back_gracefully(self) -> None:
        """CR-008 — ``atr_arr`` contains negative values. ``atr_val > 0``
        guard triggers for every anchor, and the percentage fallback
        qualifies the 0.15 gap.
        """
        h, atr_arr = build_negative_atr_bottoms()
        zones = h.get_support_zones(atr_arr=atr_arr)
        anchor = next(z for z in zones if z.anchor_wave_id == "w-2")
        assert anchor.is_double is True

    def test_adjacent_wicks_count_as_overlap(self) -> None:
        """T019d — top of one wick equals bottom of the other. Inclusive
        ``range_overlaps`` returns True, and the deeper-wick extension fires.
        """
        h, atr_arr = build_adjacent_wicks()
        zones = h.get_support_zones(atr_arr=atr_arr)
        anchor = next(z for z in zones if z.anchor_wave_id == "w-2")
        assert anchor.is_double is True
        # Zone extended down to wave-0's low (100.00).
        assert anchor.range == pytest.approx((100.00, 101.00))
        # Wave-0 contributed via overlapping wicks, so its id is logged.
        assert "w-0" in anchor.overlapping_low_wave_ids

    def test_anchor_pairs_nearest_not_deepest(self) -> None:
        """T019e — under ``proximity=2`` both slot 0 (closer) and slot 1
        (deeper) qualify. ``overlapping_low_wave_ids`` must contain slot 0
        (guard against a 'widest-match' regression that might keep only
        the price-deepest match).
        """
        h, atr_arr = build_nearest_not_deepest()
        zones = h.get_support_zones(atr_arr=atr_arr, double_bottom_proximity=2)
        anchor = next(z for z in zones if z.anchor_wave_id == "w-4")
        assert anchor.is_double is True
        # Slot 0 = wave-2 (closer, higher low 100.05).
        assert "w-2" in anchor.overlapping_low_wave_ids
        # Slot 1 = wave-0 (deeper, lower low 99.90) — also present.
        assert "w-0" in anchor.overlapping_low_wave_ids


# ---------------------------------------------------------------------------
# Default proximity raised from 1 → 2 (W/M pattern admission)
# ---------------------------------------------------------------------------


class TestDefaultProximity:
    """Acceptance coverage for the raised default proximity (FR-007, FR-008)."""

    def test_w_pattern_admitted_under_new_default(self) -> None:
        """T026 — classical W: L1 (100.00) → L2 (100.50, higher) → L3 (100.10,
        matching L1). Under the new default ``double_bottom_proximity=2`` the
        anchor (L3) can reach L1 across the intermediate L2 and qualify as
        a double bottom.

        Note: ``is_double`` tracks price-tolerance qualification (FR-013).
        ``overlapping_low_wave_ids`` tracks wick-overlap geometry — a
        separate concern. The W-pattern's L1 matches L3 in price but
        their wicks are disjoint, so L1 will not appear in the overlap
        list.
        """
        h, atr_arr = build_w_pattern_with_intermediate()
        zones = h.get_support_zones(atr_arr=atr_arr)
        anchor = next(z for z in zones if z.anchor_wave_id == "w-4")
        assert anchor.is_double is True

    def test_proximity_one_preserves_old_behaviour(self) -> None:
        """T027 — explicit ``double_bottom_proximity=1`` keeps the old
        single-step lookback: the anchor can only see L2 (which doesn't
        match L3 within tolerance), so is_double=False.
        """
        h, atr_arr = build_w_pattern_with_intermediate()
        zones = h.get_support_zones(atr_arr=atr_arr, double_bottom_proximity=1)
        anchor = next(z for z in zones if z.anchor_wave_id == "w-4")
        assert anchor.is_double is False

    def test_m_pattern_admitted_under_new_default(self) -> None:
        """T028 (mirror of T026) — canonical M-pattern: H1 → H2 (lower) →
        H3 matching H1. New default ``double_top_proximity=2`` → double
        top is labelled. See sibling W-pattern test for the note on
        `is_double` vs. `overlapping_high_wave_ids`.
        """
        h, atr_arr = build_m_pattern_with_intermediate()
        zones = h.get_resistance_zones(atr_arr=atr_arr)
        anchor = next(z for z in zones if z.anchor_wave_id == "w-4")
        assert anchor.is_double is True

    def test_proximity_one_preserves_old_behaviour_top(self) -> None:
        """T028 (mirror of T027) — explicit ``double_top_proximity=1``
        rejects the same M-pattern.
        """
        h, atr_arr = build_m_pattern_with_intermediate()
        zones = h.get_resistance_zones(atr_arr=atr_arr, double_top_proximity=1)
        anchor = next(z for z in zones if z.anchor_wave_id == "w-4")
        assert anchor.is_double is False
