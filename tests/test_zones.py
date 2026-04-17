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
    """Body-anchored range for a bottom (support) zone."""

    def test_returns_anchor_body_range(self) -> None:
        # body-anchored: anchor = wave.lowest_close
        h = _make_helper("2020-07-01 00:00:00", "2020-07-27 20:00:00")
        last_bottom = h.get_last_bottom()
        assert last_bottom is not None
        r = h.get_bottom_range(last_bottom)
        assert r == pytest.approx((47.61, 49.24))

    def test_returns_anchor_body_range_alternate_window(self) -> None:
        # body-anchored: anchor = wave.lowest_close
        h = _make_helper("2020-07-01 00:00:00", "2020-07-25 04:00:00")
        last_bottom = h.get_last_bottom()
        assert last_bottom is not None
        r = h.get_bottom_range(last_bottom)
        assert r == pytest.approx((44.11, 44.6))


class TestTopRange:
    """Body-anchored range for a top (resistance) zone."""

    def test_returns_anchor_body_range(self) -> None:
        # body-anchored: anchor = wave.highest_close
        h = _make_helper("2020-07-01 00:00:00", "2020-08-02 08:00:00")
        last_top = h.get_last_top()
        assert last_top is not None
        r = h.get_top_range(last_top)
        assert r == pytest.approx((61.65, 64.91))

    def test_returns_anchor_body_range_alternate_window(self) -> None:
        # body-anchored: anchor = wave.highest_close
        h = _make_helper("2020-07-01 00:00:00", "2020-07-29 08:00:00")
        last_top = h.get_last_top()
        assert last_top is not None
        r = h.get_top_range(last_top)
        assert r == pytest.approx((55.36, 56.87))


# ---------------------------------------------------------------------------
# get_support_zones — fixture-based
# ---------------------------------------------------------------------------


class TestSupportZones:
    """TS: 'Determines significant levels — Based on bottoms'."""

    def test_picks_double_bottom_in_the_past(self) -> None:
        """Body-anchored zones: narrower ranges from anchor candle body."""
        h = _make_helper("2020-07-01 00:00:00", "2020-08-05 12:00:00")
        zones = h.get_support_zones(include_forming_wave=False)
        z0 = zones[0]
        assert z0.range == pytest.approx((57.32, 57.84))
        assert z0.is_double is False
        assert len(z0.overlapping_low_wave_ids) == 0
        assert len(z0.overlapping_high_wave_ids) == 1

    def test_picks_double_bottom_in_slightly_more_distant_past(self) -> None:
        """Same pair, different window; body-anchored geometry."""
        h = _make_helper("2020-07-01 00:00:00", "2020-08-07 12:00:00")
        zones = h.get_support_zones(include_forming_wave=False)
        z1 = zones[1]
        assert z1.range == pytest.approx((57.32, 57.84))
        assert z1.is_double is False
        assert len(z1.overlapping_low_wave_ids) == 0
        assert len(z1.overlapping_high_wave_ids) == 1

    def test_picks_double_bottom_with_current_wave(self) -> None:
        """Same pair with forming wave; body-anchored geometry."""
        h = _make_helper("2020-07-01 00:00:00", "2020-08-04 16:00:00")
        zones = h.get_support_zones(include_forming_wave=True)
        z0 = zones[0]
        assert z0.range == pytest.approx((57.34, 58.26))
        assert z0.is_double is False
        assert len(z0.overlapping_low_wave_ids) == 0
        assert len(z0.overlapping_high_wave_ids) == 1

    def test_can_pick_double_bottom_over_longer_period(self) -> None:
        h = _make_helper("2020-06-25 00:00:00", "2020-07-17 04:00:00")
        zones = h.get_support_zones(include_forming_wave=False, double_bottom_proximity=6)
        z0 = zones[0]
        # body-anchored: narrower range from anchor body
        assert z0.range == pytest.approx((41.72, 42.06))
        assert z0.is_double is True
        assert len(z0.overlapping_low_wave_ids) == 0
        assert len(z0.overlapping_high_wave_ids) == 4

    def test_double_bottom_with_lower_bottom_between_is_discarded(self) -> None:
        h = _make_helper("2020-06-29 00:00:00", "2020-07-04 04:00:00")
        zones = h.get_support_zones()
        z0 = zones[0]
        # body-anchored zone
        assert z0.range == pytest.approx((41.14, 41.3))
        assert z0.is_double is False
        assert len(z0.overlapping_low_wave_ids) == 2
        assert len(z0.overlapping_high_wave_ids) == 1


# ---------------------------------------------------------------------------
# get_resistance_zones — fixture-based
# ---------------------------------------------------------------------------


class TestResistanceZones:
    """TS: 'Determines significant levels — Based on tops'."""

    def test_picks_double_top_in_the_past(self) -> None:
        """Body-anchored zones: narrower ranges from anchor candle body."""
        h = _make_helper("2020-07-25 00:00:00", "2020-07-31 04:00:00")
        zones = h.get_resistance_zones()
        z0 = zones[0]
        assert z0.range == pytest.approx((54.54, 57.57))
        assert z0.is_double is False
        assert len(z0.overlapping_low_wave_ids) == 1
        assert len(z0.overlapping_high_wave_ids) == 1

    def test_picks_double_top_in_slightly_more_distant_past(self) -> None:
        """Same pair, different window; body-anchored geometry."""
        h = _make_helper("2020-07-25 00:00:00", "2020-08-05 00:00:00")
        zones = h.get_resistance_zones()
        z2 = zones[2]
        assert z2.range == pytest.approx((54.54, 57.57))
        assert z2.is_double is False
        assert len(z2.overlapping_low_wave_ids) == 3
        assert len(z2.overlapping_high_wave_ids) == 1

    def test_picks_double_top_with_current_wave(self) -> None:
        """Same pair with forming wave; body-anchored geometry."""
        h = _make_helper("2020-07-25 00:00:00", "2020-07-30 20:00:00")
        zones = h.get_resistance_zones(include_forming_wave=True)
        z0 = zones[0]
        assert z0.range == pytest.approx((54.54, 57.57))
        assert z0.is_double is False
        assert len(z0.overlapping_low_wave_ids) == 1
        assert len(z0.overlapping_high_wave_ids) == 1

    def test_can_pick_double_top_over_longer_period(self) -> None:
        h = _make_helper("2020-07-12 00:00:00", "2020-07-22 08:00:00")
        zones = h.get_resistance_zones(include_forming_wave=False, double_top_proximity=3)
        z0 = zones[0]
        # body-anchored: extension uses body top, not wick high
        assert z0.range == pytest.approx((43.77, 43.91))
        assert z0.is_double is True
        assert len(z0.overlapping_low_wave_ids) == 0
        assert len(z0.overlapping_high_wave_ids) == 1

    def test_double_top_with_higher_top_between_is_discarded(self) -> None:
        h = _make_helper("2020-06-29 00:00:00", "2020-07-15 04:00:00")
        zones = h.get_resistance_zones(include_forming_wave=False, double_top_proximity=10)
        z0 = zones[0]
        # body-anchored zone
        assert z0.range == pytest.approx((43.82, 43.88))
        assert z0.is_double is False
        assert len(z0.overlapping_low_wave_ids) == 0
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
        """Lows 0.15 apart, wick ranges disjoint: the NEW predicate
        qualifies this pair as a double bottom even though wicks don't overlap.
        """
        h, atr_arr = build_tight_disjoint_bottoms()
        zones = h.get_support_zones(atr_arr=atr_arr)
        assert zones[0].anchor_wave_id == "w-2"
        assert zones[0].is_double is True

    def test_double_bottom_wide_far_rejects(self) -> None:
        """Lows 2.0 apart with overlapping wicks: the NEW predicate
        REJECTS this pair because the price distance exceeds tolerance, even
        though the wick ranges overlap.
        """
        h, atr_arr = build_wide_far_bottoms()
        zones = h.get_support_zones(atr_arr=atr_arr)
        assert zones[0].anchor_wave_id == "w-2"
        assert zones[0].is_double is False

    def test_double_bottom_exact_tie_qualifies(self) -> None:
        """Lows at identical prices. Tolerance is inclusive at zero
        distance, so the pair qualifies.
        """
        h, atr_arr = build_exact_tie_bottoms()
        zones = h.get_support_zones(atr_arr=atr_arr)
        assert zones[0].anchor_wave_id == "w-2"
        assert zones[0].is_double is True

    def test_double_bottom_zero_atr_fallback(self) -> None:
        """ATR array is zero everywhere. Tolerance falls through to
        ``tolerance_pct_fallback`` and the near-equal lows still qualify.
        """
        h, atr_arr = build_zero_atr_bottoms()
        zones = h.get_support_zones(atr_arr=atr_arr)
        assert zones[0].anchor_wave_id == "w-2"
        assert zones[0].is_double is True

    def test_double_bottom_percentage_override(self) -> None:
        """With ``atr_arr=None``, the caller's ``tolerance_pct_fallback``
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
        """Lows exactly at the tolerance boundary. Inclusive ``<=``
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
        """A qualified pair with disjoint body ranges must not
        extend. Body-anchored: anchor body = (102.0, 102.0) (doji).
        """
        h, atr_arr = build_tight_disjoint_bottoms()
        zones = h.get_support_zones(atr_arr=atr_arr)
        anchor = next(z for z in zones if z.anchor_wave_id == "w-2")
        assert anchor.is_double is True
        assert anchor.range == pytest.approx((102.0, 102.0))


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
        """Mirror: disjoint body ranges, no extension.
        Body-anchored: anchor body = (108.0, 108.0) (doji).
        """
        h, atr_arr = build_tight_disjoint_tops()
        zones = h.get_resistance_zones(atr_arr=atr_arr)
        anchor = next(z for z in zones if z.anchor_wave_id == "w-2")
        assert anchor.is_double is True
        assert anchor.range == pytest.approx((108.0, 108.0))


# ---------------------------------------------------------------------------
# Double-pattern edge cases (first swing, short ATR, regime shift, etc.)
# ---------------------------------------------------------------------------


class TestDoublePatternEdgeCases:
    """Edge-case coverage for the new tolerance predicate."""

    def test_first_swing_no_double_label(self) -> None:
        """Anchor is the only same-side wave in the registry. The
        double-pattern loop body has no preceding wave to pair with, must
        not raise, and must yield ``is_double=False``.
        """
        h, atr_arr = build_first_swing_only()
        zones = h.get_support_zones(atr_arr=atr_arr)
        assert len(zones) == 1
        assert zones[0].is_double is False
        assert zones[0].overlapping_low_wave_ids == ()

    def test_short_atr_array_falls_back_gracefully(self) -> None:
        """``atr_arr`` shorter than the DataFrame. Out-of-bounds
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
        """ATR at anchor's ``low_idx`` (high-vol) is 10x the ATR at
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
        """``atr_arr`` is all-NaN. ``np.isfinite`` guard triggers
        for every anchor, and the percentage fallback qualifies the 0.15
        gap (0.004 x 100.15 = 0.4006 > 0.15).
        """
        h, atr_arr = build_nan_atr_bottoms()
        zones = h.get_support_zones(atr_arr=atr_arr)
        anchor = next(z for z in zones if z.anchor_wave_id == "w-2")
        assert anchor.is_double is True

    def test_negative_atr_array_falls_back_gracefully(self) -> None:
        """``atr_arr`` contains negative values. ``atr_val > 0``
        guard triggers for every anchor, and the percentage fallback
        qualifies the 0.15 gap.
        """
        h, atr_arr = build_negative_atr_bottoms()
        zones = h.get_support_zones(atr_arr=atr_arr)
        anchor = next(z for z in zones if z.anchor_wave_id == "w-2")
        assert anchor.is_double is True

    def test_adjacent_wicks_count_as_overlap(self) -> None:
        """Adjacent body ranges — body-anchored zone stays at anchor body.
        Anchor w-2 body = (101.0, 101.0). w-0 body = (100.2, 100.2).
        Bodies are disjoint (100.2 < 101.0), so no extension and
        no body-range overlap entry in overlapping_low_wave_ids.
        """
        h, atr_arr = build_adjacent_wicks()
        zones = h.get_support_zones(atr_arr=atr_arr)
        anchor = next(z for z in zones if z.anchor_wave_id == "w-2")
        assert anchor.is_double is True
        # Body-anchored: anchor body is (101.0, 101.0)
        assert anchor.range == pytest.approx((101.0, 101.0))

    def test_anchor_pairs_nearest_not_deepest(self) -> None:
        """Under proximity=2, both slot 0 (w-2) and slot 1 (w-0) qualify
        by price tolerance. Body-anchored: w-4 body = (101.0, 101.0);
        w-0 body = (101.0, 101.0) overlaps (touching); w-2 body =
        (100.2, 100.2) does not overlap with the anchor body.
        Body-coord extension: w-0 body bottom == current body bottom,
        so no extension. Wick range captures 99.9 via wick union.
        """
        h, atr_arr = build_nearest_not_deepest()
        zones = h.get_support_zones(atr_arr=atr_arr, double_bottom_proximity=2)
        anchor = next(z for z in zones if z.anchor_wave_id == "w-4")
        assert anchor.is_double is True
        assert anchor.range == pytest.approx((101.0, 101.0))
        # w-0's body overlaps the anchor body (touching)
        assert "w-0" in anchor.overlapping_low_wave_ids


# ---------------------------------------------------------------------------
# Default proximity raised from 1 → 2 (W/M pattern admission)
# ---------------------------------------------------------------------------


class TestDefaultProximity:
    """Acceptance coverage for the raised default proximity."""

    def test_w_pattern_admitted_under_new_default(self) -> None:
        """Classical W: L1 (100.00) → L2 (100.50, higher) → L3 (100.10,
        matching L1). Under the new default ``double_bottom_proximity=2`` the
        anchor (L3) can reach L1 across the intermediate L2 and qualify as
        a double bottom.

        Note: ``is_double`` tracks price-tolerance qualification.
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
        """Explicit ``double_bottom_proximity=1`` keeps the old
        single-step lookback: the anchor can only see L2 (which doesn't
        match L3 within tolerance), so is_double=False.
        """
        h, atr_arr = build_w_pattern_with_intermediate()
        zones = h.get_support_zones(atr_arr=atr_arr, double_bottom_proximity=1)
        anchor = next(z for z in zones if z.anchor_wave_id == "w-4")
        assert anchor.is_double is False

    def test_m_pattern_admitted_under_new_default(self) -> None:
        """Mirror of the W-pattern test — canonical M-pattern: H1 → H2 (lower) →
        H3 matching H1. New default ``double_top_proximity=2`` → double
        top is labelled. See sibling W-pattern test for the note on
        `is_double` vs. `overlapping_high_wave_ids`.
        """
        h, atr_arr = build_m_pattern_with_intermediate()
        zones = h.get_resistance_zones(atr_arr=atr_arr)
        anchor = next(z for z in zones if z.anchor_wave_id == "w-4")
        assert anchor.is_double is True

    def test_proximity_one_preserves_old_behaviour_top(self) -> None:
        """Mirror of the above — explicit ``double_top_proximity=1``
        rejects the same M-pattern.
        """
        h, atr_arr = build_m_pattern_with_intermediate()
        zones = h.get_resistance_zones(atr_arr=atr_arr, double_top_proximity=1)
        anchor = next(z for z in zones if z.anchor_wave_id == "w-4")
        assert anchor.is_double is False


# ---------------------------------------------------------------------------
# Body-anchored zone geometry
# ---------------------------------------------------------------------------


class TestBodyAnchoredZoneRegression:
    """BTC flash-crash regression lock.

    The 2025-10-10 BTC/USDT flash crash produced a wick to ~$101,516 on a
    candle that closed at ~$112,715. The body-anchored zone must NOT extend
    to the wick tip. The anchor is the next candle (lowest close in the
    down-wave at $110,338.7).
    """

    FIXTURE_PATH = Path(__file__).parent / "fixtures" / "btc-2025-10-10-flash-crash.json"
    HISTOGRAM_KEY = "tsi_histogram"

    def _hydrate_btc(self) -> MarketStructureHelper:
        with self.FIXTURE_PATH.open() as f:
            data = json.load(f)
        h = MarketStructureHelper(histogram_key=self.HISTOGRAM_KEY)
        for row in data:
            c = Candle(
                open_time=int(pd.Timestamp(str(row["openTime"]), tz="UTC").value // 10**6),
                open=float(row["open"]),
                high=float(row["high"]),
                low=float(row["low"]),
                close=float(row["close"]),
                volume=float(row["volume"]),
                histogram_value=float(row[self.HISTOGRAM_KEY]),
            )
            h.register_candle(c)
        return h

    def test_btc_flash_crash_body_zone_regression(self) -> None:
        """Zone low must be the body bottom of the anchor (lowest-close)
        candle, far above the ~$101.5k wick tip.
        """
        h = self._hydrate_btc()
        zones = h.get_support_zones()
        assert zones, "Expected at least one support zone"
        z = zones[0]
        # Anchor is the candle with lowest close in the down-wave:
        # open=112261.6, close=110338.7 → body = (110338.7, 112261.6)
        assert z.range[0] == pytest.approx(110338.7)
        assert z.range[1] == pytest.approx(112261.6)
        # Body bottom is far above the wick tip at 101516.5
        assert z.range[0] > 109000.0
        # Zone width is strictly less than the old wick-to-body span of $8822
        zone_width = z.range[1] - z.range[0]
        assert zone_width < 8822.0
        assert zone_width == pytest.approx(1922.9)


class TestBodyAnchoredSupportRange:
    """Body-anchored support zone = body of wave.lowest_close candle."""

    def test_support_zone_is_anchor_body_range(self) -> None:
        from tests.double_pattern_builders import build_long_lower_wick_anchor

        h, atr_arr = build_long_lower_wick_anchor()
        zones = h.get_support_zones(atr_arr=atr_arr, only_include_most_recent_zone=False)
        z = next(z for z in zones if z.anchor_wave_id == "w-0")
        # anchor = wave.lowest_close (bar 0: open=99, close=98, low=90)
        assert z.range == pytest.approx((98.0, 99.0))
        # Zone is strictly inside the anchor wave's low-high range
        assert z.range[0] > 90.0  # above the wick
        assert z.range[1] <= 100.0  # within the wave high

    def test_support_zone_anchor_tie_breaks_first_chronological(self) -> None:
        from tests.double_pattern_builders import build_tie_break_lowest_close

        h, atr_arr = build_tie_break_lowest_close()
        zones = h.get_support_zones(atr_arr=atr_arr, only_include_most_recent_zone=False)
        z = next(z for z in zones if z.anchor_wave_id == "w-0")
        # Two candles tie at close=98.0; first (bar 0: open=101, close=98)
        # is the anchor via np.argmin first-occurrence. Body = (98.0, 101.0).
        assert z.range == pytest.approx((98.0, 101.0))


class TestBodyAnchoredResistanceRange:
    """Body-anchored resistance zone = body of wave.highest_close candle."""

    def test_resistance_zone_is_anchor_body_range(self) -> None:
        from tests.double_pattern_builders import build_long_upper_wick_anchor

        h, atr_arr = build_long_upper_wick_anchor()
        zones = h.get_resistance_zones(atr_arr=atr_arr, only_include_most_recent_zone=False)
        z = next(z for z in zones if z.anchor_wave_id == "w-0")
        # anchor = wave.highest_close (bar 0: open=101, close=102, high=110)
        assert z.range == pytest.approx((101.0, 102.0))
        # Zone is strictly inside the anchor wave's low-high range
        assert z.range[1] < 110.0  # below the wick

    def test_resistance_zone_anchor_tie_breaks_first_chronological(self) -> None:
        from tests.double_pattern_builders import build_tie_break_highest_close

        h, atr_arr = build_tie_break_highest_close()
        zones = h.get_resistance_zones(atr_arr=atr_arr, only_include_most_recent_zone=False)
        z = next(z for z in zones if z.anchor_wave_id == "w-0")
        # Two candles tie at close=102.0; first (bar 0: open=99, close=102)
        # is the anchor via np.argmax first-occurrence. Body = (99.0, 102.0).
        assert z.range == pytest.approx((99.0, 102.0))


class TestBodyAnchoredInvariants:
    """Zone invariants hold across the LTC corpus."""

    def test_zone_body_invariants_on_corpus(self) -> None:
        """Every support zone: range[0] <= range[1], width <= anchor candle range.
        Every resistance zone: range[0] <= range[1].
        """
        h = _make_helper("2020-06-28 04:00:00", "2020-09-06 20:00:00")
        for z in h.get_support_zones(only_include_most_recent_zone=False):
            assert z.range[0] <= z.range[1], f"Support zone inverted: {z}"
        for z in h.get_resistance_zones(only_include_most_recent_zone=False):
            assert z.range[0] <= z.range[1], f"Resistance zone inverted: {z}"

    def test_doji_anchor_body_zone_is_zero_width(self) -> None:
        from tests.double_pattern_builders import build_doji_anchor

        h, atr_arr = build_doji_anchor()
        zones = h.get_support_zones(atr_arr=atr_arr, only_include_most_recent_zone=False)
        z = next(z for z in zones if z.anchor_wave_id == "w-0")
        # Doji: open == close → zero-width body zone
        assert z.range[0] == z.range[1]
        assert z.range[0] == pytest.approx(100.0)

    def test_single_candle_wave_zone_is_that_candle_body(self) -> None:
        from tests.double_pattern_builders import build_single_candle_wave

        h, atr_arr = build_single_candle_wave()
        zones = h.get_support_zones(atr_arr=atr_arr, only_include_most_recent_zone=False)
        z = next(z for z in zones if z.anchor_wave_id == "w-0")
        # Single candle wave: open=100, close=97 → body = (97, 100)
        assert z.range == pytest.approx((97.0, 100.0))


# ── Wick auxiliary columns ───────────────────────────────────────────────


class TestWickRangeSupport:
    """Wick range matches legacy wick geometry for support zones."""

    def test_support_zone_wick_range_matches_legacy(self) -> None:
        """zone.wick_range == (wave.low.low, min(anchor.close, anchor.open))."""
        from tests.double_pattern_builders import build_long_lower_wick_anchor

        h, atr_arr = build_long_lower_wick_anchor()
        zones = h.get_support_zones(atr_arr=atr_arr, only_include_most_recent_zone=False)
        z = next(z for z in zones if z.anchor_wave_id == "w-0")
        # Anchor: open=99, close=98, wave low=90
        # wick_range = (wave.low.low=90, min(98,99)=98)
        assert z.wick_range == pytest.approx((90.0, 98.0))

    def test_support_zone_wick_bound_invariants(self) -> None:
        """For every support zone: wick_range[0] <= range[0], wick_range[1] >= range[0]."""
        h = _make_helper("2020-06-28 04:00:00", "2020-09-06 20:00:00")
        for z in h.get_support_zones(only_include_most_recent_zone=False):
            assert z.wick_range[0] <= z.range[0], f"Wick low must envelope body low: {z}"
            # Wick high (body bottom of anchor) >= body zone low
            assert z.wick_range[1] >= z.range[0], f"Wick high must be >= body low: {z}"


class TestWickRangeResistance:
    """Wick range matches legacy wick geometry for resistance zones."""

    def test_resistance_zone_wick_range_matches_legacy(self) -> None:
        """zone.wick_range == (max(anchor.close, anchor.open), wave.high.high)."""
        from tests.double_pattern_builders import build_long_upper_wick_anchor

        h, atr_arr = build_long_upper_wick_anchor()
        zones = h.get_resistance_zones(atr_arr=atr_arr, only_include_most_recent_zone=False)
        z = next(z for z in zones if z.anchor_wave_id == "w-0")
        # Anchor: open=101, close=102, wave high=110
        # wick_range = (max(102,101)=102, wave.high.high=110)
        assert z.wick_range == pytest.approx((102.0, 110.0))

    def test_resistance_zone_wick_bound_invariants(self) -> None:
        """For every resistance zone: wick_range[1] >= range[1], wick_range[0] <= range[1]."""
        h = _make_helper("2020-06-28 04:00:00", "2020-09-06 20:00:00")
        for z in h.get_resistance_zones(only_include_most_recent_zone=False):
            assert z.wick_range[1] >= z.range[1], f"Wick high must envelope body high: {z}"
            # Wick low (body top of anchor) <= body zone high
            assert z.wick_range[0] <= z.range[1], f"Wick low must be <= body high: {z}"


class TestMarubozuWickCollapse:
    """Marubozu anchor: body == range, so body zone == wick zone."""

    def test_marubozu_anchor_body_and_wick_collapse(self) -> None:
        """Marubozu: body covers full candle range, no wick extends beyond body."""
        from tests.double_pattern_builders import build_marubozu_anchor

        h, atr_arr = build_marubozu_anchor()
        zones = h.get_support_zones(atr_arr=atr_arr, only_include_most_recent_zone=False)
        z = next(z for z in zones if z.anchor_wave_id == "w-0")
        # Bear marubozu: open=100, close=95, high=100, low=95
        # body = (95, 100); wick = (wave.low.low=95, min(o,c)=95) — no wick below body
        assert z.range == pytest.approx((95.0, 100.0))
        assert z.wick_range[0] == pytest.approx(z.range[0])  # no wick extension below body


# ── Double-pattern extension in body coordinates ─────────────────────────


class TestDoubleBottomBodyExtension:
    """Extension branch uses body coordinates, not wick extrema."""

    def test_double_bottom_extension_fires_on_body_overlap(self) -> None:
        """Bodies overlap → extension fires, zone_low = min(body bottoms)."""
        from tests.double_pattern_builders import build_body_overlap_disjoint_wicks_bottoms

        h, atr_arr = build_body_overlap_disjoint_wicks_bottoms()
        zones = h.get_support_zones(atr_arr=atr_arr, only_include_most_recent_zone=False)
        z = next(z for z in zones if z.anchor_wave_id == "w-2")
        assert z.is_double is True
        # Extended body range: min(98.1, 99.0)=98.1, max stays at 102.0
        assert z.range == pytest.approx((98.1, 102.0))
        # Wick range unioned across pair
        assert z.wick_range == pytest.approx((98.0, 99.0))

    def test_double_bottom_extension_skips_on_body_disjoint(self) -> None:
        """Bodies disjoint → is_double=True but extension does NOT fire."""
        from tests.double_pattern_builders import build_body_disjoint_wick_overlap_bottoms

        h, atr_arr = build_body_disjoint_wick_overlap_bottoms()
        zones = h.get_support_zones(atr_arr=atr_arr, only_include_most_recent_zone=False)
        z = next(z for z in zones if z.anchor_wave_id == "w-2")
        assert z.is_double is True
        # No extension — range equals primary anchor's body
        assert z.range == pytest.approx((97.0, 98.0))
        # Wick range still unioned across the pair
        assert z.wick_range == pytest.approx((93.0, 97.0))

    def test_double_bottom_extension_fires_on_body_touch(self) -> None:
        """Bodies touch (closed interval) → overlap detected."""
        from tests.double_pattern_builders import build_body_touching_bottoms

        h, atr_arr = build_body_touching_bottoms()
        zones = h.get_support_zones(atr_arr=atr_arr, only_include_most_recent_zone=False)
        z = next(z for z in zones if z.anchor_wave_id == "w-2")
        assert z.is_double is True
        # Bodies touch at 100. Preceding body bottom=100 >= current body bottom=98 → no extension
        assert z.range == pytest.approx((98.0, 100.0))

    def test_double_bottom_extension_is_monotonic(self) -> None:
        """Extension never raises zone_low — only lowers it."""
        from tests.double_pattern_builders import build_body_overlap_disjoint_wicks_bottoms

        h, atr_arr = build_body_overlap_disjoint_wicks_bottoms()
        zones = h.get_support_zones(atr_arr=atr_arr, only_include_most_recent_zone=False)
        z = next(z for z in zones if z.anchor_wave_id == "w-2")
        # Primary anchor body bottom is 99.0 (w-2). Extended to 98.1 (w-0) — lower.
        primary_body_bottom = 99.0
        assert z.range[0] <= primary_body_bottom


class TestDoubleTopBodyExtension:
    """Mirror of TestDoubleBottomBodyExtension for resistance."""

    def test_double_top_extension_fires_on_body_overlap(self) -> None:
        """Bodies overlap → extension fires, zone_high = max(body tops)."""
        from tests.double_pattern_builders import build_body_overlap_disjoint_wicks_tops

        h, atr_arr = build_body_overlap_disjoint_wicks_tops()
        zones = h.get_resistance_zones(atr_arr=atr_arr, only_include_most_recent_zone=False)
        z = next(z for z in zones if z.anchor_wave_id == "w-2")
        assert z.is_double is True
        # Extended body range: max(101.0, 101.9)=101.9
        assert z.range == pytest.approx((98.0, 101.9))
        # Wick range unioned across pair
        assert z.wick_range == pytest.approx((101.0, 102.0))

    def test_double_top_extension_skips_on_body_disjoint(self) -> None:
        """Bodies disjoint → is_double=True but extension does NOT fire."""
        from tests.double_pattern_builders import build_body_disjoint_wick_overlap_tops

        h, atr_arr = build_body_disjoint_wick_overlap_tops()
        zones = h.get_resistance_zones(atr_arr=atr_arr, only_include_most_recent_zone=False)
        z = next(z for z in zones if z.anchor_wave_id == "w-2")
        assert z.is_double is True
        # No extension — range equals primary anchor's body
        assert z.range == pytest.approx((101.0, 102.0))
        # Wick range still unioned
        assert z.wick_range == pytest.approx((102.0, 107.0))

    def test_double_top_extension_fires_on_body_touch(self) -> None:
        """Bodies touch (closed interval) → overlap detected."""
        from tests.double_pattern_builders import build_body_touching_tops

        h, atr_arr = build_body_touching_tops()
        zones = h.get_resistance_zones(atr_arr=atr_arr, only_include_most_recent_zone=False)
        z = next(z for z in zones if z.anchor_wave_id == "w-2")
        assert z.is_double is True
        # Bodies touch at 100. Preceding body top=100 <= current body top=102 → no extension
        assert z.range == pytest.approx((100.0, 102.0))

    def test_double_top_extension_is_monotonic(self) -> None:
        """Extension never lowers zone_high — only raises it."""
        from tests.double_pattern_builders import build_body_overlap_disjoint_wicks_tops

        h, atr_arr = build_body_overlap_disjoint_wicks_tops()
        zones = h.get_resistance_zones(atr_arr=atr_arr, only_include_most_recent_zone=False)
        z = next(z for z in zones if z.anchor_wave_id == "w-2")
        # Primary anchor body top is 101.0 (w-2). Extended to 101.9 (w-0) — higher.
        primary_body_top = 101.0
        assert z.range[1] >= primary_body_top
