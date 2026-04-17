"""Freqtrade integration — project market-structure state onto DataFrame columns.

The public entry point is ``attach_market_structure``, which a Freqtrade
strategy calls from ``populate_indicators``.  On the first call for a pair
(backtest or initial live window), it hydrates the full DataFrame and
projects per-bar columns via a post-hoc forward pass over the wave
registry.  Subsequent calls (live mode) register only the newest candle
and fill columns from current state.

Column names in the ``columns`` parameter use short names
(``"is_trending_up"``); the corresponding DataFrame column is prefixed
with ``ms_`` (``ms_is_trending_up``).
"""

# pyright: reportPrivateUsage=false

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from market_structure.atr import _compute_atr
from market_structure.hydrate import hydrate
from market_structure.types import Candle, Wave, Zone

if TYPE_CHECKING:
    from market_structure.helper import MarketStructureHelper


class MarketStructureDesyncError(Exception):
    """Raised when the helper's state is ahead of the DataFrame."""


# ------------------------------------------------------------------
# Column registry
# ------------------------------------------------------------------

VALID_COLUMNS: frozenset[str] = frozenset(
    {
        # Wave identity
        "wave_side",
        "wave_id",
        # Price levels & swing significance
        "last_top_price",
        "last_bottom_price",
        "made_higher_high",
        "made_higher_low",
        "made_lower_high",
        "made_lower_low",
        "high_since",
        "low_since",
        "forming_wave_high",
        "forming_wave_low",
        # Trend structure
        "is_trending_up",
        "is_trending_down",
        "wave_count",
        "trend_wave_count",
        "trend_duration",
        "structure_break_level",
        "structure_break_confirmed",
        "three_push_up",
        "three_push_down",
        # Wave metrics
        "wave_length",
        "wave_amplitude",
        "wave_slope",
        "wave_volume",
        "wave_volume_ratio",
        "wave_amplitude_ratio",
        # Pullback metrics
        "pullback_length",
        "pullback_correction_factor",
        "pullback_breakout_level",
        "pullback_price_diff",
        "pullback_atr_factor",
        # Divergence
        "bearish_divergence",
        "bullish_divergence",
        # Support/resistance zones
        "support_zone_low",
        "support_zone_high",
        "support_zone_wick_low",
        "support_zone_wick_high",
        "resistance_zone_low",
        "resistance_zone_high",
        "resistance_zone_wick_low",
        "resistance_zone_wick_high",
        "support_is_double",
        "resistance_is_double",
        "support_overlap_count",
        "resistance_overlap_count",
        "support_zone_anchor_time",
        "resistance_zone_anchor_time",
        "zone_quality_support",
        "zone_quality_resistance",
        # Zone lifecycle events
        "zone_break_support",
        "zone_break_resistance",
        "zone_retest_support",
        "zone_retest_resistance",
        "zone_retest_count_support",
        "zone_retest_count_resistance",
        "zone_flip_support",
        "zone_flip_resistance",
        "zone_failed_retest_support",
        "zone_failed_retest_resistance",
        # Volatility & distance
        "atr",
        "distance_to_support",
        "distance_to_resistance",
        # Swing failure pattern
        "sfp_high",
        "sfp_low",
        "bars_since_last_top",
        "bars_since_last_bottom",
    }
)


def _validate_columns(columns: tuple[str, ...] | None) -> tuple[str, ...]:
    if columns is None:
        return tuple(sorted(VALID_COLUMNS))
    unknown = set(columns) - VALID_COLUMNS
    if unknown:
        msg = f"Unknown column(s): {', '.join(sorted(unknown))}. Valid: {', '.join(sorted(VALID_COLUMNS))}"
        raise ValueError(msg)
    return columns


# ------------------------------------------------------------------
# Edge detection — convert sustained-True arrays into rising-edge-only
# ------------------------------------------------------------------


def _edge_detect(arr: np.ndarray) -> np.ndarray:
    """Return True only on the first bar of each contiguous True run."""
    result = np.zeros_like(arr, dtype=bool)
    result[0] = arr[0]
    result[1:] = arr[1:] & ~arr[:-1]
    return result


def _edge_detect_with_level(arr: np.ndarray, level: np.ndarray) -> np.ndarray:
    """Edge-detect but also re-fire when the reference *level* changes."""
    result = np.zeros_like(arr, dtype=bool)
    result[0] = arr[0]
    level_changed = np.zeros(len(arr), dtype=bool)
    level_changed[1:] = level[1:] != level[:-1]
    result[1:] = arr[1:] & (~arr[:-1] | level_changed[1:])
    return result


# ------------------------------------------------------------------
# Value extractors (mirror helper._hco_value / _lco_value)
# ------------------------------------------------------------------


def _hco_value(wave: Wave) -> float:
    c = wave.highest_close_or_open
    return max(c.close, c.open)


def _lco_value(wave: Wave) -> float:
    c = wave.lowest_close_or_open
    return min(c.close, c.open)


# ------------------------------------------------------------------
# Shared column formulas (used by both backtest and live paths)
# ------------------------------------------------------------------


def _compute_structure_break(
    last_top: Wave | None,
    last_bottom: Wave | None,
    prev_top: Wave | None,
    prev_bottom: Wave | None,
) -> tuple[float, bool | None]:
    """Compute structure break level and direction.

    Returns ``(level, is_uptrend)`` where *level* is the price that,
    if breached, confirms a structure break.  *is_uptrend* is True for
    HH+HL structure, False for LH+LL, or None when indeterminate.
    """
    if last_top and last_bottom and prev_top and prev_bottom:
        hh = _hco_value(last_top) > _hco_value(prev_top)
        hl = _lco_value(last_bottom) > _lco_value(prev_bottom)
        lh = _hco_value(last_top) < _hco_value(prev_top)
        ll = _lco_value(last_bottom) < _lco_value(prev_bottom)
        if hh and hl:
            return _lco_value(last_bottom), True
        if lh and ll:
            return _hco_value(last_top), False
    return np.nan, None


def _compute_trend_wave_count(tops: list[Wave], bottoms: list[Wave]) -> int:
    """Count consecutive HH+HL or LH+LL pairs walking backward.

    The lockstep index walk relies on the alternating wave invariant:
    histogram zero-crossings produce strictly alternating up/down waves,
    so ``tops[i]`` and ``bottoms[i]`` are adjacent in the wave sequence
    and form natural pairs for trend comparison.
    """
    count = 0
    if len(tops) >= 2 and len(bottoms) >= 2:
        ti = len(tops) - 1
        bi = len(bottoms) - 1
        while ti >= 1 and bi >= 1:
            t_hh = _hco_value(tops[ti]) > _hco_value(tops[ti - 1])
            b_hl = _lco_value(bottoms[bi]) > _lco_value(bottoms[bi - 1])
            t_lh = _hco_value(tops[ti]) < _hco_value(tops[ti - 1])
            b_ll = _lco_value(bottoms[bi]) < _lco_value(bottoms[bi - 1])
            if (t_hh and b_hl) or (t_lh and b_ll):
                count += 1
            else:
                break
            ti -= 1
            bi -= 1
    return count


def _is_three_push_up(tops: list[Wave]) -> bool:
    """Detect three-push exhaustion pattern on up-waves.

    Requires three consecutive tops with diminishing amplitude,
    successive higher highs, and convergent extremes.  Amplitude uses
    full wave range (high - low) because waves alternate direction, so
    H-L captures the directional move for each wave type.
    """
    if len(tops) < 3:
        return False
    t1, t2, t3 = tops[-3], tops[-2], tops[-1]
    h1, h2, h3 = _hco_value(t1), _hco_value(t2), _hco_value(t3)
    a1 = t1.high.high - t1.low.low
    a2 = t2.high.high - t2.low.low
    a3 = t3.high.high - t3.low.low
    return a3 < a2 < a1 and h3 > h2 > h1 and (h3 - h2) < (h2 - h1)


def _is_three_push_down(bottoms: list[Wave]) -> bool:
    """Detect three-push exhaustion pattern on down-waves."""
    if len(bottoms) < 3:
        return False
    b1, b2, b3 = bottoms[-3], bottoms[-2], bottoms[-1]
    l1, l2, l3 = _lco_value(b1), _lco_value(b2), _lco_value(b3)
    a1 = b1.high.high - b1.low.low
    a2 = b2.high.high - b2.low.low
    a3 = b3.high.high - b3.low.low
    return a3 < a2 < a1 and l3 < l2 < l1 and (l2 - l3) < (l1 - l2)


def _compute_wave_amplitude_ratio(
    wave: Wave,
    prev_same: Wave | None,
    atr_arr: np.ndarray,
) -> float:
    """Ratio of current wave's ATR-normalized amplitude to previous same-direction wave."""
    fb = wave.formation_bar_index
    atr_at_fb = atr_arr[fb] if fb < len(atr_arr) else np.nan
    cur_amp = (
        (wave.high.high - wave.low.low) / atr_at_fb
        if not np.isnan(atr_at_fb) and atr_at_fb > 0
        else np.nan
    )
    if prev_same is None or np.isnan(cur_amp):
        return np.nan
    prev_fb = prev_same.formation_bar_index
    prev_atr = atr_arr[prev_fb] if prev_fb < len(atr_arr) else np.nan
    prev_amp = (
        (prev_same.high.high - prev_same.low.low) / prev_atr
        if not np.isnan(prev_atr) and prev_atr > 0
        else np.nan
    )
    return cur_amp / prev_amp if not np.isnan(prev_amp) and prev_amp > 0 else np.nan


# ------------------------------------------------------------------
# Zone snapshot helper
# ------------------------------------------------------------------


def _zone_anchor_time(zone: Zone, registry_slice: list[Wave]) -> float:
    """Compute the earliest defining candle timestamp for a zone.

    For support zones (side="down"), the zone is defined by the anchor
    wave's low wick and LCO body candles.  For resistance zones
    (side="up"), by the high wick and HCO body candles.

    When the zone is a double and its range was extended by a preceding
    wave's wick, we also consider that preceding wave's defining candle
    so the rectangle starts early enough.
    """
    wave_by_id: dict[str, Wave] = {w.id: w for w in registry_slice}
    anchor = wave_by_id.get(zone.anchor_wave_id)
    if anchor is None:
        return np.nan

    if zone.side == "down":
        # Support zone: anchor is a down-wave
        anchor_time = min(anchor.low.open_time, anchor.lowest_close_or_open.open_time)
        # If double-bottom extended the range, check the preceding wave
        if zone.is_double:
            for wid in zone.overlapping_low_wave_ids:
                w = wave_by_id.get(wid)
                if w is not None and w.low.low <= zone.range[0]:
                    anchor_time = min(anchor_time, w.low.open_time)
    else:
        # Resistance zone: anchor is an up-wave
        anchor_time = min(anchor.high.open_time, anchor.highest_close_or_open.open_time)
        # If double-top extended the range, check the preceding wave
        if zone.is_double:
            for wid in zone.overlapping_high_wave_ids:
                w = wave_by_id.get(wid)
                if w is not None and w.high.high >= zone.range[1]:
                    anchor_time = min(anchor_time, w.high.open_time)

    return float(anchor_time)


def _snapshot_zones(
    snap: dict[str, object],
    helper: MarketStructureHelper,
    registry_slice: list[Wave],
    tops: list[Wave],
    bottoms: list[Wave],
    columns: tuple[str, ...],
    atr_arr: np.ndarray,
) -> None:
    """Capture nearest support/resistance zone state into *snap*.

    Temporarily swaps the helper's internal wave lists to match the
    accumulated state at this snapshot point, queries zone methods
    (reusing the existing zone logic exactly), then restores the
    originals.  This guarantees parity with the brute-force reference
    where ``get_support_zones`` / ``get_resistance_zones`` see only
    the waves confirmed so far.
    """
    col_set = frozenset(columns)
    need_support = bool(
        {
            "support_zone_low",
            "support_zone_high",
            "support_zone_wick_low",
            "support_zone_wick_high",
            "support_is_double",
            "support_overlap_count",
            "support_zone_anchor_time",
            "zone_quality_support",
        }
        & col_set
    )
    need_resistance = bool(
        {
            "resistance_zone_low",
            "resistance_zone_high",
            "resistance_zone_wick_low",
            "resistance_zone_wick_high",
            "resistance_is_double",
            "resistance_overlap_count",
            "resistance_zone_anchor_time",
            "zone_quality_resistance",
        }
        & col_set
    )

    # Save original helper state.
    orig_registry = helper._wave_registry
    orig_tops = helper._top_waves
    orig_bottoms = helper._bottom_waves
    orig_cache_s = helper._zone_cache_support
    orig_cache_r = helper._zone_cache_resistance

    # Temporarily set helper to the accumulated state at this wave boundary.
    helper._wave_registry = registry_slice
    helper._top_waves = tops
    helper._bottom_waves = bottoms
    helper._zone_cache_support = None
    helper._zone_cache_resistance = None

    try:
        sz: Zone | None = None
        rz: Zone | None = None

        if need_support:
            zones = helper.get_support_zones(atr_arr=atr_arr)
            sz = zones[0] if zones else None
        if need_resistance:
            zones = helper.get_resistance_zones(atr_arr=atr_arr)
            rz = zones[0] if zones else None
    finally:
        # Restore original helper state.
        helper._wave_registry = orig_registry
        helper._top_waves = orig_tops
        helper._bottom_waves = orig_bottoms
        helper._zone_cache_support = orig_cache_s
        helper._zone_cache_resistance = orig_cache_r

    if "support_zone_low" in col_set:
        snap["support_zone_low"] = sz.range[0] if sz else np.nan
    if "support_zone_high" in col_set:
        snap["support_zone_high"] = sz.range[1] if sz else np.nan
    if "support_zone_wick_low" in col_set:
        snap["support_zone_wick_low"] = sz.wick_range[0] if sz else np.nan
    if "support_zone_wick_high" in col_set:
        snap["support_zone_wick_high"] = sz.wick_range[1] if sz else np.nan
    if "support_is_double" in col_set:
        snap["support_is_double"] = sz.is_double if sz else pd.NA
    if "support_overlap_count" in col_set:
        snap["support_overlap_count"] = len(sz.overlapping_low_wave_ids) if sz else pd.NA
    if "support_zone_anchor_time" in col_set:
        snap["support_zone_anchor_time"] = _zone_anchor_time(sz, registry_slice) if sz else np.nan
    snap["_support_zone_obj"] = sz

    if "resistance_zone_low" in col_set:
        snap["resistance_zone_low"] = rz.range[0] if rz else np.nan
    if "resistance_zone_high" in col_set:
        snap["resistance_zone_high"] = rz.range[1] if rz else np.nan
    if "resistance_zone_wick_low" in col_set:
        snap["resistance_zone_wick_low"] = rz.wick_range[0] if rz else np.nan
    if "resistance_zone_wick_high" in col_set:
        snap["resistance_zone_wick_high"] = rz.wick_range[1] if rz else np.nan
    if "resistance_is_double" in col_set:
        snap["resistance_is_double"] = rz.is_double if rz else pd.NA
    if "resistance_overlap_count" in col_set:
        snap["resistance_overlap_count"] = len(rz.overlapping_high_wave_ids) if rz else pd.NA
    if "resistance_zone_anchor_time" in col_set:
        snap["resistance_zone_anchor_time"] = (
            _zone_anchor_time(rz, registry_slice) if rz else np.nan
        )
    snap["_resistance_zone_obj"] = rz


# ------------------------------------------------------------------
# Backtest projection
# ------------------------------------------------------------------


def _score_zone_quality(
    zone: Zone,
    atr: float,
    bars_since_formed: int,
) -> float:
    """Composite zone quality score (0-10).

    Formula:
    - overlap count: min(count, 5) * 0.8 (max 4.0)
    - double status: 2.0 if is_double else 0.0
    - ATR-normalized width: max(0, 2.0 - width_in_atr * 2) (max 2.0)
    - Recency: 2.0 * exp(-bars_since_formed / 100) (max 2.0)
    - Touch count (overlap as proxy): min(touches, 3) * 0.7 (max 2.1)
    """
    overlap_count = (
        len(zone.overlapping_low_wave_ids)
        if zone.side == "down"
        else len(zone.overlapping_high_wave_ids)
    )

    score = min(overlap_count, 5) * 0.8
    score += 2.0 if zone.is_double else 0.0

    if not np.isnan(atr) and atr > 0:
        width = zone.range[1] - zone.range[0]
        width_in_atr = width / atr
        score += max(0.0, 2.0 - width_in_atr * 2)
    # else: skip width component (NaN ATR)

    score += 2.0 * math.exp(-bars_since_formed / 100)

    # Touch count: use the opposite side's overlapping waves as a proxy
    # (the same-side overlaps are already captured in overlap_count above).
    touch_count = (
        len(zone.overlapping_high_wave_ids)
        if zone.side == "down"
        else len(zone.overlapping_low_wave_ids)
    )
    score += min(touch_count, 3) * 0.7

    return min(score, 10.0)


def _project_backtest(
    df: pd.DataFrame,
    helper: MarketStructureHelper,
    hist_col: str,
    columns: tuple[str, ...],
    atr_arr: np.ndarray,
    retest_mode: str = "wick",
) -> None:
    """Project requested columns onto *df* using the full wave registry.

    Mutates *df* in-place.  Assumes *helper* was hydrated with eviction
    disabled so ``_wave_registry`` contains every confirmed wave.
    """
    col_set = frozenset(columns)
    n = len(df)
    if n == 0:
        return

    waves = helper._wave_registry

    # ------------------------------------------------------------------
    # Numpy pre-pass — recompute wave boundaries (same logic as hydrate)
    # ------------------------------------------------------------------
    hist = df[hist_col].to_numpy(dtype=float)
    sign = np.where(hist >= 0, 1, -1)
    flip = np.empty(n, dtype=bool)
    flip[0] = False
    flip[1:] = sign[1:] != sign[:-1]

    flip_indices = np.flatnonzero(flip)
    n_flips = len(flip_indices)
    n_waves = n_flips + 1
    n_confirmed = n_waves - 1

    wave_starts = np.empty(n_waves, dtype=np.intp)
    wave_starts[0] = 0
    if n_flips > 0:
        wave_starts[1:] = flip_indices

    wave_ends = np.empty(n_waves, dtype=np.intp)
    if n_flips > 0:
        wave_ends[:n_flips] = flip_indices - 1
    wave_ends[-1] = n - 1

    # ------------------------------------------------------------------
    # Tier 1 — wave-constant columns
    # ------------------------------------------------------------------
    need_wave_side = "wave_side" in col_set
    need_wave_id = "wave_id" in col_set

    if need_wave_side:
        side_arr = np.empty(n, dtype=object)
        for wi in range(n_waves):
            s = slice(int(wave_starts[wi]), int(wave_ends[wi]) + 1)
            side_arr[s] = "up" if sign[int(wave_starts[wi])] >= 0 else "down"
        df["ms_wave_side"] = side_arr

    if need_wave_id:
        id_arr = np.empty(n, dtype=object)
        for wi in range(n_confirmed):
            s = slice(int(wave_starts[wi]), int(wave_ends[wi]) + 1)
            id_arr[s] = waves[wi].id
        # Forming wave
        forming_start = int(wave_starts[-1])
        forming_end = int(wave_ends[-1])
        forming_id = f"forming-{helper._next_wave_id}"
        id_arr[forming_start : forming_end + 1] = forming_id
        df["ms_wave_id"] = id_arr

    # ------------------------------------------------------------------
    # Tier 2 + Tier 3 base — forward pass over wave list
    # ------------------------------------------------------------------
    tier1_only = {"wave_side", "wave_id"}
    need_tier2_3 = bool(set(columns) - tier1_only)

    if not need_tier2_3:
        return

    # Build per-wave snapshots via a forward pass.
    snapshots = _compute_snapshots(waves, columns, helper, atr_arr)

    # Broadcast snapshots to bars.
    # Bars in wave 0 (before any confirmed wave): defaults.
    # Bars in wave j (j >= 1): snapshot[j - 1].
    # Forming wave bars: snapshot[last].

    _broadcast_tier2(df, snapshots, wave_starts, wave_ends, n_confirmed, n_waves, columns)

    # ------------------------------------------------------------------
    # Tier 3 — trending correction
    # ------------------------------------------------------------------
    lows = df["low"].to_numpy(dtype=float)
    highs = df["high"].to_numpy(dtype=float)

    if "is_trending_up" in col_set:
        _correct_trending_up(
            df, snapshots, wave_starts, wave_ends, n_confirmed, n_waves, sign, lows
        )

    if "is_trending_down" in col_set:
        _correct_trending_down(
            df, snapshots, wave_starts, wave_ends, n_confirmed, n_waves, sign, highs
        )

    # ------------------------------------------------------------------
    # Tier 3 — forming wave extremes (per-bar running high/low)
    # ------------------------------------------------------------------
    if "forming_wave_high" in col_set:
        arr = np.full(n, np.nan)
        for wi in range(n_waves):
            ws = int(wave_starts[wi])
            we = int(wave_ends[wi])
            arr[ws : we + 1] = np.maximum.accumulate(highs[ws : we + 1])
        df["ms_forming_wave_high"] = arr

    if "forming_wave_low" in col_set:
        arr = np.full(n, np.nan)
        for wi in range(n_waves):
            ws = int(wave_starts[wi])
            we = int(wave_ends[wi])
            arr[ws : we + 1] = np.minimum.accumulate(lows[ws : we + 1])
        df["ms_forming_wave_low"] = arr

    # ------------------------------------------------------------------
    # Tier 3 — ATR, distance, bars_since, SFP, structure break confirmed
    # ------------------------------------------------------------------
    closes = df["close"].to_numpy(dtype=float)

    if "atr" in col_set:
        df["ms_atr"] = atr_arr

    if "distance_to_support" in col_set or "distance_to_resistance" in col_set:
        # Sign convention: both distances are positive when the zone is in its
        # expected position (support below close, resistance above close).
        # distance_to_support  = (close - zone_high) / ATR  → positive when above support
        # distance_to_resistance = (zone_low - close) / ATR → positive when below resistance
        need_dist_s = "distance_to_support" in col_set
        need_dist_r = "distance_to_resistance" in col_set
        if need_dist_s:
            sup_high = (
                df["ms_support_zone_high"].to_numpy(dtype=float)
                if "ms_support_zone_high" in df.columns
                else np.full(n, np.nan)
            )
            df["ms_distance_to_support"] = np.where(
                np.isnan(sup_high) | np.isnan(atr_arr) | (atr_arr == 0),
                np.nan,
                (closes - sup_high) / atr_arr,
            )
        if need_dist_r:
            res_low = (
                df["ms_resistance_zone_low"].to_numpy(dtype=float)
                if "ms_resistance_zone_low" in df.columns
                else np.full(n, np.nan)
            )
            df["ms_distance_to_resistance"] = np.where(
                np.isnan(res_low) | np.isnan(atr_arr) | (atr_arr == 0),
                np.nan,
                (res_low - closes) / atr_arr,
            )

    if "bars_since_last_top" in col_set or "bars_since_last_bottom" in col_set:
        # Walk wave boundaries to find bars where new tops/bottoms are confirmed.
        top_confirm = np.zeros(n, dtype=bool)
        bottom_confirm = np.zeros(n, dtype=bool)
        for wi in range(n_confirmed):
            w = waves[wi]
            fb = w.formation_bar_index
            if fb < n:
                if w.side == "up":
                    top_confirm[fb] = True
                else:
                    bottom_confirm[fb] = True

        if "bars_since_last_top" in col_set:
            # Vectorized bars-since: forward-fill confirmation bar indices,
            # then subtract from the bar index to get distance.
            confirm_indices = np.flatnonzero(top_confirm)
            if len(confirm_indices) > 0:
                # Build array of "most recent confirmation bar" per bar.
                last_bar = np.full(n, -1, dtype=np.int32)
                last_bar[confirm_indices] = confirm_indices
                # Forward-fill using maximum.accumulate (works because indices increase).
                np.maximum.accumulate(last_bar, out=last_bar)
                bar_idx = np.arange(n, dtype=np.int32)
                arr = np.where(last_bar >= 0, bar_idx - last_bar, -1)
            else:
                arr = np.full(n, -1, dtype=np.int32)
            result = pd.array(np.where(arr >= 0, arr, pd.NA), dtype="Int32")  # type: ignore[arg-type]
            df["ms_bars_since_last_top"] = result

        if "bars_since_last_bottom" in col_set:
            confirm_indices = np.flatnonzero(bottom_confirm)
            if len(confirm_indices) > 0:
                last_bar = np.full(n, -1, dtype=np.int32)
                last_bar[confirm_indices] = confirm_indices
                np.maximum.accumulate(last_bar, out=last_bar)
                bar_idx = np.arange(n, dtype=np.int32)
                arr = np.where(last_bar >= 0, bar_idx - last_bar, -1)
            else:
                arr = np.full(n, -1, dtype=np.int32)
            result = pd.array(np.where(arr >= 0, arr, pd.NA), dtype="Int32")  # type: ignore[arg-type]
            df["ms_bars_since_last_bottom"] = result

    if "sfp_high" in col_set or "sfp_low" in col_set:
        # SFP high: wick exceeds last top HCO but close rejects below it.
        # SFP low: wick drops below last bottom LCO but close rejects above it.
        # Event semantics: fires only on the first bar of each cluster.
        # Re-fires when the reference level changes (new swing confirmed).
        if "sfp_high" in col_set:
            ltp = (
                df["ms_last_top_price"].to_numpy(dtype=float)
                if "ms_last_top_price" in df.columns
                else np.full(n, np.nan)
            )
            sfp_h_raw = np.where(
                np.isnan(ltp),
                False,
                (highs > ltp) & (closes < ltp),
            )
            sfp_h = _edge_detect_with_level(sfp_h_raw, ltp)
            df["ms_sfp_high"] = pd.array(sfp_h, dtype="boolean")  # type: ignore[arg-type]

        if "sfp_low" in col_set:
            lbp = (
                df["ms_last_bottom_price"].to_numpy(dtype=float)
                if "ms_last_bottom_price" in df.columns
                else np.full(n, np.nan)
            )
            sfp_l_raw = np.where(
                np.isnan(lbp),
                False,
                (lows < lbp) & (closes > lbp),
            )
            sfp_l = _edge_detect_with_level(sfp_l_raw, lbp)
            df["ms_sfp_low"] = pd.array(sfp_l, dtype="boolean")  # type: ignore[arg-type]

    if "structure_break_confirmed" in col_set:
        # True on bar where close crosses structure_break_level.
        # Use the hidden _structure_break_is_uptrend from snapshots to determine direction.
        sbl_arr = np.full(n, np.nan)
        is_up_arr = np.full(n, False, dtype=bool)
        is_down_arr = np.full(n, False, dtype=bool)

        # Broadcast from snapshots (same pattern as Tier 2)
        for j in range(1, n_confirmed):
            s = slice(int(wave_starts[j]), int(wave_ends[j]) + 1)
            snap = snapshots[j - 1]
            if "structure_break_level" in snap:
                sbl_arr[s] = snap["structure_break_level"]  # type: ignore[assignment]
            direction = snap.get("_structure_break_is_uptrend")
            if direction is True:
                is_up_arr[s] = True
            elif direction is False:
                is_down_arr[s] = True

        if snapshots:
            last_snap = snapshots[-1]
            forming_s = slice(int(wave_starts[n_waves - 1]), int(wave_ends[n_waves - 1]) + 1)
            if "structure_break_level" in last_snap:
                sbl_arr[forming_s] = last_snap["structure_break_level"]  # type: ignore[assignment]
            direction = last_snap.get("_structure_break_is_uptrend")
            if direction is True:
                is_up_arr[forming_s] = True
            elif direction is False:
                is_down_arr[forming_s] = True

        sbc_raw = np.where(
            np.isnan(sbl_arr),
            False,
            (is_up_arr & (closes < sbl_arr)) | (is_down_arr & (closes > sbl_arr)),
        )
        # Event semantics: fire only on the first bar that crosses the level.
        # Re-fire when the break level itself changes (new wave boundary).
        sbc = _edge_detect_with_level(sbc_raw, sbl_arr)
        df["ms_structure_break_confirmed"] = pd.array(sbc, dtype="boolean")  # type: ignore[arg-type]

    # ------------------------------------------------------------------
    # Tier 3 — zone lifecycle (break / retest / flip / failed retest)
    # ------------------------------------------------------------------
    lifecycle_cols = {
        "zone_break_support",
        "zone_break_resistance",
        "zone_retest_support",
        "zone_retest_resistance",
        "zone_retest_count_support",
        "zone_retest_count_resistance",
        "zone_flip_support",
        "zone_flip_resistance",
        "zone_failed_retest_support",
        "zone_failed_retest_resistance",
    }
    need_lifecycle = bool(lifecycle_cols & col_set)
    if need_lifecycle:
        _project_zone_lifecycle(
            df,
            n,
            snapshots,
            wave_starts,
            wave_ends,
            n_confirmed,
            n_waves,
            highs,
            lows,
            closes,
            columns,
            retest_mode,
        )

    # ------------------------------------------------------------------
    # Tier 3 — trend duration (per-bar counter since trend epoch)
    # ------------------------------------------------------------------
    if "trend_duration" in col_set:
        # Use the corrected per-bar trending columns to determine epoch.
        up_arr = (
            df["ms_is_trending_up"].to_numpy(dtype=bool)
            if "ms_is_trending_up" in df.columns
            else np.zeros(n, dtype=bool)
        )
        down_arr = (
            df["ms_is_trending_down"].to_numpy(dtype=bool)
            if "ms_is_trending_down" in df.columns
            else np.zeros(n, dtype=bool)
        )
        td_arr = np.full(n, pd.NA, dtype=object)
        td_epoch: int | None = None
        td_prev_trending = False
        for i in range(n):
            is_trending = bool(up_arr[i]) or bool(down_arr[i])
            if is_trending and not td_prev_trending:
                td_epoch = i
            elif not is_trending:
                td_epoch = None
            if td_epoch is not None and is_trending:
                td_arr[i] = i - td_epoch
            td_prev_trending = is_trending
        df["ms_trend_duration"] = pd.array(td_arr, dtype="Int32")  # type: ignore[arg-type]


def _project_zone_lifecycle(
    df: pd.DataFrame,
    n: int,
    snapshots: list[dict[str, object]],
    wave_starts: np.ndarray,
    wave_ends: np.ndarray,
    n_confirmed: int,
    n_waves: int,
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    columns: tuple[str, ...],
    retest_mode: str,
) -> None:
    """Forward pass computing zone lifecycle events per bar.

    State machine per zone side (support / resistance):
      intact -> broken -> retested -> {flipped | failed_retest -> intact}

    A zone is identified by its (low, high) range from Tier 2 snapshots.
    When a new zone replaces an old one at a wave boundary, the lifecycle
    resets to intact.
    """
    col_set = frozenset(columns)
    # Initialize output arrays.
    break_s = np.zeros(n, dtype=bool)
    break_r = np.zeros(n, dtype=bool)
    retest_s = np.zeros(n, dtype=bool)
    retest_r = np.zeros(n, dtype=bool)
    retest_count_s = np.zeros(n, dtype=np.int32)
    retest_count_r = np.zeros(n, dtype=np.int32)
    flip_s = np.zeros(n, dtype=bool)
    flip_r = np.zeros(n, dtype=bool)
    failed_retest_s = np.zeros(n, dtype=bool)
    failed_retest_r = np.zeros(n, dtype=bool)

    # Per-side state tracking.
    # State: "intact", "broken", "retested"
    sup_state = "intact"
    res_state = "intact"
    sup_zone: tuple[float, float] | None = None  # (low, high)
    res_zone: tuple[float, float] | None = None
    sup_retest_count = 0
    res_retest_count = 0

    # Build per-bar zone arrays from snapshots.
    sup_low_arr = np.full(n, np.nan)
    sup_high_arr = np.full(n, np.nan)
    res_low_arr = np.full(n, np.nan)
    res_high_arr = np.full(n, np.nan)

    for j in range(1, n_confirmed):
        s = slice(int(wave_starts[j]), int(wave_ends[j]) + 1)
        snap = snapshots[j - 1]
        sl = snap.get("support_zone_low", np.nan)
        sh = snap.get("support_zone_high", np.nan)
        rl = snap.get("resistance_zone_low", np.nan)
        rh = snap.get("resistance_zone_high", np.nan)
        sup_low_arr[s] = sl  # type: ignore[assignment]
        sup_high_arr[s] = sh  # type: ignore[assignment]
        res_low_arr[s] = rl  # type: ignore[assignment]
        res_high_arr[s] = rh  # type: ignore[assignment]

    if snapshots:
        last_snap = snapshots[-1]
        forming_s = slice(int(wave_starts[n_waves - 1]), int(wave_ends[n_waves - 1]) + 1)
        sup_low_arr[forming_s] = last_snap.get("support_zone_low", np.nan)  # type: ignore[assignment]
        sup_high_arr[forming_s] = last_snap.get("support_zone_high", np.nan)  # type: ignore[assignment]
        res_low_arr[forming_s] = last_snap.get("resistance_zone_low", np.nan)  # type: ignore[assignment]
        res_high_arr[forming_s] = last_snap.get("resistance_zone_high", np.nan)  # type: ignore[assignment]

    # Forward pass — evaluate lifecycle transitions bar by bar.
    for i in range(n):
        c = closes[i]
        h = highs[i]
        lo = lows[i]

        # Current zone for this bar.
        cur_sup = (sup_low_arr[i], sup_high_arr[i]) if not np.isnan(sup_low_arr[i]) else None
        cur_res = (res_low_arr[i], res_high_arr[i]) if not np.isnan(res_low_arr[i]) else None

        # Zone change detection — reset lifecycle when zone changes.
        if cur_sup != sup_zone:
            sup_zone = cur_sup
            sup_state = "intact"
            sup_retest_count = 0
        if cur_res != res_zone:
            res_zone = cur_res
            res_state = "intact"
            res_retest_count = 0

        # Support lifecycle.
        if sup_zone is not None:
            z_low, z_high = sup_zone
            if sup_state == "intact":
                # Break: close below zone low.
                if c < z_low:
                    sup_state = "broken"
                    break_s[i] = True
            elif sup_state == "broken":
                # Retest: price returns to touch the zone from below.
                touched = (h >= z_low) if retest_mode == "wick" else (c >= z_low and c <= z_high)
                if touched:
                    sup_state = "retested"
                    sup_retest_count += 1
                    retest_s[i] = True
            elif sup_state == "retested":
                # Flip: close back below zone (confirms role reversal to resistance).
                if c < z_low:
                    sup_state = "intact"  # reset — zone now acts as resistance conceptually
                    flip_s[i] = True
                    sup_retest_count = 0
                # Failed retest: close back above zone high (zone resumes as support).
                elif c > z_high:
                    sup_state = "intact"
                    failed_retest_s[i] = True
                    sup_retest_count = 0
                # Additional retest while in retested state: another touch.
                elif retest_mode == "wick" and lo <= z_high and h >= z_low:
                    pass  # still in retested state
            retest_count_s[i] = sup_retest_count

        # Resistance lifecycle.
        if res_zone is not None:
            z_low, z_high = res_zone
            if res_state == "intact":
                # Break: close above zone high.
                if c > z_high:
                    res_state = "broken"
                    break_r[i] = True
            elif res_state == "broken":
                # Retest: price returns to touch the zone from above.
                touched = (lo <= z_high) if retest_mode == "wick" else (c >= z_low and c <= z_high)
                if touched:
                    res_state = "retested"
                    res_retest_count += 1
                    retest_r[i] = True
            elif res_state == "retested":
                # Flip: close back above zone (confirms role reversal to support).
                if c > z_high:
                    res_state = "intact"
                    flip_r[i] = True
                    res_retest_count = 0
                # Failed retest: close back below zone low (zone resumes as resistance).
                elif c < z_low:
                    res_state = "intact"
                    failed_retest_r[i] = True
                    res_retest_count = 0

            retest_count_r[i] = res_retest_count

    # Write columns.
    if "zone_break_support" in col_set:
        df["ms_zone_break_support"] = pd.array(break_s, dtype="boolean")  # type: ignore[arg-type]
    if "zone_break_resistance" in col_set:
        df["ms_zone_break_resistance"] = pd.array(break_r, dtype="boolean")  # type: ignore[arg-type]
    if "zone_retest_support" in col_set:
        df["ms_zone_retest_support"] = pd.array(retest_s, dtype="boolean")  # type: ignore[arg-type]
    if "zone_retest_resistance" in col_set:
        df["ms_zone_retest_resistance"] = pd.array(retest_r, dtype="boolean")  # type: ignore[arg-type]
    if "zone_retest_count_support" in col_set:
        df["ms_zone_retest_count_support"] = pd.array(retest_count_s, dtype="Int32")  # type: ignore[arg-type]
    if "zone_retest_count_resistance" in col_set:
        df["ms_zone_retest_count_resistance"] = pd.array(retest_count_r, dtype="Int32")  # type: ignore[arg-type]
    if "zone_flip_support" in col_set:
        df["ms_zone_flip_support"] = pd.array(flip_s, dtype="boolean")  # type: ignore[arg-type]
    if "zone_flip_resistance" in col_set:
        df["ms_zone_flip_resistance"] = pd.array(flip_r, dtype="boolean")  # type: ignore[arg-type]
    if "zone_failed_retest_support" in col_set:
        df["ms_zone_failed_retest_support"] = pd.array(failed_retest_s, dtype="boolean")  # type: ignore[arg-type]
    if "zone_failed_retest_resistance" in col_set:
        df["ms_zone_failed_retest_resistance"] = pd.array(failed_retest_r, dtype="boolean")  # type: ignore[arg-type]


def _compute_snapshots(
    waves: list[Wave],
    columns: tuple[str, ...],
    helper: MarketStructureHelper,
    atr_arr: np.ndarray,
) -> list[dict[str, object]]:
    """Forward pass over confirmed waves, capturing state at each boundary."""
    from market_structure.helper import MarketStructureHelper

    col_set = frozenset(columns)

    tops: list[Wave] = []
    bottoms: list[Wave] = []
    snapshots: list[dict[str, object]] = []
    prev_trending_up = False
    prev_trending_down = False
    trend_epoch_bar: int | None = None

    need_zones = bool(
        {
            "support_zone_low",
            "support_zone_high",
            "support_zone_wick_low",
            "support_zone_wick_high",
            "resistance_zone_low",
            "resistance_zone_high",
            "resistance_zone_wick_low",
            "resistance_zone_wick_high",
            "support_is_double",
            "resistance_is_double",
            "support_overlap_count",
            "resistance_overlap_count",
            "support_zone_anchor_time",
            "resistance_zone_anchor_time",
            "zone_quality_support",
            "zone_quality_resistance",
        }
        & col_set
    )

    for wave_idx, wave in enumerate(waves):
        if wave.side == "up":
            tops.append(wave)
        else:
            bottoms.append(wave)

        snap: dict[str, object] = {}

        last_top = tops[-1] if tops else None
        last_bottom = bottoms[-1] if bottoms else None
        prev_top = tops[-2] if len(tops) >= 2 else None
        prev_bottom = bottoms[-2] if len(bottoms) >= 2 else None

        if "last_top_price" in col_set:
            snap["last_top_price"] = _hco_value(last_top) if last_top else np.nan
        if "last_bottom_price" in col_set:
            snap["last_bottom_price"] = _lco_value(last_bottom) if last_bottom else np.nan

        if "made_higher_high" in col_set:
            snap["made_higher_high"] = (
                _hco_value(last_top) > _hco_value(prev_top) if last_top and prev_top else pd.NA
            )
        if "made_lower_high" in col_set:
            snap["made_lower_high"] = (
                _hco_value(last_top) < _hco_value(prev_top) if last_top and prev_top else pd.NA
            )
        if "made_higher_low" in col_set:
            snap["made_higher_low"] = (
                _lco_value(last_bottom) > _lco_value(prev_bottom)
                if last_bottom and prev_bottom
                else pd.NA
            )
        if "made_lower_low" in col_set:
            snap["made_lower_low"] = (
                _lco_value(last_bottom) < _lco_value(prev_bottom)
                if last_bottom and prev_bottom
                else pd.NA
            )

        if "high_since" in col_set:
            snap["high_since"] = last_top.high_since if last_top else pd.NA
        if "low_since" in col_set:
            snap["low_since"] = last_bottom.low_since if last_bottom else pd.NA

        if "pullback_length" in col_set:
            snap["pullback_length"] = wave.pullback.length if wave.pullback else pd.NA
        if "pullback_correction_factor" in col_set:
            snap["pullback_correction_factor"] = (
                wave.pullback.correction_factor
                if wave.pullback and wave.pullback.correction_factor is not None
                else np.nan
            )

        # Batch 1 new columns
        if "wave_length" in col_set:
            snap["wave_length"] = len(wave.candles)
        if "pullback_breakout_level" in col_set:
            snap["pullback_breakout_level"] = (
                wave.pullback.breakout_level if wave.pullback else np.nan
            )
        if "pullback_price_diff" in col_set:
            snap["pullback_price_diff"] = wave.pullback.price_diff if wave.pullback else np.nan
        if "bearish_divergence" in col_set:
            snap["bearish_divergence"] = (
                MarketStructureHelper.is_diverging(last_top, prev_top)
                if last_top and prev_top and last_top.side == "up"
                else pd.NA
            )
        if "bullish_divergence" in col_set:
            snap["bullish_divergence"] = (
                MarketStructureHelper.is_diverging(last_bottom, prev_bottom)
                if last_bottom and prev_bottom and last_bottom.side == "down"
                else pd.NA
            )
        if "wave_count" in col_set:
            snap["wave_count"] = len(tops) + len(bottoms)

        if "structure_break_level" in col_set or "structure_break_confirmed" in col_set:
            sbl, is_up = _compute_structure_break(last_top, last_bottom, prev_top, prev_bottom)
            snap["structure_break_level"] = sbl
            snap["_structure_break_is_uptrend"] = is_up

        if "wave_amplitude" in col_set or "wave_slope" in col_set:
            # ATR at the wave's formation bar
            formation_bar = wave.formation_bar_index
            atr_at_formation = atr_arr[formation_bar] if formation_bar < len(atr_arr) else np.nan
            amplitude_raw = wave.high.high - wave.low.low
            amplitude = (
                amplitude_raw / atr_at_formation
                if not np.isnan(atr_at_formation) and atr_at_formation > 0
                else np.nan
            )
            if "wave_amplitude" in col_set:
                snap["wave_amplitude"] = amplitude
            if "wave_slope" in col_set:
                wlen = len(wave.candles)
                snap["wave_slope"] = (
                    amplitude / wlen if not np.isnan(amplitude) and wlen > 0 else np.nan
                )

        # Batch 2 — zone columns (query helper with accumulated state)
        if need_zones:
            registry_slice = waves[: wave_idx + 1]
            _snapshot_zones(snap, helper, registry_slice, tops, bottoms, columns, atr_arr)

        # Zone quality — computed from the zone objects captured by _snapshot_zones.
        if "zone_quality_support" in col_set or "zone_quality_resistance" in col_set:
            formation_bar = wave.formation_bar_index
            atr_at_bar = atr_arr[formation_bar] if formation_bar < len(atr_arr) else np.nan

            # Build wave-id → formation_bar lookup for anchor resolution.
            wave_fb_by_id: dict[str, int] = {
                w.id: w.formation_bar_index for w in waves[: wave_idx + 1]
            }

            if "zone_quality_support" in col_set:
                sz_obj = snap.get("_support_zone_obj")
                if sz_obj is not None:
                    anchor_fb = wave_fb_by_id.get(sz_obj.anchor_wave_id, formation_bar)  # type: ignore[union-attr]
                    bars_since = formation_bar - anchor_fb
                    snap["zone_quality_support"] = _score_zone_quality(
                        sz_obj,  # type: ignore[arg-type]
                        atr_at_bar,
                        bars_since,
                    )
                else:
                    snap["zone_quality_support"] = np.nan
            if "zone_quality_resistance" in col_set:
                rz_obj = snap.get("_resistance_zone_obj")
                if rz_obj is not None:
                    anchor_fb = wave_fb_by_id.get(rz_obj.anchor_wave_id, formation_bar)  # type: ignore[union-attr]
                    bars_since = formation_bar - anchor_fb
                    snap["zone_quality_resistance"] = _score_zone_quality(
                        rz_obj,  # type: ignore[arg-type]
                        atr_at_bar,
                        bars_since,
                    )
                else:
                    snap["zone_quality_resistance"] = np.nan

        # Pullback ATR factor
        if "pullback_atr_factor" in col_set:
            if wave.pullback:
                fb = wave.formation_bar_index
                atr_at_fb = atr_arr[fb] if fb < len(atr_arr) else np.nan
                if not np.isnan(atr_at_fb) and atr_at_fb > 0:
                    snap["pullback_atr_factor"] = abs(wave.pullback.price_diff) / atr_at_fb
                else:
                    snap["pullback_atr_factor"] = np.nan
            else:
                snap["pullback_atr_factor"] = np.nan

        # Wave volume and amplitude ratios
        if "wave_volume" in col_set or "wave_volume_ratio" in col_set:
            vol = sum(c.volume for c in wave.candles)
            if "wave_volume" in col_set:
                snap["wave_volume"] = vol
            if "wave_volume_ratio" in col_set:
                # Find previous same-direction wave.
                same_dir = tops if wave.side == "up" else bottoms
                prev_same = same_dir[-2] if len(same_dir) >= 2 else None
                if prev_same is not None:
                    prev_vol = sum(c.volume for c in prev_same.candles)
                    snap["wave_volume_ratio"] = vol / prev_vol if prev_vol > 0 else np.nan
                else:
                    snap["wave_volume_ratio"] = np.nan

        if "wave_amplitude_ratio" in col_set:
            same_dir = tops if wave.side == "up" else bottoms
            prev_same = same_dir[-2] if len(same_dir) >= 2 else None
            snap["wave_amplitude_ratio"] = _compute_wave_amplitude_ratio(wave, prev_same, atr_arr)

        if "trend_wave_count" in col_set:
            snap["trend_wave_count"] = _compute_trend_wave_count(tops, bottoms)

        if "three_push_up" in col_set:
            snap["three_push_up"] = _is_three_push_up(tops)

        if "three_push_down" in col_set:
            snap["three_push_down"] = _is_three_push_down(bottoms)

        # Tier 3 base — always compute if trending columns requested.
        need_trend_base = bool({"is_trending_up", "is_trending_down", "trend_duration"} & col_set)
        if need_trend_base:
            if last_top and last_bottom and prev_top and prev_bottom:
                hh = _hco_value(last_top) > _hco_value(prev_top)
                hl = _lco_value(last_bottom) > _lco_value(prev_bottom)
                snap["_trend_up_base"] = hh and hl
                snap["_trend_up_threshold"] = _lco_value(last_bottom)

                lh = _hco_value(last_top) < _hco_value(prev_top)
                ll = _lco_value(last_bottom) < _lco_value(prev_bottom)
                snap["_trend_down_base"] = lh and ll
                snap["_trend_down_threshold"] = _hco_value(last_top)
            else:
                snap["_trend_up_base"] = False
                snap["_trend_up_threshold"] = np.nan
                snap["_trend_down_base"] = False
                snap["_trend_down_threshold"] = np.nan

        # Trend epoch tracking — record when current trend started.
        if "trend_duration" in col_set:
            is_up = bool(snap.get("_trend_up_base", False))
            is_down = bool(snap.get("_trend_down_base", False))
            is_trending = is_up or is_down
            was_trending = prev_trending_up or prev_trending_down
            if is_trending and not was_trending:
                trend_epoch_bar = wave.formation_bar_index
            elif not is_trending:
                trend_epoch_bar = None
            snap["_trend_epoch_bar"] = trend_epoch_bar
            prev_trending_up = is_up
            prev_trending_down = is_down

        snapshots.append(snap)

    return snapshots


# ------------------------------------------------------------------
# Tier 2 defaults (bars before any confirmed wave)
# ------------------------------------------------------------------

_TIER2_DEFAULTS: dict[str, object] = {
    "last_top_price": np.nan,
    "last_bottom_price": np.nan,
    "made_higher_high": pd.NA,
    "made_higher_low": pd.NA,
    "made_lower_high": pd.NA,
    "made_lower_low": pd.NA,
    "high_since": pd.NA,
    "low_since": pd.NA,
    "pullback_length": pd.NA,
    "pullback_correction_factor": np.nan,
    "wave_length": pd.NA,
    "pullback_breakout_level": np.nan,
    "pullback_price_diff": np.nan,
    "bearish_divergence": pd.NA,
    "bullish_divergence": pd.NA,
    "wave_count": 0,
    "support_zone_low": np.nan,
    "support_zone_high": np.nan,
    "support_zone_wick_low": np.nan,
    "support_zone_wick_high": np.nan,
    "resistance_zone_low": np.nan,
    "resistance_zone_high": np.nan,
    "resistance_zone_wick_low": np.nan,
    "resistance_zone_wick_high": np.nan,
    "support_is_double": pd.NA,
    "resistance_is_double": pd.NA,
    "support_overlap_count": pd.NA,
    "resistance_overlap_count": pd.NA,
    "support_zone_anchor_time": np.nan,
    "resistance_zone_anchor_time": np.nan,
    "structure_break_level": np.nan,
    "wave_amplitude": np.nan,
    "wave_slope": np.nan,
    "zone_quality_support": np.nan,
    "zone_quality_resistance": np.nan,
    "pullback_atr_factor": np.nan,
    "wave_volume": np.nan,
    "wave_volume_ratio": np.nan,
    "wave_amplitude_ratio": np.nan,
    "trend_wave_count": 0,
    "three_push_up": False,
    "three_push_down": False,
}


def _broadcast_tier2(
    df: pd.DataFrame,
    snapshots: list[dict[str, object]],
    wave_starts: np.ndarray,
    wave_ends: np.ndarray,
    n_confirmed: int,
    n_waves: int,
    columns: tuple[str, ...],
) -> None:
    """Assign Tier 2 and Tier 3 base values from snapshots to DataFrame rows."""
    col_set = frozenset(columns)
    n = len(df)
    tier2_cols = [c for c in col_set if c in _TIER2_DEFAULTS]
    trend_cols = [c for c in ("is_trending_up", "is_trending_down") if c in col_set]

    # Dtype map determines how each column is stored and written.
    dtype_map: dict[str, str] = {
        "last_top_price": "float64",
        "last_bottom_price": "float64",
        "made_higher_high": "boolean",
        "made_higher_low": "boolean",
        "made_lower_high": "boolean",
        "made_lower_low": "boolean",
        "high_since": "Int32",
        "low_since": "Int32",
        "pullback_length": "Int32",
        "pullback_correction_factor": "float64",
        "wave_length": "Int32",
        "pullback_breakout_level": "float64",
        "pullback_price_diff": "float64",
        "bearish_divergence": "boolean",
        "bullish_divergence": "boolean",
        "wave_count": "Int32",
        "support_zone_low": "float64",
        "support_zone_high": "float64",
        "support_zone_wick_low": "float64",
        "support_zone_wick_high": "float64",
        "resistance_zone_low": "float64",
        "resistance_zone_high": "float64",
        "resistance_zone_wick_low": "float64",
        "resistance_zone_wick_high": "float64",
        "support_is_double": "boolean",
        "resistance_is_double": "boolean",
        "support_overlap_count": "Int32",
        "resistance_overlap_count": "Int32",
        "support_zone_anchor_time": "float64",
        "resistance_zone_anchor_time": "float64",
        "structure_break_level": "float64",
        "wave_amplitude": "float64",
        "wave_slope": "float64",
        "zone_quality_support": "float64",
        "zone_quality_resistance": "float64",
        "pullback_atr_factor": "float64",
        "wave_volume": "float64",
        "wave_volume_ratio": "float64",
        "wave_amplitude_ratio": "float64",
        "trend_wave_count": "Int32",
        "three_push_up": "boolean",
        "three_push_down": "boolean",
        "is_trending_up": "bool",
        "is_trending_down": "bool",
    }

    # Split columns by storage type: float64 columns use typed numpy arrays
    # (avoiding Python object boxing), others use object arrays for pd.NA support.
    float_cols = [c for c in tier2_cols if dtype_map.get(c) == "float64"]
    other_cols = [c for c in tier2_cols if dtype_map.get(c) != "float64"]

    # Pre-allocate arrays — typed for float64, object for the rest.
    arrays: dict[str, np.ndarray] = {}
    for col in float_cols:
        arrays[col] = np.full(n, np.nan, dtype=np.float64)
    for col in other_cols:
        arrays[col] = np.empty(n, dtype=object)
        arrays[col][:] = _TIER2_DEFAULTS[col]
    for col in trend_cols:
        arrays[col] = np.zeros(n, dtype=bool)

    # Wave 0 bars keep defaults (no confirmed waves yet).
    # Wave j (j >= 1): snapshot[j - 1].
    for j in range(1, n_confirmed):
        s = slice(int(wave_starts[j]), int(wave_ends[j]) + 1)
        snap = snapshots[j - 1]
        for col in tier2_cols:
            if col in snap:
                arrays[col][s] = snap[col]
        for col in trend_cols:
            base_key = f"_trend_{'up' if col == 'is_trending_up' else 'down'}_base"
            if base_key in snap:
                arrays[col][s] = snap[base_key]

    # Forming wave bars: snapshot[last] (if any confirmed waves exist).
    if snapshots:
        last_snap = snapshots[-1]
        forming_s = slice(int(wave_starts[n_waves - 1]), int(wave_ends[n_waves - 1]) + 1)
        for col in tier2_cols:
            if col in last_snap:
                arrays[col][forming_s] = last_snap[col]
        for col in trend_cols:
            base_key = f"_trend_{'up' if col == 'is_trending_up' else 'down'}_base"
            if base_key in last_snap:
                arrays[col][forming_s] = last_snap[base_key]

    # Write to DataFrame — float64 arrays go directly, others via pd.array().
    for col in float_cols:
        df[f"ms_{col}"] = arrays[col]
    for col in other_cols:
        dtype = dtype_map.get(col, "object")
        df[f"ms_{col}"] = pd.array(arrays[col], dtype=dtype)  # type: ignore[arg-type]
    for col in trend_cols:
        dtype = dtype_map.get(col, "bool")
        df[f"ms_{col}"] = arrays[col]

    # Event semantics for three_push: fire only on the first bar of each
    # broadcast span (the wave boundary bar), not for the entire wave.
    for tp_col in ("three_push_up", "three_push_down"):
        if tp_col in col_set and f"ms_{tp_col}" in df.columns:
            raw = df[f"ms_{tp_col}"].to_numpy(dtype=bool)
            df[f"ms_{tp_col}"] = pd.array(_edge_detect(raw), dtype="boolean")  # type: ignore[arg-type]


def _correct_trending_up(
    df: pd.DataFrame,
    snapshots: list[dict[str, object]],
    wave_starts: np.ndarray,
    wave_ends: np.ndarray,
    n_confirmed: int,
    n_waves: int,
    sign: np.ndarray,
    lows: np.ndarray,
) -> None:
    """Apply per-bar correction for ``ms_is_trending_up``.

    When the trend base is True and the wave is a down-wave, the trend
    breaks at the first bar where the running low drops to or below the
    last bottom's LCO level.
    """
    arr = df["ms_is_trending_up"].to_numpy(dtype=bool).copy()

    for j in range(1, n_waves):
        snap_idx = min(j - 1, len(snapshots) - 1)
        snap = snapshots[snap_idx]
        if not snap.get("_trend_up_base", False):
            continue
        ws = int(wave_starts[j])
        we = int(wave_ends[j])
        # Only down-waves can break an uptrend.
        if sign[ws] >= 0:
            continue
        threshold = float(snap["_trend_up_threshold"])  # type: ignore[arg-type]
        if np.isnan(threshold):
            continue
        segment_lows = lows[ws : we + 1]
        running_low = np.minimum.accumulate(segment_lows)
        broken = running_low <= threshold
        if broken.any():
            break_offset = int(np.argmax(broken))
            arr[ws + break_offset : we + 1] = False

    df["ms_is_trending_up"] = arr


def _correct_trending_down(
    df: pd.DataFrame,
    snapshots: list[dict[str, object]],
    wave_starts: np.ndarray,
    wave_ends: np.ndarray,
    n_confirmed: int,
    n_waves: int,
    sign: np.ndarray,
    highs: np.ndarray,
) -> None:
    """Apply per-bar correction for ``ms_is_trending_down``.

    When the trend base is True and the wave is an up-wave, the trend
    breaks at the first bar where the running high reaches or exceeds the
    last top's HCO level.
    """
    arr = df["ms_is_trending_down"].to_numpy(dtype=bool).copy()

    for j in range(1, n_waves):
        snap_idx = min(j - 1, len(snapshots) - 1)
        snap = snapshots[snap_idx]
        if not snap.get("_trend_down_base", False):
            continue
        ws = int(wave_starts[j])
        we = int(wave_ends[j])
        # Only up-waves can break a downtrend.
        if sign[ws] < 0:
            continue
        threshold = float(snap["_trend_down_threshold"])  # type: ignore[arg-type]
        if np.isnan(threshold):
            continue
        segment_highs = highs[ws : we + 1]
        running_high = np.maximum.accumulate(segment_highs)
        broken = running_high >= threshold
        if broken.any():
            break_offset = int(np.argmax(broken))
            arr[ws + break_offset : we + 1] = False

    df["ms_is_trending_down"] = arr


# ------------------------------------------------------------------
# Live projection
# ------------------------------------------------------------------


class _LiveContext:
    """Shared state for live-mode column computation."""

    __slots__ = (
        "atr_arr",
        "current",
        "df",
        "helper",
        "last_bottom",
        "last_top",
        "last_wave",
        "n",
        "prev_bottom",
        "prev_top",
    )

    def __init__(
        self,
        helper: MarketStructureHelper,
        df: pd.DataFrame,
        atr_arr: np.ndarray,
    ) -> None:
        self.helper = helper
        self.df = df
        self.atr_arr = atr_arr
        self.n = len(df)
        self.current = helper.get_current_wave()
        self.last_top = helper.get_last_top()
        self.last_bottom = helper.get_last_bottom()
        self.prev_top = helper.get_previous_top()
        self.prev_bottom = helper.get_previous_bottom()
        self.last_wave = helper._wave_registry[-1] if helper._wave_registry else None

    @property
    def last_atr(self) -> float:
        return self.atr_arr[-1] if len(self.atr_arr) > 0 else np.nan

    @property
    def last_atr_positive(self) -> float:
        v = self.last_atr
        return v if not np.isnan(v) and v > 0 else np.nan


def _live_wave_identity(col: str, ctx: _LiveContext) -> object:
    """Wave identity and price level columns."""
    if col == "wave_side":
        return ctx.current.side if ctx.current else ""
    if col == "wave_id":
        return f"forming-{ctx.helper._next_wave_id}" if ctx.current else ""
    if col == "last_top_price":
        return _hco_value(ctx.last_top) if ctx.last_top else np.nan
    if col == "last_bottom_price":
        return _lco_value(ctx.last_bottom) if ctx.last_bottom else np.nan
    if col == "made_higher_high":
        return (
            _hco_value(ctx.last_top) > _hco_value(ctx.prev_top)
            if ctx.last_top and ctx.prev_top
            else pd.NA
        )
    if col == "made_lower_high":
        return (
            _hco_value(ctx.last_top) < _hco_value(ctx.prev_top)
            if ctx.last_top and ctx.prev_top
            else pd.NA
        )
    if col == "made_higher_low":
        return (
            _lco_value(ctx.last_bottom) > _lco_value(ctx.prev_bottom)
            if ctx.last_bottom and ctx.prev_bottom
            else pd.NA
        )
    if col == "made_lower_low":
        return (
            _lco_value(ctx.last_bottom) < _lco_value(ctx.prev_bottom)
            if ctx.last_bottom and ctx.prev_bottom
            else pd.NA
        )
    if col == "high_since":
        return ctx.last_top.high_since if ctx.last_top else pd.NA
    if col == "low_since":
        return ctx.last_bottom.low_since if ctx.last_bottom else pd.NA
    if col == "forming_wave_high":
        return ctx.current.high.high if ctx.current else np.nan
    if col == "forming_wave_low":
        return ctx.current.low.low if ctx.current else np.nan
    if col == "wave_count":
        return len(ctx.helper._wave_registry)
    return None  # not handled


def _live_wave_metrics(col: str, ctx: _LiveContext) -> object:
    """Wave length, amplitude, slope, volume, and ratio columns."""
    lw = ctx.last_wave
    if col == "wave_length":
        return len(lw.candles) if lw else pd.NA
    if col == "wave_amplitude" or col == "wave_slope":
        if not lw:
            return np.nan
        atr_val = ctx.last_atr_positive
        amp = (lw.high.high - lw.low.low) / atr_val if not np.isnan(atr_val) else np.nan
        if col == "wave_amplitude":
            return amp
        wlen = len(lw.candles)
        return amp / wlen if not np.isnan(amp) and wlen > 0 else np.nan
    if col == "wave_volume":
        return sum(c.volume for c in lw.candles) if lw else np.nan
    if col == "wave_volume_ratio":
        if not lw:
            return np.nan
        cur_vol = sum(c.volume for c in lw.candles)
        same_dir = ctx.helper._top_waves if lw.side == "up" else ctx.helper._bottom_waves
        prev_same = same_dir[-2] if len(same_dir) >= 2 else None
        if prev_same is not None:
            prev_vol = sum(c.volume for c in prev_same.candles)
            return cur_vol / prev_vol if prev_vol > 0 else np.nan
        return np.nan
    if col == "wave_amplitude_ratio":
        if not lw:
            return np.nan
        same_dir = ctx.helper._top_waves if lw.side == "up" else ctx.helper._bottom_waves
        prev_same = same_dir[-2] if len(same_dir) >= 2 else None
        return _compute_wave_amplitude_ratio(lw, prev_same, ctx.atr_arr)
    return None  # not handled


def _live_pullback(col: str, ctx: _LiveContext) -> object:
    """Pullback columns."""
    lw = ctx.last_wave
    pb = lw.pullback if lw else None
    if col == "pullback_length":
        return pb.length if pb else pd.NA
    if col == "pullback_correction_factor":
        return pb.correction_factor if pb and pb.correction_factor is not None else np.nan
    if col == "pullback_breakout_level":
        return pb.breakout_level if pb else np.nan
    if col == "pullback_price_diff":
        return pb.price_diff if pb else np.nan
    if col == "pullback_atr_factor":
        if not pb:
            return np.nan
        atr_val = ctx.last_atr_positive
        return abs(pb.price_diff) / atr_val if not np.isnan(atr_val) else np.nan
    return None  # not handled


def _live_divergence(col: str, ctx: _LiveContext) -> object:
    """Divergence columns."""
    from market_structure.helper import MarketStructureHelper

    if col == "bearish_divergence":
        return (
            MarketStructureHelper.is_diverging(ctx.last_top, ctx.prev_top)
            if ctx.last_top and ctx.prev_top
            else pd.NA
        )
    if col == "bullish_divergence":
        return (
            MarketStructureHelper.is_diverging(ctx.last_bottom, ctx.prev_bottom)
            if ctx.last_bottom and ctx.prev_bottom
            else pd.NA
        )
    return None  # not handled


def _live_trend(col: str, ctx: _LiveContext) -> object:
    """Trend structure columns."""
    if col == "is_trending_up":
        return ctx.helper.is_trending_up()
    if col == "is_trending_down":
        return ctx.helper.is_trending_down()
    if col == "trend_wave_count":
        return _compute_trend_wave_count(
            list(ctx.helper._top_waves), list(ctx.helper._bottom_waves)
        )
    if col == "three_push_up":
        return _is_three_push_up(list(ctx.helper._top_waves))
    if col == "three_push_down":
        return _is_three_push_down(list(ctx.helper._bottom_waves))
    if col == "trend_duration":
        return pd.NA
    if col == "structure_break_level":
        sbl, _ = _compute_structure_break(
            ctx.last_top, ctx.last_bottom, ctx.prev_top, ctx.prev_bottom
        )
        return sbl
    if col == "structure_break_confirmed":
        sbl, is_up = _compute_structure_break(
            ctx.last_top, ctx.last_bottom, ctx.prev_top, ctx.prev_bottom
        )
        if np.isnan(sbl) or ctx.n == 0:
            return False
        last_close = float(ctx.df.iloc[-1]["close"])  # type: ignore[arg-type]
        cur = last_close < sbl if is_up else last_close > sbl
        if not cur:
            return False
        # Edge-detect: suppress if the previous bar also crossed.
        if ctx.n > 1:
            prev_close = float(ctx.df.iloc[-2]["close"])  # type: ignore[arg-type]
            prev = prev_close < sbl if is_up else prev_close > sbl
            return not prev
        return True
    return None  # not handled


def _live_zone(col: str, ctx: _LiveContext) -> object:
    """Support/resistance zone columns (including quality and anchor time)."""
    # Support zone group
    if col in {
        "support_zone_low",
        "support_zone_high",
        "support_zone_wick_low",
        "support_zone_wick_high",
        "support_is_double",
        "support_overlap_count",
        "support_zone_anchor_time",
        "zone_quality_support",
    }:
        zones = ctx.helper.get_support_zones(atr_arr=ctx.atr_arr)
        sz = zones[0] if zones else None
        if col == "support_zone_low":
            return sz.range[0] if sz else np.nan
        if col == "support_zone_high":
            return sz.range[1] if sz else np.nan
        if col == "support_zone_wick_low":
            return sz.wick_range[0] if sz else np.nan
        if col == "support_zone_wick_high":
            return sz.wick_range[1] if sz else np.nan
        if col == "support_is_double":
            return sz.is_double if sz else pd.NA
        if col == "support_overlap_count":
            return len(sz.overlapping_low_wave_ids) if sz else pd.NA
        if col == "support_zone_anchor_time":
            return _zone_anchor_time(sz, ctx.helper._wave_registry) if sz else np.nan
        # zone_quality_support
        if not sz:
            return np.nan
        anchor_w = next((w for w in ctx.helper._wave_registry if w.id == sz.anchor_wave_id), None)
        bars_since = (ctx.n - 1 - anchor_w.formation_bar_index) if anchor_w else 0
        return _score_zone_quality(sz, ctx.last_atr, bars_since)

    # Resistance zone group
    if col in {
        "resistance_zone_low",
        "resistance_zone_high",
        "resistance_zone_wick_low",
        "resistance_zone_wick_high",
        "resistance_is_double",
        "resistance_overlap_count",
        "resistance_zone_anchor_time",
        "zone_quality_resistance",
    }:
        zones = ctx.helper.get_resistance_zones(atr_arr=ctx.atr_arr)
        rz = zones[0] if zones else None
        if col == "resistance_zone_low":
            return rz.range[0] if rz else np.nan
        if col == "resistance_zone_high":
            return rz.range[1] if rz else np.nan
        if col == "resistance_zone_wick_low":
            return rz.wick_range[0] if rz else np.nan
        if col == "resistance_zone_wick_high":
            return rz.wick_range[1] if rz else np.nan
        if col == "resistance_is_double":
            return rz.is_double if rz else pd.NA
        if col == "resistance_overlap_count":
            return len(rz.overlapping_high_wave_ids) if rz else pd.NA
        if col == "resistance_zone_anchor_time":
            return _zone_anchor_time(rz, ctx.helper._wave_registry) if rz else np.nan
        # zone_quality_resistance
        if not rz:
            return np.nan
        anchor_w = next((w for w in ctx.helper._wave_registry if w.id == rz.anchor_wave_id), None)
        bars_since = (ctx.n - 1 - anchor_w.formation_bar_index) if anchor_w else 0
        return _score_zone_quality(rz, ctx.last_atr, bars_since)

    # Zone lifecycle events — not computable from single-bar live state.
    if col in {
        "zone_break_support",
        "zone_break_resistance",
        "zone_retest_support",
        "zone_retest_resistance",
        "zone_flip_support",
        "zone_flip_resistance",
        "zone_failed_retest_support",
        "zone_failed_retest_resistance",
    }:
        return False
    if col in {"zone_retest_count_support", "zone_retest_count_resistance"}:
        return 0

    return None  # not handled


def _live_volatility_distance(col: str, ctx: _LiveContext) -> object:
    """ATR, distance, bars-since, and SFP columns."""
    if col == "atr":
        return ctx.last_atr
    if col == "distance_to_support":
        zones = ctx.helper.get_support_zones(atr_arr=ctx.atr_arr)
        sz = zones[0] if zones else None
        last_close = float(ctx.df.iloc[-1]["close"]) if ctx.n > 0 else np.nan  # type: ignore[arg-type]
        atr_val = ctx.last_atr_positive
        if sz and not np.isnan(atr_val):
            return (last_close - sz.range[1]) / atr_val
        return np.nan
    if col == "distance_to_resistance":
        zones = ctx.helper.get_resistance_zones(atr_arr=ctx.atr_arr)
        rz = zones[0] if zones else None
        last_close = float(ctx.df.iloc[-1]["close"]) if ctx.n > 0 else np.nan  # type: ignore[arg-type]
        atr_val = ctx.last_atr_positive
        if rz and not np.isnan(atr_val):
            return (rz.range[0] - last_close) / atr_val
        return np.nan
    if col == "bars_since_last_top":
        return ctx.n - 1 - ctx.last_top.formation_bar_index if ctx.last_top else pd.NA
    if col == "bars_since_last_bottom":
        return ctx.n - 1 - ctx.last_bottom.formation_bar_index if ctx.last_bottom else pd.NA
    if col == "sfp_high":
        if ctx.last_top and ctx.n > 1:
            ltp_val = _hco_value(ctx.last_top)
            last_row = ctx.df.iloc[-1]
            cur = float(last_row["high"]) > ltp_val and float(last_row["close"]) < ltp_val  # type: ignore[arg-type]
            if not cur:
                return False
            # Edge-detect: suppress if the previous bar also fired SFP against same level.
            prev_row = ctx.df.iloc[-2]
            prev = float(prev_row["high"]) > ltp_val and float(prev_row["close"]) < ltp_val  # type: ignore[arg-type]
            return not prev
        if ctx.last_top and ctx.n == 1:
            ltp_val = _hco_value(ctx.last_top)
            last_row = ctx.df.iloc[-1]
            return float(last_row["high"]) > ltp_val and float(last_row["close"]) < ltp_val  # type: ignore[arg-type]
        return False
    if col == "sfp_low":
        if ctx.last_bottom and ctx.n > 1:
            lbp_val = _lco_value(ctx.last_bottom)
            last_row = ctx.df.iloc[-1]
            cur = float(last_row["low"]) < lbp_val and float(last_row["close"]) > lbp_val  # type: ignore[arg-type]
            if not cur:
                return False
            prev_row = ctx.df.iloc[-2]
            prev = float(prev_row["low"]) < lbp_val and float(prev_row["close"]) > lbp_val  # type: ignore[arg-type]
            return not prev
        if ctx.last_bottom and ctx.n == 1:
            lbp_val = _lco_value(ctx.last_bottom)
            last_row = ctx.df.iloc[-1]
            return float(last_row["low"]) < lbp_val and float(last_row["close"]) > lbp_val  # type: ignore[arg-type]
        return False
    return None  # not handled


# Dispatch table: each group function returns a value or None if the column
# doesn't belong to that group.  Order matters only for performance (most
# common columns first); correctness is independent of order.
_LIVE_GROUPS = (
    _live_wave_identity,
    _live_trend,
    _live_zone,
    _live_volatility_distance,
    _live_wave_metrics,
    _live_pullback,
    _live_divergence,
)


def _project_live(
    df: pd.DataFrame,
    helper: MarketStructureHelper,
    columns: tuple[str, ...],
    atr_arr: np.ndarray,
) -> None:
    """Fill all rows with current helper state for requested columns.

    In live mode only the last row matters for strategy decisions.
    Filling all rows with current state is fast and sufficient.
    """
    if len(df) == 0:
        return

    ctx = _LiveContext(helper, df, atr_arr)

    for col in columns:
        for group_fn in _LIVE_GROUPS:
            value = group_fn(col, ctx)
            if value is not None:
                df[f"ms_{col}"] = value
                break


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------


def attach_market_structure(
    df: pd.DataFrame,
    metadata: dict[str, str],
    store: dict[str, MarketStructureHelper],
    *,
    hist_col: str = "tsi_hist",
    columns: tuple[str, ...] | None = None,
    max_waves: int = 200,
    atr_period: int = 14,
    retest_mode: str = "wick",
    auto_histogram: bool = False,
    tsi_r: int = 12,
    tsi_s: int = 8,
    tsi_signal_period: int = 4,
) -> tuple[pd.DataFrame, MarketStructureHelper]:
    """Project market-structure state onto DataFrame columns.

    On the first call for a pair (empty *store*), hydrates the full
    DataFrame and projects per-bar columns via a post-hoc forward pass.
    On subsequent calls (live mode), registers only the newest candle
    and fills columns from current helper state.

    Args:
        df: OHLCV DataFrame with a histogram column.  Mutated in-place.
        metadata: Must contain ``"pair"`` key.
        store: Caller-owned ``dict[str, MarketStructureHelper]``.
        hist_col: Histogram column name.  Ignored when *auto_histogram*
            is True.
        columns: Which columns to project (short names, no ``ms_``
            prefix).  ``None`` means all available columns.
        max_waves: Maximum confirmed waves to retain after projection.
        atr_period: Wilder's ATR lookback period.  Only computed when
            at least one ATR-dependent column is requested.
        retest_mode: ``"wick"`` (default) or ``"close"`` — how zone
            retests are detected.  ``"wick"`` triggers on any wick
            touching the zone; ``"close"`` requires a candle close
            inside the zone.
        auto_histogram: When True, compute TSI from ``df["close"]``
            and use the resulting histogram as the structural input.
        tsi_r: TSI long smoothing period (first EMA).
        tsi_s: TSI short smoothing period (second EMA).
        tsi_signal_period: TSI signal line EMA period.

    Returns:
        ``(df, helper)`` — the same DataFrame with ``ms_*`` columns
        added, and the helper instance (also stored in *store*).

    Raises:
        ValueError: Unknown column name in *columns*.
        MarketStructureDesyncError: Live-mode DataFrame is older than
            the helper's last registered candle.
    """
    validated = _validate_columns(columns)
    pair = metadata["pair"]
    helper = store.get(pair)

    # Auto-histogram: compute TSI and use its histogram as structural input.
    if auto_histogram:
        from market_structure.tsi import compute_tsi

        tsi_df = compute_tsi(df["close"], r=tsi_r, s=tsi_s, signal_period=tsi_signal_period)  # type: ignore[arg-type]
        hist_col = "_ms_auto_tsi_histogram"
        df[hist_col] = tsi_df["tsi_histogram"]

    # Freqtrade provides 'date' (datetime); hydrate expects 'open_time' (epoch ms).
    if "open_time" not in df.columns and "date" in df.columns:
        df["open_time"] = df["date"].astype("int64") // 10**6

    if helper is None:
        # First call — hydrate full frame with eviction disabled.
        df = df.reset_index(drop=True)
        safe_max = max(max_waves, len(df) // 2 + 10)
        helper = hydrate(df, histogram_key=hist_col, max_waves=safe_max)

        atr_arr = _compute_atr(
            df["high"].to_numpy(dtype=float),
            df["low"].to_numpy(dtype=float),
            df["close"].to_numpy(dtype=float),
            period=atr_period,
        )

        _project_backtest(df, helper, hist_col, validated, atr_arr, retest_mode)

        # Restore the requested max_waves and trim the registry.
        helper.max_waves = max_waves
        helper._evict_old_waves()
        store[pair] = helper
    else:
        # Subsequent call — live incremental path.
        if len(df) > 0:
            last_ot = int(df.iloc[-1]["open_time"])  # type: ignore[arg-type]
            if (
                helper._last_registered_open_time is not None
                and last_ot < helper._last_registered_open_time
            ):
                msg = (
                    f"DataFrame last open_time ({last_ot}) is older than "
                    f"helper's last registered ({helper._last_registered_open_time}). "
                    f"Delete the pair from the store to force rehydration."
                )
                raise MarketStructureDesyncError(msg)

            last_row = df.iloc[-1]
            helper.register_candle(
                Candle(
                    open_time=int(last_row["open_time"]),  # type: ignore[arg-type]
                    open=float(last_row["open"]),  # type: ignore[arg-type]
                    high=float(last_row["high"]),  # type: ignore[arg-type]
                    low=float(last_row["low"]),  # type: ignore[arg-type]
                    close=float(last_row["close"]),  # type: ignore[arg-type]
                    volume=float(last_row["volume"]),  # type: ignore[arg-type]
                    histogram_value=float(last_row[hist_col]),  # type: ignore[arg-type]
                )
            )

        atr_arr = _compute_atr(
            df["high"].to_numpy(dtype=float),
            df["low"].to_numpy(dtype=float),
            df["close"].to_numpy(dtype=float),
            period=atr_period,
        )

        _project_live(df, helper, validated, atr_arr)

    return df, helper
