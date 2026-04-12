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

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

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
        # Tier 1 — wave-constant
        "wave_side",
        "wave_id",
        # Tier 2 — wave-boundary
        "last_top_price",
        "last_bottom_price",
        "made_higher_high",
        "made_higher_low",
        "made_lower_high",
        "made_lower_low",
        "high_since",
        "low_since",
        "pullback_length",
        "pullback_correction_factor",
        "wave_length",
        "pullback_breakout_level",
        "pullback_price_diff",
        "bearish_divergence",
        "bullish_divergence",
        "wave_count",
        "support_zone_low",
        "support_zone_high",
        "resistance_zone_low",
        "resistance_zone_high",
        "support_is_double",
        "resistance_is_double",
        "support_overlap_count",
        "resistance_overlap_count",
        # Tier 3 — per-bar
        "is_trending_up",
        "is_trending_down",
        "forming_wave_high",
        "forming_wave_low",
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
# Value extractors (mirror helper._hco_value / _lco_value)
# ------------------------------------------------------------------


def _hco_value(wave: Wave) -> float:
    c = wave.highest_close_or_open
    return max(c.close, c.open)


def _lco_value(wave: Wave) -> float:
    c = wave.lowest_close_or_open
    return min(c.close, c.open)


# ------------------------------------------------------------------
# Zone snapshot helper
# ------------------------------------------------------------------


def _snapshot_zones(
    snap: dict[str, object],
    helper: MarketStructureHelper,
    registry_slice: list[Wave],
    tops: list[Wave],
    bottoms: list[Wave],
    columns: tuple[str, ...],
) -> None:
    """Capture nearest support/resistance zone state into *snap*.

    Temporarily swaps the helper's internal wave lists to match the
    accumulated state at this snapshot point, queries zone methods
    (reusing the existing zone logic exactly), then restores the
    originals.  This guarantees parity with the brute-force reference
    where ``get_support_zones`` / ``get_resistance_zones`` see only
    the waves confirmed so far.
    """
    need_support = bool(
        {"support_zone_low", "support_zone_high", "support_is_double", "support_overlap_count"}
        & set(columns)
    )
    need_resistance = bool(
        {
            "resistance_zone_low",
            "resistance_zone_high",
            "resistance_is_double",
            "resistance_overlap_count",
        }
        & set(columns)
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
            zones = helper.get_support_zones()
            sz = zones[0] if zones else None
        if need_resistance:
            zones = helper.get_resistance_zones()
            rz = zones[0] if zones else None
    finally:
        # Restore original helper state.
        helper._wave_registry = orig_registry
        helper._top_waves = orig_tops
        helper._bottom_waves = orig_bottoms
        helper._zone_cache_support = orig_cache_s
        helper._zone_cache_resistance = orig_cache_r

    if "support_zone_low" in columns:
        snap["support_zone_low"] = sz.range[0] if sz else np.nan
    if "support_zone_high" in columns:
        snap["support_zone_high"] = sz.range[1] if sz else np.nan
    if "support_is_double" in columns:
        snap["support_is_double"] = sz.is_double if sz else pd.NA
    if "support_overlap_count" in columns:
        snap["support_overlap_count"] = len(sz.overlapping_low_wave_ids) if sz else pd.NA

    if "resistance_zone_low" in columns:
        snap["resistance_zone_low"] = rz.range[0] if rz else np.nan
    if "resistance_zone_high" in columns:
        snap["resistance_zone_high"] = rz.range[1] if rz else np.nan
    if "resistance_is_double" in columns:
        snap["resistance_is_double"] = rz.is_double if rz else pd.NA
    if "resistance_overlap_count" in columns:
        snap["resistance_overlap_count"] = len(rz.overlapping_high_wave_ids) if rz else pd.NA


# ------------------------------------------------------------------
# Backtest projection
# ------------------------------------------------------------------


def _project_backtest(
    df: pd.DataFrame,
    helper: MarketStructureHelper,
    hist_col: str,
    columns: tuple[str, ...],
) -> None:
    """Project requested columns onto *df* using the full wave registry.

    Mutates *df* in-place.  Assumes *helper* was hydrated with eviction
    disabled so ``_wave_registry`` contains every confirmed wave.
    """
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
    need_wave_side = "wave_side" in columns
    need_wave_id = "wave_id" in columns

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
    snapshots = _compute_snapshots(waves, columns, helper)

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

    if "is_trending_up" in columns:
        _correct_trending_up(
            df, snapshots, wave_starts, wave_ends, n_confirmed, n_waves, sign, lows
        )

    if "is_trending_down" in columns:
        _correct_trending_down(
            df, snapshots, wave_starts, wave_ends, n_confirmed, n_waves, sign, highs
        )

    # ------------------------------------------------------------------
    # Tier 3 — forming wave extremes (per-bar running high/low)
    # ------------------------------------------------------------------
    if "forming_wave_high" in columns:
        arr = np.full(n, np.nan)
        for wi in range(n_waves):
            ws = int(wave_starts[wi])
            we = int(wave_ends[wi])
            arr[ws : we + 1] = np.maximum.accumulate(highs[ws : we + 1])
        df["ms_forming_wave_high"] = arr

    if "forming_wave_low" in columns:
        arr = np.full(n, np.nan)
        for wi in range(n_waves):
            ws = int(wave_starts[wi])
            we = int(wave_ends[wi])
            arr[ws : we + 1] = np.minimum.accumulate(lows[ws : we + 1])
        df["ms_forming_wave_low"] = arr


def _compute_snapshots(
    waves: list[Wave],
    columns: tuple[str, ...],
    helper: MarketStructureHelper,
) -> list[dict[str, object]]:
    """Forward pass over confirmed waves, capturing state at each boundary."""
    from market_structure.helper import MarketStructureHelper

    tops: list[Wave] = []
    bottoms: list[Wave] = []
    snapshots: list[dict[str, object]] = []

    need_zones = bool(
        {
            "support_zone_low",
            "support_zone_high",
            "resistance_zone_low",
            "resistance_zone_high",
            "support_is_double",
            "resistance_is_double",
            "support_overlap_count",
            "resistance_overlap_count",
        }
        & set(columns)
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

        if "last_top_price" in columns:
            snap["last_top_price"] = _hco_value(last_top) if last_top else np.nan
        if "last_bottom_price" in columns:
            snap["last_bottom_price"] = _lco_value(last_bottom) if last_bottom else np.nan

        if "made_higher_high" in columns:
            snap["made_higher_high"] = (
                _hco_value(last_top) > _hco_value(prev_top) if last_top and prev_top else pd.NA
            )
        if "made_lower_high" in columns:
            snap["made_lower_high"] = (
                _hco_value(last_top) < _hco_value(prev_top) if last_top and prev_top else pd.NA
            )
        if "made_higher_low" in columns:
            snap["made_higher_low"] = (
                _lco_value(last_bottom) > _lco_value(prev_bottom)
                if last_bottom and prev_bottom
                else pd.NA
            )
        if "made_lower_low" in columns:
            snap["made_lower_low"] = (
                _lco_value(last_bottom) < _lco_value(prev_bottom)
                if last_bottom and prev_bottom
                else pd.NA
            )

        if "high_since" in columns:
            snap["high_since"] = last_top.high_since if last_top else pd.NA
        if "low_since" in columns:
            snap["low_since"] = last_bottom.low_since if last_bottom else pd.NA

        if "pullback_length" in columns:
            snap["pullback_length"] = wave.pullback.length if wave.pullback else pd.NA
        if "pullback_correction_factor" in columns:
            snap["pullback_correction_factor"] = (
                wave.pullback.correction_factor
                if wave.pullback and wave.pullback.correction_factor is not None
                else np.nan
            )

        # Batch 1 new columns
        if "wave_length" in columns:
            snap["wave_length"] = len(wave.candles)
        if "pullback_breakout_level" in columns:
            snap["pullback_breakout_level"] = (
                wave.pullback.breakout_level if wave.pullback else np.nan
            )
        if "pullback_price_diff" in columns:
            snap["pullback_price_diff"] = wave.pullback.price_diff if wave.pullback else np.nan
        if "bearish_divergence" in columns:
            snap["bearish_divergence"] = (
                MarketStructureHelper.is_diverging(last_top, prev_top)
                if last_top and prev_top and last_top.side == "up"
                else pd.NA
            )
        if "bullish_divergence" in columns:
            snap["bullish_divergence"] = (
                MarketStructureHelper.is_diverging(last_bottom, prev_bottom)
                if last_bottom and prev_bottom and last_bottom.side == "down"
                else pd.NA
            )
        if "wave_count" in columns:
            snap["wave_count"] = len(tops) + len(bottoms)

        # Batch 2 — zone columns (query helper with accumulated state)
        if need_zones:
            registry_slice = waves[: wave_idx + 1]
            _snapshot_zones(snap, helper, registry_slice, list(tops), list(bottoms), columns)

        # Tier 3 base — always compute if trending columns requested.
        if "is_trending_up" in columns or "is_trending_down" in columns:
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
    "resistance_zone_low": np.nan,
    "resistance_zone_high": np.nan,
    "support_is_double": pd.NA,
    "resistance_is_double": pd.NA,
    "support_overlap_count": pd.NA,
    "resistance_overlap_count": pd.NA,
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
    n = len(df)
    tier2_cols = [c for c in columns if c in _TIER2_DEFAULTS]
    trend_cols = [c for c in ("is_trending_up", "is_trending_down") if c in columns]

    # Pre-allocate arrays.
    arrays: dict[str, np.ndarray] = {}
    for col in tier2_cols:
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

    # Write to DataFrame with appropriate dtypes.
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
        "resistance_zone_low": "float64",
        "resistance_zone_high": "float64",
        "support_is_double": "boolean",
        "resistance_is_double": "boolean",
        "support_overlap_count": "Int32",
        "resistance_overlap_count": "Int32",
        "is_trending_up": "bool",
        "is_trending_down": "bool",
    }

    for col in tier2_cols + trend_cols:
        dtype = dtype_map.get(col, "object")
        df[f"ms_{col}"] = pd.array(arrays[col], dtype=dtype)  # type: ignore[arg-type]


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


def _project_live(
    df: pd.DataFrame,
    helper: MarketStructureHelper,
    columns: tuple[str, ...],
) -> None:
    """Fill all rows with current helper state for requested columns.

    In live mode only the last row matters for strategy decisions.
    Filling all rows with current state is fast and sufficient.
    """
    n = len(df)
    if n == 0:
        return

    from market_structure.helper import MarketStructureHelper

    current = helper.get_current_wave()
    last_top = helper.get_last_top()
    last_bottom = helper.get_last_bottom()
    prev_top = helper.get_previous_top()
    prev_bottom = helper.get_previous_bottom()
    last_wave = helper._wave_registry[-1] if helper._wave_registry else None

    for col in columns:
        value: object
        if col == "wave_side":
            value = current.side if current else ""
        elif col == "wave_id":
            value = f"forming-{helper._next_wave_id}" if current else ""
        elif col == "last_top_price":
            value = _hco_value(last_top) if last_top else np.nan
        elif col == "last_bottom_price":
            value = _lco_value(last_bottom) if last_bottom else np.nan
        elif col == "made_higher_high":
            value = _hco_value(last_top) > _hco_value(prev_top) if last_top and prev_top else pd.NA
        elif col == "made_lower_high":
            value = _hco_value(last_top) < _hco_value(prev_top) if last_top and prev_top else pd.NA
        elif col == "made_higher_low":
            value = (
                _lco_value(last_bottom) > _lco_value(prev_bottom)
                if last_bottom and prev_bottom
                else pd.NA
            )
        elif col == "made_lower_low":
            value = (
                _lco_value(last_bottom) < _lco_value(prev_bottom)
                if last_bottom and prev_bottom
                else pd.NA
            )
        elif col == "high_since":
            value = last_top.high_since if last_top else pd.NA
        elif col == "low_since":
            value = last_bottom.low_since if last_bottom else pd.NA
        elif col == "pullback_length":
            value = last_wave.pullback.length if last_wave and last_wave.pullback else pd.NA
        elif col == "pullback_correction_factor":
            value = (
                last_wave.pullback.correction_factor
                if last_wave
                and last_wave.pullback
                and last_wave.pullback.correction_factor is not None
                else np.nan
            )
        elif col == "wave_length":
            value = len(last_wave.candles) if last_wave else pd.NA
        elif col == "pullback_breakout_level":
            value = (
                last_wave.pullback.breakout_level if last_wave and last_wave.pullback else np.nan
            )
        elif col == "pullback_price_diff":
            value = last_wave.pullback.price_diff if last_wave and last_wave.pullback else np.nan
        elif col == "bearish_divergence":
            value = (
                MarketStructureHelper.is_diverging(last_top, prev_top)
                if last_top and prev_top
                else pd.NA
            )
        elif col == "bullish_divergence":
            value = (
                MarketStructureHelper.is_diverging(last_bottom, prev_bottom)
                if last_bottom and prev_bottom
                else pd.NA
            )
        elif col == "wave_count":
            value = len(helper._wave_registry)
        elif col in {
            "support_zone_low",
            "support_zone_high",
            "support_is_double",
            "support_overlap_count",
        }:
            zones = helper.get_support_zones()
            sz = zones[0] if zones else None
            if col == "support_zone_low":
                value = sz.range[0] if sz else np.nan
            elif col == "support_zone_high":
                value = sz.range[1] if sz else np.nan
            elif col == "support_is_double":
                value = sz.is_double if sz else pd.NA
            else:
                value = len(sz.overlapping_low_wave_ids) if sz else pd.NA
        elif col in {
            "resistance_zone_low",
            "resistance_zone_high",
            "resistance_is_double",
            "resistance_overlap_count",
        }:
            zones = helper.get_resistance_zones()
            rz = zones[0] if zones else None
            if col == "resistance_zone_low":
                value = rz.range[0] if rz else np.nan
            elif col == "resistance_zone_high":
                value = rz.range[1] if rz else np.nan
            elif col == "resistance_is_double":
                value = rz.is_double if rz else pd.NA
            else:
                value = len(rz.overlapping_high_wave_ids) if rz else pd.NA
        elif col == "forming_wave_high":
            value = current.high.high if current else np.nan
        elif col == "forming_wave_low":
            value = current.low.low if current else np.nan
        elif col == "is_trending_up":
            value = helper.is_trending_up()
        elif col == "is_trending_down":
            value = helper.is_trending_down()
        else:
            continue  # validated earlier, should never happen

        df[f"ms_{col}"] = value


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
        hist_col: Histogram column name.
        columns: Which columns to project (short names, no ``ms_``
            prefix).  ``None`` means all available columns.
        max_waves: Maximum confirmed waves to retain after projection.

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

    # Freqtrade provides 'date' (datetime); hydrate expects 'open_time' (epoch ms).
    if "open_time" not in df.columns and "date" in df.columns:
        df["open_time"] = df["date"].astype("int64") // 10**6

    if helper is None:
        # First call — hydrate full frame with eviction disabled.
        df = df.reset_index(drop=True)
        safe_max = max(max_waves, len(df) // 2 + 10)
        helper = hydrate(df, histogram_key=hist_col, max_waves=safe_max)

        _project_backtest(df, helper, hist_col, validated)

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

        _project_live(df, helper, validated)

    return df, helper
