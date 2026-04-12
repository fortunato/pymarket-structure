"""Stage 12 tests for the Freqtrade wrapper (attach_market_structure).

The keystone test is ``TestPerBarParity``: a brute-force ``register_candle``
loop captures helper state at every bar, then ``attach_market_structure``
projects columns over the same data.  The two must match exactly.
"""

# pyright: reportPrivateUsage=false

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from market_structure import MarketStructureHelper
from market_structure.freqtrade import (
    VALID_COLUMNS,
    MarketStructureDesyncError,
    attach_market_structure,
)
from market_structure.types import Candle

# ---------------------------------------------------------------------------
# Fixture loading (same as test_parity.py / test_zones.py)
# ---------------------------------------------------------------------------

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "ms-LTCUSDT-4h.json"
HISTOGRAM_KEY = "tsi_histogram"


def _load_raw() -> list[dict[str, object]]:
    with FIXTURE_PATH.open() as f:
        return json.load(f)


def _to_dataframe(rows: list[dict[str, object]]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    df["open_time"] = (
        pd.to_datetime(df["openTime"]).dt.tz_localize(None).astype("datetime64[ms]").astype("int64")
    )
    cols = ["open_time", "open", "high", "low", "close", "volume", HISTOGRAM_KEY]
    return pd.DataFrame(df[cols])


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


# ---------------------------------------------------------------------------
# Synthetic helpers
# ---------------------------------------------------------------------------


def _synthetic_df() -> pd.DataFrame:
    """7-bar synthetic frame with 3 confirmed waves + 1 forming."""
    return pd.DataFrame(
        {
            "open_time": [1000, 2000, 3000, 4000, 5000, 6000, 7000],
            "open": [100.0, 102.0, 107.0, 104.0, 99.0, 100.0, 106.0],
            "high": [105.0, 110.0, 109.0, 106.0, 103.0, 108.0, 107.0],
            "low": [98.0, 100.0, 101.0, 97.0, 96.0, 99.0, 95.0],
            "close": [103.0, 108.0, 106.0, 98.0, 101.0, 107.0, 96.0],
            "volume": [1.0] * 7,
            "tsi_hist": [0.4, 0.2, -0.1, -0.3, 0.2, 0.5, -0.4],
        }
    )


# ---------------------------------------------------------------------------
# Per-bar parity (the keystone test)
# ---------------------------------------------------------------------------


def _build_reference(raw: list[dict[str, object]]) -> pd.DataFrame:
    """Brute-force register_candle loop, capturing state at every bar."""
    h = MarketStructureHelper(histogram_key=HISTOGRAM_KEY)
    records: list[dict[str, object]] = []

    for row in raw:
        h.register_candle(_make_candle(row))

        current = h.get_current_wave()
        last_top = h.get_last_top()
        last_bottom = h.get_last_bottom()
        prev_top = h.get_previous_top()
        prev_bottom = h.get_previous_bottom()
        last_confirmed = h._wave_registry[-1] if h._wave_registry else None

        records.append(
            {
                "wave_side": current.side if current else "",
                "is_trending_up": h.is_trending_up(),
                "is_trending_down": h.is_trending_down(),
                "last_top_price": (
                    max(last_top.highest_close_or_open.close, last_top.highest_close_or_open.open)
                    if last_top
                    else np.nan
                ),
                "last_bottom_price": (
                    min(
                        last_bottom.lowest_close_or_open.close,
                        last_bottom.lowest_close_or_open.open,
                    )
                    if last_bottom
                    else np.nan
                ),
                "made_higher_high": (
                    MarketStructureHelper.made_higher_high(last_top, prev_top)
                    if last_top and prev_top
                    else pd.NA
                ),
                "made_lower_high": (
                    MarketStructureHelper.made_lower_high(last_top, prev_top)
                    if last_top and prev_top
                    else pd.NA
                ),
                "made_higher_low": (
                    MarketStructureHelper.made_higher_low(last_bottom, prev_bottom)
                    if last_bottom and prev_bottom
                    else pd.NA
                ),
                "made_lower_low": (
                    MarketStructureHelper.made_lower_low(last_bottom, prev_bottom)
                    if last_bottom and prev_bottom
                    else pd.NA
                ),
                "high_since": last_top.high_since if last_top else pd.NA,
                "low_since": last_bottom.low_since if last_bottom else pd.NA,
                "pullback_length": (
                    last_confirmed.pullback.length
                    if last_confirmed and last_confirmed.pullback
                    else pd.NA
                ),
                "pullback_correction_factor": (
                    last_confirmed.pullback.correction_factor
                    if last_confirmed
                    and last_confirmed.pullback
                    and last_confirmed.pullback.correction_factor is not None
                    else np.nan
                ),
                "wave_length": (len(last_confirmed.candles) if last_confirmed else pd.NA),
                "pullback_breakout_level": (
                    last_confirmed.pullback.breakout_level
                    if last_confirmed and last_confirmed.pullback
                    else np.nan
                ),
                "pullback_price_diff": (
                    last_confirmed.pullback.price_diff
                    if last_confirmed and last_confirmed.pullback
                    else np.nan
                ),
                "bearish_divergence": (
                    MarketStructureHelper.is_diverging(last_top, prev_top)
                    if last_top and prev_top
                    else pd.NA
                ),
                "bullish_divergence": (
                    MarketStructureHelper.is_diverging(last_bottom, prev_bottom)
                    if last_bottom and prev_bottom
                    else pd.NA
                ),
                "wave_count": len(h._wave_registry),
                "forming_wave_high": current.high.high if current else np.nan,
                "forming_wave_low": current.low.low if current else np.nan,
            }
        )

        # Zone columns — query helper after each bar (brute-force reference).
        sup_zones = h.get_support_zones()
        sz = sup_zones[0] if sup_zones else None
        res_zones = h.get_resistance_zones()
        rz = res_zones[0] if res_zones else None
        records[-1].update(
            {
                "support_zone_low": sz.range[0] if sz else np.nan,
                "support_zone_high": sz.range[1] if sz else np.nan,
                "support_is_double": sz.is_double if sz else pd.NA,
                "support_overlap_count": len(sz.overlapping_low_wave_ids) if sz else pd.NA,
                "resistance_zone_low": rz.range[0] if rz else np.nan,
                "resistance_zone_high": rz.range[1] if rz else np.nan,
                "resistance_is_double": rz.is_double if rz else pd.NA,
                "resistance_overlap_count": (len(rz.overlapping_high_wave_ids) if rz else pd.NA),
            }
        )

    return pd.DataFrame(records)


@pytest.fixture()
def parity_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build both brute-force reference and projected DataFrames."""
    raw = _load_raw()
    df = _to_dataframe(raw)

    reference = _build_reference(raw)

    store: dict[str, MarketStructureHelper] = {}
    projected, _ = attach_market_structure(
        df,
        {"pair": "LTC/USDT"},
        store,
        hist_col=HISTOGRAM_KEY,
    )
    return reference, projected


class TestPerBarParity:
    """Brute-force per-bar reference vs projected columns.

    Each test validates one column for clear failure diagnostics.
    """

    def test_wave_side(self, parity_data: tuple[pd.DataFrame, pd.DataFrame]) -> None:
        ref, proj = parity_data
        assert list(ref["wave_side"]) == list(proj["ms_wave_side"])

    def test_is_trending_up(self, parity_data: tuple[pd.DataFrame, pd.DataFrame]) -> None:
        ref, proj = parity_data
        assert list(ref["is_trending_up"]) == list(proj["ms_is_trending_up"])

    def test_is_trending_down(self, parity_data: tuple[pd.DataFrame, pd.DataFrame]) -> None:
        ref, proj = parity_data
        assert list(ref["is_trending_down"]) == list(proj["ms_is_trending_down"])

    def test_last_top_price(self, parity_data: tuple[pd.DataFrame, pd.DataFrame]) -> None:
        ref, proj = parity_data
        ref_vals = ref["last_top_price"].to_numpy(dtype=float)
        proj_vals = proj["ms_last_top_price"].to_numpy(dtype=float)
        np.testing.assert_array_equal(ref_vals, proj_vals)

    def test_last_bottom_price(self, parity_data: tuple[pd.DataFrame, pd.DataFrame]) -> None:
        ref, proj = parity_data
        ref_vals = ref["last_bottom_price"].to_numpy(dtype=float)
        proj_vals = proj["ms_last_bottom_price"].to_numpy(dtype=float)
        np.testing.assert_array_equal(ref_vals, proj_vals)

    def test_made_higher_high(self, parity_data: tuple[pd.DataFrame, pd.DataFrame]) -> None:
        ref, proj = parity_data
        ref_s = pd.array(ref["made_higher_high"], dtype="boolean")  # type: ignore[arg-type]
        proj_s = proj["ms_made_higher_high"]
        pd.testing.assert_extension_array_equal(ref_s, proj_s.array)  # type: ignore[arg-type]

    def test_made_lower_high(self, parity_data: tuple[pd.DataFrame, pd.DataFrame]) -> None:
        ref, proj = parity_data
        ref_s = pd.array(ref["made_lower_high"], dtype="boolean")  # type: ignore[arg-type]
        proj_s = proj["ms_made_lower_high"]
        pd.testing.assert_extension_array_equal(ref_s, proj_s.array)  # type: ignore[arg-type]

    def test_made_higher_low(self, parity_data: tuple[pd.DataFrame, pd.DataFrame]) -> None:
        ref, proj = parity_data
        ref_s = pd.array(ref["made_higher_low"], dtype="boolean")  # type: ignore[arg-type]
        proj_s = proj["ms_made_higher_low"]
        pd.testing.assert_extension_array_equal(ref_s, proj_s.array)  # type: ignore[arg-type]

    def test_made_lower_low(self, parity_data: tuple[pd.DataFrame, pd.DataFrame]) -> None:
        ref, proj = parity_data
        ref_s = pd.array(ref["made_lower_low"], dtype="boolean")  # type: ignore[arg-type]
        proj_s = proj["ms_made_lower_low"]
        pd.testing.assert_extension_array_equal(ref_s, proj_s.array)  # type: ignore[arg-type]

    def test_high_since(self, parity_data: tuple[pd.DataFrame, pd.DataFrame]) -> None:
        ref, proj = parity_data
        ref_s = pd.array(ref["high_since"], dtype="Int32")  # type: ignore[arg-type]
        proj_s = proj["ms_high_since"]
        pd.testing.assert_extension_array_equal(ref_s, proj_s.array)  # type: ignore[arg-type]

    def test_low_since(self, parity_data: tuple[pd.DataFrame, pd.DataFrame]) -> None:
        ref, proj = parity_data
        ref_s = pd.array(ref["low_since"], dtype="Int32")  # type: ignore[arg-type]
        proj_s = proj["ms_low_since"]
        pd.testing.assert_extension_array_equal(ref_s, proj_s.array)  # type: ignore[arg-type]

    def test_pullback_length(self, parity_data: tuple[pd.DataFrame, pd.DataFrame]) -> None:
        ref, proj = parity_data
        ref_s = pd.array(ref["pullback_length"], dtype="Int32")  # type: ignore[arg-type]
        proj_s = proj["ms_pullback_length"]
        pd.testing.assert_extension_array_equal(ref_s, proj_s.array)  # type: ignore[arg-type]

    def test_pullback_correction_factor(
        self, parity_data: tuple[pd.DataFrame, pd.DataFrame]
    ) -> None:
        ref, proj = parity_data
        ref_vals = ref["pullback_correction_factor"].to_numpy(dtype=float)
        proj_vals = proj["ms_pullback_correction_factor"].to_numpy(dtype=float)
        np.testing.assert_array_equal(ref_vals, proj_vals)

    def test_wave_length(self, parity_data: tuple[pd.DataFrame, pd.DataFrame]) -> None:
        ref, proj = parity_data
        ref_s = pd.array(ref["wave_length"], dtype="Int32")  # type: ignore[arg-type]
        proj_s = proj["ms_wave_length"]
        pd.testing.assert_extension_array_equal(ref_s, proj_s.array)  # type: ignore[arg-type]

    def test_pullback_breakout_level(self, parity_data: tuple[pd.DataFrame, pd.DataFrame]) -> None:
        ref, proj = parity_data
        ref_vals = ref["pullback_breakout_level"].to_numpy(dtype=float)
        proj_vals = proj["ms_pullback_breakout_level"].to_numpy(dtype=float)
        np.testing.assert_array_equal(ref_vals, proj_vals)

    def test_pullback_price_diff(self, parity_data: tuple[pd.DataFrame, pd.DataFrame]) -> None:
        ref, proj = parity_data
        ref_vals = ref["pullback_price_diff"].to_numpy(dtype=float)
        proj_vals = proj["ms_pullback_price_diff"].to_numpy(dtype=float)
        np.testing.assert_array_equal(ref_vals, proj_vals)

    def test_bearish_divergence(self, parity_data: tuple[pd.DataFrame, pd.DataFrame]) -> None:
        ref, proj = parity_data
        ref_s = pd.array(ref["bearish_divergence"], dtype="boolean")  # type: ignore[arg-type]
        proj_s = proj["ms_bearish_divergence"]
        pd.testing.assert_extension_array_equal(ref_s, proj_s.array)  # type: ignore[arg-type]

    def test_bullish_divergence(self, parity_data: tuple[pd.DataFrame, pd.DataFrame]) -> None:
        ref, proj = parity_data
        ref_s = pd.array(ref["bullish_divergence"], dtype="boolean")  # type: ignore[arg-type]
        proj_s = proj["ms_bullish_divergence"]
        pd.testing.assert_extension_array_equal(ref_s, proj_s.array)  # type: ignore[arg-type]

    def test_wave_count(self, parity_data: tuple[pd.DataFrame, pd.DataFrame]) -> None:
        ref, proj = parity_data
        ref_s = pd.array(ref["wave_count"], dtype="Int32")  # type: ignore[arg-type]
        proj_s = proj["ms_wave_count"]
        pd.testing.assert_extension_array_equal(ref_s, proj_s.array)  # type: ignore[arg-type]

    def test_forming_wave_high(self, parity_data: tuple[pd.DataFrame, pd.DataFrame]) -> None:
        ref, proj = parity_data
        ref_vals = ref["forming_wave_high"].to_numpy(dtype=float)
        proj_vals = proj["ms_forming_wave_high"].to_numpy(dtype=float)
        np.testing.assert_array_equal(ref_vals, proj_vals)

    def test_forming_wave_low(self, parity_data: tuple[pd.DataFrame, pd.DataFrame]) -> None:
        ref, proj = parity_data
        ref_vals = ref["forming_wave_low"].to_numpy(dtype=float)
        proj_vals = proj["ms_forming_wave_low"].to_numpy(dtype=float)
        np.testing.assert_array_equal(ref_vals, proj_vals)

    def test_support_zone_low(self, parity_data: tuple[pd.DataFrame, pd.DataFrame]) -> None:
        ref, proj = parity_data
        ref_vals = ref["support_zone_low"].to_numpy(dtype=float)
        proj_vals = proj["ms_support_zone_low"].to_numpy(dtype=float)
        np.testing.assert_array_equal(ref_vals, proj_vals)

    def test_support_zone_high(self, parity_data: tuple[pd.DataFrame, pd.DataFrame]) -> None:
        ref, proj = parity_data
        ref_vals = ref["support_zone_high"].to_numpy(dtype=float)
        proj_vals = proj["ms_support_zone_high"].to_numpy(dtype=float)
        np.testing.assert_array_equal(ref_vals, proj_vals)

    def test_resistance_zone_low(self, parity_data: tuple[pd.DataFrame, pd.DataFrame]) -> None:
        ref, proj = parity_data
        ref_vals = ref["resistance_zone_low"].to_numpy(dtype=float)
        proj_vals = proj["ms_resistance_zone_low"].to_numpy(dtype=float)
        np.testing.assert_array_equal(ref_vals, proj_vals)

    def test_resistance_zone_high(self, parity_data: tuple[pd.DataFrame, pd.DataFrame]) -> None:
        ref, proj = parity_data
        ref_vals = ref["resistance_zone_high"].to_numpy(dtype=float)
        proj_vals = proj["ms_resistance_zone_high"].to_numpy(dtype=float)
        np.testing.assert_array_equal(ref_vals, proj_vals)

    def test_support_is_double(self, parity_data: tuple[pd.DataFrame, pd.DataFrame]) -> None:
        ref, proj = parity_data
        ref_s = pd.array(ref["support_is_double"], dtype="boolean")  # type: ignore[arg-type]
        proj_s = proj["ms_support_is_double"]
        pd.testing.assert_extension_array_equal(ref_s, proj_s.array)  # type: ignore[arg-type]

    def test_resistance_is_double(self, parity_data: tuple[pd.DataFrame, pd.DataFrame]) -> None:
        ref, proj = parity_data
        ref_s = pd.array(ref["resistance_is_double"], dtype="boolean")  # type: ignore[arg-type]
        proj_s = proj["ms_resistance_is_double"]
        pd.testing.assert_extension_array_equal(ref_s, proj_s.array)  # type: ignore[arg-type]

    def test_support_overlap_count(self, parity_data: tuple[pd.DataFrame, pd.DataFrame]) -> None:
        ref, proj = parity_data
        ref_s = pd.array(ref["support_overlap_count"], dtype="Int32")  # type: ignore[arg-type]
        proj_s = proj["ms_support_overlap_count"]
        pd.testing.assert_extension_array_equal(ref_s, proj_s.array)  # type: ignore[arg-type]

    def test_resistance_overlap_count(self, parity_data: tuple[pd.DataFrame, pd.DataFrame]) -> None:
        ref, proj = parity_data
        ref_s = pd.array(ref["resistance_overlap_count"], dtype="Int32")  # type: ignore[arg-type]
        proj_s = proj["ms_resistance_overlap_count"]
        pd.testing.assert_extension_array_equal(ref_s, proj_s.array)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Column validation
# ---------------------------------------------------------------------------


class TestColumnValidation:
    def test_unknown_column_raises(self) -> None:
        df = _synthetic_df()
        store: dict[str, MarketStructureHelper] = {}
        with pytest.raises(ValueError, match="Unknown column"):
            attach_market_structure(
                df, {"pair": "TEST"}, store, hist_col="tsi_hist", columns=("bogus",)
            )

    def test_none_projects_all(self) -> None:
        df = _synthetic_df()
        store: dict[str, MarketStructureHelper] = {}
        result, _ = attach_market_structure(
            df, {"pair": "TEST"}, store, hist_col="tsi_hist", columns=None
        )
        for col in VALID_COLUMNS:
            assert f"ms_{col}" in result.columns

    def test_subset_only_requested(self) -> None:
        df = _synthetic_df()
        store: dict[str, MarketStructureHelper] = {}
        result, _ = attach_market_structure(
            df, {"pair": "TEST"}, store, hist_col="tsi_hist", columns=("wave_side",)
        )
        ms_cols = [c for c in result.columns if c.startswith("ms_")]
        assert ms_cols == ["ms_wave_side"]


# ---------------------------------------------------------------------------
# Backtest projection (synthetic)
# ---------------------------------------------------------------------------


class TestBacktestProjection:
    def test_wave_side_values(self) -> None:
        df = _synthetic_df()
        store: dict[str, MarketStructureHelper] = {}
        result, _ = attach_market_structure(
            df, {"pair": "TEST"}, store, hist_col="tsi_hist", columns=("wave_side",)
        )
        sides = list(result["ms_wave_side"])
        # Bars 0-1: up (hist >= 0), bars 2-3: down, bars 4-5: up, bar 6: down
        assert sides == ["up", "up", "down", "down", "up", "up", "down"]

    def test_wave_id_values(self) -> None:
        df = _synthetic_df()
        store: dict[str, MarketStructureHelper] = {}
        result, _ = attach_market_structure(
            df, {"pair": "TEST"}, store, hist_col="tsi_hist", columns=("wave_id",)
        )
        ids = list(result["ms_wave_id"])
        assert ids == ["w-0", "w-0", "w-1", "w-1", "w-2", "w-2", "forming-3"]

    def test_helper_stored(self) -> None:
        df = _synthetic_df()
        store: dict[str, MarketStructureHelper] = {}
        _, helper = attach_market_structure(
            df, {"pair": "TEST"}, store, hist_col="tsi_hist", columns=()
        )
        assert store["TEST"] is helper

    def test_max_waves_restored(self) -> None:
        df = _synthetic_df()
        store: dict[str, MarketStructureHelper] = {}
        _, helper = attach_market_structure(
            df, {"pair": "TEST"}, store, hist_col="tsi_hist", columns=(), max_waves=50
        )
        assert helper.max_waves == 50


# ---------------------------------------------------------------------------
# Live path
# ---------------------------------------------------------------------------


class TestLivePath:
    def test_first_call_hydrates(self) -> None:
        df = _synthetic_df()
        store: dict[str, MarketStructureHelper] = {}
        _, helper = attach_market_structure(
            df, {"pair": "TEST"}, store, hist_col="tsi_hist", columns=("wave_side",)
        )
        assert "TEST" in store
        assert helper.total_candles_registered == 7

    def test_subsequent_call_registers_candle(self) -> None:
        df = _synthetic_df()
        store: dict[str, MarketStructureHelper] = {}
        attach_market_structure(df, {"pair": "TEST"}, store, hist_col="tsi_hist", columns=())

        # Append a new bar and call again.
        df2 = pd.concat(
            [
                df,
                pd.DataFrame(
                    [
                        {
                            "open_time": 8000,
                            "open": 97.0,
                            "high": 100.0,
                            "low": 94.0,
                            "close": 95.0,
                            "volume": 1.0,
                            "tsi_hist": -0.5,
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )
        _, helper = attach_market_structure(
            df2, {"pair": "TEST"}, store, hist_col="tsi_hist", columns=("wave_side",)
        )
        assert helper.total_candles_registered == 8

    def test_dedup_on_same_open_time(self) -> None:
        df = _synthetic_df()
        store: dict[str, MarketStructureHelper] = {}
        attach_market_structure(df, {"pair": "TEST"}, store, hist_col="tsi_hist", columns=())

        # Call again with the same DataFrame — dedup should prevent double-count.
        _, helper = attach_market_structure(
            df, {"pair": "TEST"}, store, hist_col="tsi_hist", columns=("wave_side",)
        )
        assert helper.total_candles_registered == 7


# ---------------------------------------------------------------------------
# Desync detection
# ---------------------------------------------------------------------------


class TestDesync:
    def test_raises_on_backward_open_time(self) -> None:
        df = _synthetic_df()
        store: dict[str, MarketStructureHelper] = {}
        attach_market_structure(df, {"pair": "TEST"}, store, hist_col="tsi_hist", columns=())

        # Older DataFrame: last open_time = 5000 < helper's 7000.
        older_df = df.iloc[:5].copy().reset_index(drop=True)
        with pytest.raises(MarketStructureDesyncError, match="older"):
            attach_market_structure(
                older_df, {"pair": "TEST"}, store, hist_col="tsi_hist", columns=()
            )

    def test_no_error_on_same_open_time(self) -> None:
        df = _synthetic_df()
        store: dict[str, MarketStructureHelper] = {}
        attach_market_structure(df, {"pair": "TEST"}, store, hist_col="tsi_hist", columns=())

        # Same DataFrame again — should not raise.
        attach_market_structure(df, {"pair": "TEST"}, store, hist_col="tsi_hist", columns=())

    def test_no_error_on_forward_open_time(self) -> None:
        df = _synthetic_df()
        store: dict[str, MarketStructureHelper] = {}
        attach_market_structure(df, {"pair": "TEST"}, store, hist_col="tsi_hist", columns=())

        # Newer DataFrame.
        newer_df = pd.concat(
            [
                df,
                pd.DataFrame(
                    [
                        {
                            "open_time": 8000,
                            "open": 97.0,
                            "high": 100.0,
                            "low": 94.0,
                            "close": 95.0,
                            "volume": 1.0,
                            "tsi_hist": -0.5,
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )
        attach_market_structure(newer_df, {"pair": "TEST"}, store, hist_col="tsi_hist", columns=())


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_empty_dataframe(self) -> None:
        df = pd.DataFrame(
            columns=["open_time", "open", "high", "low", "close", "volume", "tsi_hist"]
        )
        store: dict[str, MarketStructureHelper] = {}
        result, helper = attach_market_structure(
            df, {"pair": "TEST"}, store, hist_col="tsi_hist", columns=("wave_side",)
        )
        assert len(result) == 0
        assert helper.total_candles_registered == 0

    def test_single_candle(self) -> None:
        df = pd.DataFrame(
            [
                {
                    "open_time": 1000,
                    "open": 100.0,
                    "high": 105.0,
                    "low": 98.0,
                    "close": 103.0,
                    "volume": 1.0,
                    "tsi_hist": 0.4,
                }
            ]
        )
        store: dict[str, MarketStructureHelper] = {}
        result, _ = attach_market_structure(
            df,
            {"pair": "TEST"},
            store,
            hist_col="tsi_hist",
            columns=("wave_side", "is_trending_up"),
        )
        assert list(result["ms_wave_side"]) == ["up"]
        assert list(result["ms_is_trending_up"]) == [False]

    def test_no_flips(self) -> None:
        """All same-sign histogram — one long wave, no confirmed waves."""
        df = pd.DataFrame(
            {
                "open_time": [1000, 2000, 3000, 4000],
                "open": [100.0, 102.0, 104.0, 103.0],
                "high": [105.0, 106.0, 108.0, 107.0],
                "low": [98.0, 100.0, 102.0, 101.0],
                "close": [103.0, 105.0, 107.0, 104.0],
                "volume": [1.0] * 4,
                "tsi_hist": [0.1, 0.2, 0.3, 0.1],
            }
        )
        store: dict[str, MarketStructureHelper] = {}
        result, _ = attach_market_structure(
            df,
            {"pair": "TEST"},
            store,
            hist_col="tsi_hist",
            columns=("wave_side", "last_top_price"),
        )
        assert all(s == "up" for s in result["ms_wave_side"])
        assert all(np.isnan(v) for v in result["ms_last_top_price"])
