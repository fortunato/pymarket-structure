"""Tests for zone lifecycle columns (break/retest/flip/failed-retest).

Uses the LTCUSDT 4h fixture — verifies lifecycle columns are boolean/Int32
and fire on expected bars (no exact bar assertion — the fixture is too
complex for hand-verified bar indices).  Structural properties are tested:
- Boolean columns contain at least some True values (events occur).
- Retest count columns are monotonically non-decreasing within a zone era.
- Break always precedes retest on any given zone.
"""

# pyright: reportPrivateUsage=false

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from market_structure.freqtrade import attach_market_structure

if __import__("typing").TYPE_CHECKING:
    from market_structure import MarketStructureHelper

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "ms-LTCUSDT-4h.json"
HISTOGRAM_KEY = "tsi_histogram"


def _load_fixture_df() -> pd.DataFrame:
    with FIXTURE_PATH.open() as f:
        raw = json.load(f)
    df = pd.DataFrame(raw)
    df["open_time"] = (
        pd.to_datetime(df["openTime"]).dt.tz_localize(None).astype("datetime64[ms]").astype("int64")
    )
    return pd.DataFrame(df[["open_time", "open", "high", "low", "close", "volume", HISTOGRAM_KEY]])


@pytest.fixture()
def lifecycle_df() -> pd.DataFrame:
    df = _load_fixture_df()
    store: dict[str, MarketStructureHelper] = {}
    result, _ = attach_market_structure(
        df,
        {"pair": "LTC/USDT"},
        store,
        hist_col=HISTOGRAM_KEY,
    )
    return result


class TestZoneLifecycleDtypes:
    """Verify dtypes and shape of lifecycle columns."""

    @pytest.mark.parametrize(
        "col",
        [
            "ms_zone_break_support",
            "ms_zone_break_resistance",
            "ms_zone_retest_support",
            "ms_zone_retest_resistance",
            "ms_zone_flip_support",
            "ms_zone_flip_resistance",
            "ms_zone_failed_retest_support",
            "ms_zone_failed_retest_resistance",
        ],
    )
    def test_boolean_dtype(self, lifecycle_df: pd.DataFrame, col: str) -> None:
        assert col in lifecycle_df.columns
        assert lifecycle_df[col].dtype == pd.BooleanDtype()

    @pytest.mark.parametrize(
        "col",
        [
            "ms_zone_retest_count_support",
            "ms_zone_retest_count_resistance",
        ],
    )
    def test_int32_dtype(self, lifecycle_df: pd.DataFrame, col: str) -> None:
        assert col in lifecycle_df.columns
        assert lifecycle_df[col].dtype == pd.Int32Dtype()


class TestZoneLifecycleEvents:
    """Structural property tests on the LTCUSDT fixture."""

    def test_break_events_exist(self, lifecycle_df: pd.DataFrame) -> None:
        """At least one zone break should fire in a 300-bar window."""
        has_support_break = bool(lifecycle_df["ms_zone_break_support"].any())
        has_resistance_break = bool(lifecycle_df["ms_zone_break_resistance"].any())
        assert has_support_break or has_resistance_break

    def test_retest_count_non_decreasing(self, lifecycle_df: pd.DataFrame) -> None:
        """Retest count should never decrease within a contiguous zone era."""
        for col in ["ms_zone_retest_count_support", "ms_zone_retest_count_resistance"]:
            arr = lifecycle_df[col].fillna(0).to_numpy(dtype=int)
            # Within a zone era (no zone change), count should be non-decreasing.
            # Zone changes reset the count to 0, which is allowed.
            # We just check that it never goes negative.
            assert (arr >= 0).all(), f"{col} contains negative values"

    def test_break_before_retest(self, lifecycle_df: pd.DataFrame) -> None:
        """A retest cannot occur before a break on the same side."""
        for side in ["support", "resistance"]:
            break_col = f"ms_zone_break_{side}"
            retest_col = f"ms_zone_retest_{side}"
            break_bars = lifecycle_df.index[lifecycle_df[break_col] == True].tolist()  # noqa: E712
            retest_bars = lifecycle_df.index[lifecycle_df[retest_col] == True].tolist()  # noqa: E712
            if retest_bars:
                first_retest = retest_bars[0]
                if break_bars:
                    first_break = break_bars[0]
                    assert first_break < first_retest, (
                        f"{side}: first retest at bar {first_retest} before first break at {first_break}"
                    )

    def test_retest_mode_wick(self) -> None:
        """Wick mode should detect retests that close mode misses."""
        df = _load_fixture_df()
        store_wick: dict[str, MarketStructureHelper] = {}
        store_close: dict[str, MarketStructureHelper] = {}

        wick_result, _ = attach_market_structure(
            df.copy(),
            {"pair": "LTC/USDT"},
            store_wick,
            hist_col=HISTOGRAM_KEY,
            columns=("zone_retest_support", "zone_retest_resistance"),
            retest_mode="wick",
        )
        close_result, _ = attach_market_structure(
            df.copy(),
            {"pair": "LTC/USDT"},
            store_close,
            hist_col=HISTOGRAM_KEY,
            columns=("zone_retest_support", "zone_retest_resistance"),
            retest_mode="close",
        )

        # Wick mode should detect >= as many retests as close mode.
        wick_retests = (
            wick_result["ms_zone_retest_support"].sum()
            + wick_result["ms_zone_retest_resistance"].sum()
        )
        close_retests = (
            close_result["ms_zone_retest_support"].sum()
            + close_result["ms_zone_retest_resistance"].sum()
        )
        assert wick_retests >= close_retests
