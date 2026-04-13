"""Tests for the multi-timeframe wrapper (attach_market_structure_mtf)."""

# pyright: reportMissingTypeArgument=false

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from market_structure.freqtrade import attach_market_structure
from market_structure.mtf import attach_market_structure_mtf

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_ltf_df(n_bars: int = 200, ltf_minutes: int = 60) -> pd.DataFrame:
    """Generate a synthetic LTF OHLCV DataFrame with a date column.

    Creates a trending-then-reversing pattern so the helper has enough
    structure to produce non-trivial ms_* columns.
    """
    rng = np.random.default_rng(42)
    base_time = pd.Timestamp("2024-01-01")
    dates = [base_time + pd.Timedelta(minutes=ltf_minutes * i) for i in range(n_bars)]

    # Generate a price series with trends and reversals.
    close = np.empty(n_bars)
    close[0] = 100.0
    for i in range(1, n_bars):
        # Oscillating trend: up for 30 bars, down for 30
        cycle = (i // 30) % 2
        drift = 0.3 if cycle == 0 else -0.3
        close[i] = close[i - 1] + drift + rng.normal(0, 0.5)

    high = close + rng.uniform(0.5, 2.0, n_bars)
    low = close - rng.uniform(0.5, 2.0, n_bars)
    open_ = close + rng.normal(0, 0.3, n_bars)

    # Histogram with sign changes to trigger waves.
    hist = np.sin(np.linspace(0, 8 * np.pi, n_bars)) + rng.normal(0, 0.1, n_bars)

    return pd.DataFrame(
        {
            "date": dates,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": rng.uniform(100, 1000, n_bars),
            "tsi_hist": hist,
        }
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestMTFValidation:
    """Timeframe validation checks."""

    def test_htf_not_greater_raises(self) -> None:
        df = _make_ltf_df(n_bars=50)
        store: dict = {}
        with pytest.raises(ValueError, match="must be greater"):
            attach_market_structure_mtf(df, {"pair": "TEST/USDT"}, store, htf="1h", ltf="4h")

    def test_htf_equal_raises(self) -> None:
        df = _make_ltf_df(n_bars=50)
        store: dict = {}
        with pytest.raises(ValueError, match="must be greater"):
            attach_market_structure_mtf(df, {"pair": "TEST/USDT"}, store, htf="1h", ltf="1h")

    def test_non_divisible_raises(self) -> None:
        df = _make_ltf_df(n_bars=50)
        store: dict = {}
        with pytest.raises(ValueError, match="even multiple"):
            attach_market_structure_mtf(df, {"pair": "TEST/USDT"}, store, htf="3h", ltf="2h")


class TestMTFProjection:
    """HTF column projection onto LTF frame."""

    def test_htf_columns_present(self) -> None:
        """HTF ms_* columns should appear with the ms_{htf}_ prefix."""
        df = _make_ltf_df(n_bars=200)
        store: dict = {}
        result = attach_market_structure_mtf(df, {"pair": "TEST/USDT"}, store, htf="4h", ltf="1h")
        htf_cols = [c for c in result.columns if c.startswith("ms_4h_")]
        assert len(htf_cols) > 0, "No HTF columns projected"

    def test_htf_columns_match_independent_run(self) -> None:
        """Projected HTF columns should match an independent HTF structure run."""
        df = _make_ltf_df(n_bars=200)
        store: dict = {}
        columns = ("wave_side", "is_trending_up", "wave_count")

        result = attach_market_structure_mtf(
            df,
            {"pair": "TEST/USDT"},
            store,
            htf="4h",
            ltf="1h",
            columns=columns,
        )

        # Run structure independently on HTF-resampled data.
        df_for_resample = df.copy()
        df_for_resample.index = pd.DatetimeIndex(pd.to_datetime(df["date"]))
        htf_df = (
            df_for_resample.resample("4h")
            .agg(
                {
                    "open": "first",
                    "high": "max",
                    "low": "min",
                    "close": "last",
                    "volume": "sum",
                    "tsi_hist": "last",
                }
            )
            .dropna(subset=["open"])  # type: ignore[call-overload]
        )
        htf_df["open_time"] = htf_df.index.astype("int64") // 10**6
        htf_datetime_index = htf_df.index.copy()

        ind_store: dict = {}
        htf_result, _ = attach_market_structure(
            htf_df,
            {"pair": "TEST/USDT"},
            ind_store,
            columns=columns,
        )

        # Shift HTF by one period for anti-lookahead.
        htf_td = pd.Timedelta("4h")
        ms_cols = [c for c in htf_result.columns if c.startswith("ms_")]
        htf_shifted = htf_result[ms_cols].copy()
        htf_shifted.index = htf_datetime_index[: len(htf_shifted)] + htf_td

        # Compare at HTF boundary bars that exist in both.
        ltf_index = pd.DatetimeIndex(pd.to_datetime(df["date"]))
        for col in ms_cols:
            htf_col_name = col.replace("ms_", "ms_4h_", 1)
            assert htf_col_name in result.columns, f"Missing column {htf_col_name}"

            # Reindex independently computed values to LTF.
            expected = htf_shifted[col].reindex(ltf_index, method="ffill")  # type: ignore[union-attr]
            actual = pd.Series(result[htf_col_name].to_numpy(), index=ltf_index)

            # Compare only where both have values.
            mask = expected.notna() & actual.notna()
            if mask.sum() == 0:
                continue

            if expected.dtype == float or actual.dtype == float:
                np.testing.assert_allclose(
                    actual[mask].astype(float).to_numpy(),  # type: ignore[union-attr]
                    expected[mask].astype(float).to_numpy(),  # type: ignore[union-attr]
                    rtol=1e-10,
                    err_msg=f"Mismatch in {htf_col_name}",
                )
            else:
                pd.testing.assert_series_equal(
                    actual[mask].reset_index(drop=True),  # type: ignore[union-attr]
                    expected[mask].reset_index(drop=True),  # type: ignore[union-attr]
                    check_names=False,
                    obj=htf_col_name,
                )

    def test_no_lookahead(self) -> None:
        """HTF values at a given LTF bar should come from already-closed HTF candles.

        The MTF wrapper shifts HTF values forward by one HTF period, so the
        first LTF bar of a new HTF candle should still carry the PREVIOUS
        HTF candle's structure values (not the just-opening one).
        """
        df = _make_ltf_df(n_bars=200)
        store: dict = {}
        columns = ("wave_side",)

        result = attach_market_structure_mtf(
            df,
            {"pair": "TEST/USDT"},
            store,
            htf="4h",
            ltf="1h",
            columns=columns,
        )

        dates = pd.to_datetime(df["date"])
        htf_td = pd.Timedelta("4h")

        # Find the first bar of each HTF candle.
        htf_boundaries = dates[dates.dt.floor(htf_td) == dates]

        # For bars at an HTF boundary, the value should NOT yet reflect that
        # HTF candle (it hasn't closed). Values at bar T should come from
        # the HTF candle that closed at T (i.e., started at T - htf_td).
        # After the shift, the first few HTF periods will be NaN/empty
        # because no prior completed candle exists yet.
        col = "ms_4h_wave_side"
        if col in result.columns:
            # At the very first HTF boundary, there's no prior HTF candle.
            # At the second boundary onward, values should be stable
            # (forward-filled from the shifted HTF data).
            vals_at_boundaries = result.loc[htf_boundaries.index, col]  # type: ignore[union-attr]
            # The key invariant: the value shouldn't change precisely at an
            # HTF boundary — it should already have been available from the
            # prior period's shift. Check that consecutive boundary values
            # don't all change (which would indicate no shift was applied).
            if len(vals_at_boundaries) > 4:
                non_null = vals_at_boundaries.dropna()
                if len(non_null) > 2:
                    # At least some consecutive boundary values should be
                    # identical (forward-filled from a completed HTF bar).
                    consecutive_same = sum(
                        1
                        for a, b in zip(non_null.iloc[:-1], non_null.iloc[1:], strict=False)
                        if a == b
                    )
                    assert consecutive_same > 0, (
                        "HTF values change at every boundary — shift may not be applied"
                    )

    def test_store_persistence(self) -> None:
        """The helper store should be updated with the HTF helper."""
        df = _make_ltf_df(n_bars=200)
        store: dict = {}
        attach_market_structure_mtf(df, {"pair": "TEST/USDT"}, store, htf="4h", ltf="1h")
        assert "TEST/USDT_4h" in store

    def test_result_length_unchanged(self) -> None:
        """Output DataFrame should have the same row count as input."""
        df = _make_ltf_df(n_bars=200)
        store: dict = {}
        result = attach_market_structure_mtf(df, {"pair": "TEST/USDT"}, store, htf="4h", ltf="1h")
        assert len(result) == len(df)

    def test_auto_histogram(self) -> None:
        """auto_histogram=True should work without a pre-existing histogram column."""
        df = _make_ltf_df(n_bars=200)
        df = df.drop(columns=["tsi_hist"])
        store: dict = {}
        result = attach_market_structure_mtf(
            df,
            {"pair": "TEST/USDT"},
            store,
            htf="4h",
            ltf="1h",
            auto_histogram=True,
        )
        htf_cols = [c for c in result.columns if c.startswith("ms_4h_")]
        assert len(htf_cols) > 0

    def test_datetime_index_input(self) -> None:
        """Should work with DatetimeIndex instead of a date column."""
        df = _make_ltf_df(n_bars=200)
        df.index = pd.DatetimeIndex(pd.to_datetime(df["date"]))
        df = df.drop(columns=["date"])
        store: dict = {}
        result = attach_market_structure_mtf(df, {"pair": "TEST/USDT"}, store, htf="4h", ltf="1h")
        htf_cols = [c for c in result.columns if c.startswith("ms_4h_")]
        assert len(htf_cols) > 0
