"""Tests for internal ATR computation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from market_structure.atr import _compute_atr

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "ms-LTCUSDT-4h.json"


@pytest.fixture()
def fixture_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load OHLC + expected ATR from the LTCUSDT fixture."""
    with FIXTURE_PATH.open() as f:
        raw = json.load(f)
    highs = np.array([float(r["high"]) for r in raw])
    lows = np.array([float(r["low"]) for r in raw])
    closes = np.array([float(r["close"]) for r in raw])
    expected_atr = np.array([float(r["atr"]) for r in raw])
    return highs, lows, closes, expected_atr


class TestATRCorrectness:
    def test_matches_fixture(
        self, fixture_data: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    ) -> None:
        highs, lows, closes, expected = fixture_data
        result = _compute_atr(highs, lows, closes, period=14)

        # First 13 bars should be NaN
        assert np.all(np.isnan(result[:13]))

        # The fixture ATR is pre-warmed from historical data before the
        # fixture window, so early values diverge.  Wilder's smoothing
        # converges exponentially; by bar 100 the difference is <0.02%.
        # Fixture ATR is pre-warmed; convergence is ~1e-6 by bar 150.
        convergence_bar = 150
        np.testing.assert_allclose(
            result[convergence_bar:],
            expected[convergence_bar:],
            rtol=1e-5,
        )

    def test_period_length(
        self, fixture_data: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    ) -> None:
        highs, lows, closes, _ = fixture_data
        result = _compute_atr(highs, lows, closes, period=14)
        assert len(result) == len(highs)


class TestATREdgeCases:
    def test_empty_input(self) -> None:
        result = _compute_atr(np.array([]), np.array([]), np.array([]), period=14)
        assert len(result) == 0

    def test_insufficient_bars(self) -> None:
        highs = np.array([10.0, 11.0, 12.0])
        lows = np.array([9.0, 10.0, 11.0])
        closes = np.array([9.5, 10.5, 11.5])
        result = _compute_atr(highs, lows, closes, period=14)
        assert np.all(np.isnan(result))

    def test_constant_price(self) -> None:
        n = 30
        price = np.full(n, 100.0)
        result = _compute_atr(price, price, price, period=14)
        # First 13 NaN, bar 13 onward should be 0 (no range)
        valid = ~np.isnan(result)
        np.testing.assert_array_equal(result[valid], 0.0)

    def test_single_bar(self) -> None:
        result = _compute_atr(np.array([10.0]), np.array([9.0]), np.array([9.5]), period=14)
        assert len(result) == 1
        assert np.isnan(result[0])
