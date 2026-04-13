"""Tests for the TSI computation module."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from market_structure.tsi import compute_tsi

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "ms-LTCUSDT-4h.json"


def _load_fixture() -> tuple[pd.Series, pd.DataFrame]:  # type: ignore[type-arg]
    """Load close prices and expected TSI values from fixture."""
    with FIXTURE_PATH.open() as f:
        raw = json.load(f)
    df = pd.DataFrame(raw)
    close: pd.Series = df["close"].astype(float)  # type: ignore[assignment]
    expected: pd.DataFrame = df[["tsi", "tsi_signal", "tsi_histogram"]].astype(float)  # type: ignore[assignment]
    return close, expected


class TestComputeTSI:
    def test_histogram_equals_tsi_minus_signal(self) -> None:
        close, _ = _load_fixture()
        result = compute_tsi(close)
        diff = result["tsi"] - result["tsi_signal"]
        np.testing.assert_allclose(
            result["tsi_histogram"].to_numpy(),
            diff.to_numpy(),
            rtol=1e-10,
        )

    def test_tsi_bounded(self) -> None:
        """TSI should be in [-100, 100] range after warmup."""
        close, _ = _load_fixture()
        result = compute_tsi(close)
        tsi_vals = result["tsi"].dropna().to_numpy()
        assert (tsi_vals >= -100).all() and (tsi_vals <= 100).all()

    def test_histogram_sign_correlation(self) -> None:
        """Histogram sign should mostly agree with the fixture's sign pattern."""
        close, expected = _load_fixture()
        result = compute_tsi(close)
        start = 30  # skip warmup
        computed_sign = np.sign(result["tsi_histogram"].to_numpy()[start:])
        expected_sign = np.sign(expected["tsi_histogram"].to_numpy()[start:])
        agreement = np.mean(computed_sign == expected_sign)
        assert agreement > 0.85, f"Histogram sign agreement {agreement:.2%} too low"

    def test_output_columns(self) -> None:
        close = pd.Series([100.0, 101.0, 99.0, 102.0, 98.0])
        result = compute_tsi(close)
        assert list(result.columns) == ["tsi", "tsi_signal", "tsi_histogram"]

    def test_short_series(self) -> None:
        """Series shorter than warmup still returns valid DataFrame shape."""
        close = pd.Series([100.0, 101.0])
        result = compute_tsi(close)
        assert len(result) == 2
        assert np.isnan(result["tsi"].iloc[0])

    def test_constant_price_produces_nan(self) -> None:
        """Constant prices → zero momentum → 0/0 → NaN TSI."""
        close = pd.Series([100.0] * 20)
        result = compute_tsi(close)
        # After first bar, momentum is 0, so TSI is 0/0 = NaN.
        assert result["tsi"].iloc[2:].isna().all()

    def test_monotonic_up_positive_tsi(self) -> None:
        """Strictly rising prices should produce positive TSI after warmup."""
        close = pd.Series([100.0 + i for i in range(50)])
        result = compute_tsi(close)
        # After warmup, TSI should be positive.
        assert (result["tsi"].iloc[15:] > 0).all()
