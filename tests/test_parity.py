"""Stage 9 — Parity test against the LTC/USDT 4h fixture.

Two construction paths (``hydrate(df)`` and per-candle ``register_candle``)
must produce identical wave registries on real market data.  This test also
spot-checks specific wave properties against values from the TS test suite.
"""

import json
from pathlib import Path

import pandas as pd
import pytest

from market_structure import MarketStructureHelper
from market_structure.hydrate import hydrate
from market_structure.types import Candle

# ---------------------------------------------------------------------------
# Fixture loading
# ---------------------------------------------------------------------------

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "ms-LTCUSDT-4h.json"
HISTOGRAM_KEY = "tsi_histogram"


def _load_raw() -> list[dict[str, object]]:
    with FIXTURE_PATH.open() as f:
        return json.load(f)


def _filter_range(
    raw: list[dict[str, object]], from_str: str, to_str: str
) -> list[dict[str, object]]:
    """Filter candles matching the TS helper's getCandlesInRange logic.

    TS: ``openTime.isSameOrAfter(from) && closeTime.isBefore(to)``
    """
    from_ts = pd.Timestamp(from_str, tz="UTC")
    to_ts = pd.Timestamp(to_str, tz="UTC")
    return [
        row
        for row in raw
        if pd.Timestamp(str(row["openTime"])) >= from_ts
        and pd.Timestamp(str(row["closeTime"])) < to_ts
    ]


def _to_dataframe(rows: list[dict[str, object]]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    # pandas 2.x parses 'Z' suffixed ISO strings as datetime64[us, UTC].
    # Drop timezone → datetime64[ms] → int64 gives epoch milliseconds.
    df["open_time"] = (
        pd.to_datetime(df["openTime"]).dt.tz_localize(None).astype("datetime64[ms]").astype("int64")
    )
    cols = ["open_time", "open", "high", "low", "close", "volume", HISTOGRAM_KEY]
    return pd.DataFrame(df[cols])


def _ts(iso: str) -> int:
    """Convert an ISO-ish timestamp to epoch ms."""
    return int(pd.Timestamp(iso, tz="UTC").value // 10**6)


def _make_candle(row: dict[str, object]) -> Candle:
    return Candle(
        open_time=_ts(str(row["openTime"])),
        open=float(row["open"]),  # type: ignore[arg-type]
        high=float(row["high"]),  # type: ignore[arg-type]
        low=float(row["low"]),  # type: ignore[arg-type]
        close=float(row["close"]),  # type: ignore[arg-type]
        volume=float(row["volume"]),  # type: ignore[arg-type]
        histogram_value=float(row[HISTOGRAM_KEY]),  # type: ignore[arg-type]
    )


def _make_helper(from_str: str, to_str: str) -> MarketStructureHelper:
    """Build a helper via register_candle for a date range, matching the TS initHelper."""
    raw = _load_raw()
    rows = _filter_range(raw, from_str, to_str)
    h = MarketStructureHelper(histogram_key=HISTOGRAM_KEY)
    for row in rows:
        h.register_candle(_make_candle(row))
    return h


# ---------------------------------------------------------------------------
# Core parity: hydrate(df) == register_candle loop
# ---------------------------------------------------------------------------


class TestCoreParity:
    """The keystone test — both construction paths produce identical results."""

    def test_full_fixture_wave_registry_matches(self) -> None:
        raw = _load_raw()
        df = _to_dataframe(raw)

        # Incremental path
        h_inc = MarketStructureHelper(histogram_key=HISTOGRAM_KEY)
        for row in raw:
            h_inc.register_candle(_make_candle(row))

        # Hydrate path
        h_hyd = hydrate(df, histogram_key=HISTOGRAM_KEY)

        assert len(h_inc.wave_registry) == len(h_hyd.wave_registry)
        for w_inc, w_hyd in zip(h_inc.wave_registry, h_hyd.wave_registry, strict=True):
            assert w_inc == w_hyd, f"Wave mismatch at {w_inc.id}"

    def test_full_fixture_forming_wave_matches(self) -> None:
        raw = _load_raw()
        df = _to_dataframe(raw)

        h_inc = MarketStructureHelper(histogram_key=HISTOGRAM_KEY)
        for row in raw:
            h_inc.register_candle(_make_candle(row))

        h_hyd = hydrate(df, histogram_key=HISTOGRAM_KEY)

        w_inc = h_inc.get_current_wave()
        w_hyd = h_hyd.get_current_wave()
        assert w_inc is not None
        assert w_hyd is not None
        assert w_inc == w_hyd

    def test_subrange_parity(self) -> None:
        """Parity on the standard TS test range (not the full fixture)."""
        raw = _load_raw()
        rows = _filter_range(raw, "2020-06-28 04:00:00", "2020-07-12 20:00:00")
        df = _to_dataframe(rows)

        h_inc = MarketStructureHelper(histogram_key=HISTOGRAM_KEY)
        for row in rows:
            h_inc.register_candle(_make_candle(row))

        h_hyd = hydrate(df, histogram_key=HISTOGRAM_KEY)

        assert len(h_inc.wave_registry) == len(h_hyd.wave_registry)
        for w_inc, w_hyd in zip(h_inc.wave_registry, h_hyd.wave_registry, strict=True):
            assert w_inc == w_hyd, f"Wave mismatch at {w_inc.id}"


# ---------------------------------------------------------------------------
# Spot checks against TS test values
# ---------------------------------------------------------------------------


class TestWaveCount:
    """TS: 'Produced a registry of WaveType[]'"""

    def test_18_waves_in_standard_range(self) -> None:
        h = _make_helper("2020-06-28 04:00:00", "2020-07-12 20:00:00")
        assert len(h.wave_registry) == 18

    def test_9_up_9_down(self) -> None:
        h = _make_helper("2020-06-28 04:00:00", "2020-07-12 20:00:00")
        ups = [w for w in h.wave_registry if w.side == "up"]
        downs = [w for w in h.wave_registry if w.side == "down"]
        assert len(ups) == 9
        assert len(downs) == 9


class TestExtremes:
    """TS: 'Determines tops and bottoms' — spot-check extremum timestamps."""

    @pytest.fixture()
    def h(self) -> MarketStructureHelper:
        return _make_helper("2020-06-28 04:00:00", "2020-07-12 20:00:00")

    def test_last_bottom_lco(self, h: MarketStructureHelper) -> None:
        bottom = h.get_last_bottom()
        assert bottom is not None
        assert bottom.lowest_close_or_open.open_time == _ts("2020-07-11T08:00:00.000Z")

    def test_last_bottom_lowest_close(self, h: MarketStructureHelper) -> None:
        bottom = h.get_last_bottom()
        assert bottom is not None
        assert bottom.lowest_close.open_time == _ts("2020-07-11T08:00:00.000Z")

    def test_last_bottom_low(self, h: MarketStructureHelper) -> None:
        bottom = h.get_last_bottom()
        assert bottom is not None
        assert bottom.low.open_time == _ts("2020-07-11T08:00:00.000Z")

    def test_previous_bottom_lco(self, h: MarketStructureHelper) -> None:
        bottom = h.get_previous_bottom()
        assert bottom is not None
        assert bottom.lowest_close_or_open.open_time == _ts("2020-07-09T16:00:00.000Z")

    def test_previous_bottom_lowest_close(self, h: MarketStructureHelper) -> None:
        bottom = h.get_previous_bottom()
        assert bottom is not None
        assert bottom.lowest_close.open_time == _ts("2020-07-09T16:00:00.000Z")

    def test_previous_bottom_low(self, h: MarketStructureHelper) -> None:
        bottom = h.get_previous_bottom()
        assert bottom is not None
        assert bottom.low.open_time == _ts("2020-07-10T04:00:00.000Z")

    def test_last_top_hco(self, h: MarketStructureHelper) -> None:
        top = h.get_last_top()
        assert top is not None
        assert top.highest_close_or_open.open_time == _ts("2020-07-12T00:00:00.000Z")

    def test_last_top_highest_close(self, h: MarketStructureHelper) -> None:
        top = h.get_last_top()
        assert top is not None
        assert top.highest_close.open_time == _ts("2020-07-12T00:00:00.000Z")

    def test_last_top_high(self, h: MarketStructureHelper) -> None:
        top = h.get_last_top()
        assert top is not None
        assert top.high.open_time == _ts("2020-07-12T04:00:00.000Z")

    def test_previous_top_hco(self, h: MarketStructureHelper) -> None:
        top = h.get_previous_top()
        assert top is not None
        assert top.highest_close_or_open.open_time == _ts("2020-07-11T04:00:00.000Z")

    def test_previous_top_highest_close(self, h: MarketStructureHelper) -> None:
        top = h.get_previous_top()
        assert top is not None
        assert top.highest_close.open_time == _ts("2020-07-11T04:00:00.000Z")

    def test_previous_top_high(self, h: MarketStructureHelper) -> None:
        top = h.get_previous_top()
        assert top is not None
        assert top.high.open_time == _ts("2020-07-11T00:00:00.000Z")

    def test_forming_wave(self, h: MarketStructureHelper) -> None:
        wave = h.get_current_wave()
        assert wave is not None
        assert wave.side == "down"
        assert wave.lowest_close_or_open.open_time == _ts("2020-07-12T16:00:00.000Z")
        assert wave.lowest_close.open_time == _ts("2020-07-12T12:00:00.000Z")
        assert wave.low.open_time == _ts("2020-07-12T12:00:00.000Z")


class TestHighSince:
    """TS: 'Determines long term highs'"""

    def test_high_since_64(self) -> None:
        h = _make_helper("2020-06-28 04:00:00", "2020-07-10 00:00:00")
        top = h.get_last_top()
        assert top is not None
        assert top.high_since == 64

    def test_high_since_25(self) -> None:
        h = _make_helper("2020-06-28 04:00:00", "2020-07-13 20:00:00")
        top = h.get_last_top()
        assert top is not None
        assert top.high_since == 25

    def test_high_since_184(self) -> None:
        h = _make_helper("2020-06-28 04:00:00", "2020-07-30 20:00:00")
        top = h.get_last_top()
        assert top is not None
        assert top.high_since == 184


class TestLowSince:
    """TS: 'Determines long term lows'"""

    def test_low_since_23(self) -> None:
        h = _make_helper("2020-06-28 04:00:00", "2020-07-21 12:00:00")
        bottom = h.get_last_bottom()
        assert bottom is not None
        assert bottom.low_since == 23

    def test_low_since_32(self) -> None:
        h = _make_helper("2020-07-30 04:00:00", "2020-08-08 16:00:00")
        bottom = h.get_last_bottom()
        assert bottom is not None
        assert bottom.low_since == 32

    def test_low_since_77(self) -> None:
        h = _make_helper("2020-07-30 04:00:00", "2020-08-13 00:00:00")
        bottom = h.get_last_bottom()
        assert bottom is not None
        assert bottom.low_since == 77


class TestPullback:
    """TS: 'Determines pullback properties'"""

    def test_pullback_from_top_length(self) -> None:
        h = _make_helper("2020-07-20 00:00:00", "2020-07-28 00:00:00")
        bottom = h.get_last_bottom()
        assert bottom is not None
        assert bottom.pullback is not None
        assert bottom.pullback.length == 7

    def test_pullback_from_bottom_length(self) -> None:
        h = _make_helper("2020-07-10 00:00:00", "2020-07-19 16:00:00")
        top = h.get_last_top()
        assert top is not None
        assert top.pullback is not None
        assert top.pullback.length == 10

    def test_pullback_from_top_breakout_level(self) -> None:
        h = _make_helper("2020-07-20 00:00:00", "2020-07-28 00:00:00")
        bottom = h.get_last_bottom()
        assert bottom is not None
        assert bottom.pullback is not None
        assert bottom.pullback.breakout_level == 49.43

    def test_pullback_from_bottom_breakout_level(self) -> None:
        h = _make_helper("2020-07-10 00:00:00", "2020-07-19 16:00:00")
        top = h.get_last_top()
        assert top is not None
        assert top.pullback is not None
        assert top.pullback.breakout_level == 41.72

    def test_pullback_from_top_price_diff(self) -> None:
        h = _make_helper("2020-07-20 00:00:00", "2020-07-28 00:00:00")
        bottom = h.get_last_bottom()
        assert bottom is not None
        assert bottom.pullback is not None
        assert round(bottom.pullback.price_diff, 2) == -1.82

    def test_pullback_from_bottom_price_diff(self) -> None:
        h = _make_helper("2020-07-10 00:00:00", "2020-07-19 16:00:00")
        top = h.get_last_top()
        assert top is not None
        assert top.pullback is not None
        assert round(top.pullback.price_diff, 2) == 0.89

    def test_pullback_from_top_correction_factor(self) -> None:
        h = _make_helper("2020-07-20 00:00:00", "2020-07-28 00:00:00")
        bottom = h.get_last_bottom()
        assert bottom is not None
        assert bottom.pullback is not None
        assert bottom.pullback.correction_factor is not None
        assert round(bottom.pullback.correction_factor, 2) == 0.34

    def test_pullback_from_bottom_correction_factor(self) -> None:
        h = _make_helper("2020-07-10 00:00:00", "2020-07-19 16:00:00")
        top = h.get_last_top()
        assert top is not None
        assert top.pullback is not None
        assert top.pullback.correction_factor is not None
        assert round(top.pullback.correction_factor, 2) == 0.41

    def test_atr_factor_not_implemented(self) -> None:
        """Known divergence: TS computes atr_factor from ATR column, we return None."""
        h = _make_helper("2020-07-20 00:00:00", "2020-07-28 00:00:00")
        bottom = h.get_last_bottom()
        assert bottom is not None
        assert bottom.pullback is not None
        assert bottom.pullback.atr_factor is None
