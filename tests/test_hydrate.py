"""Stage 8 tests for the vectorized hydrate path.

``hydrate(df)`` bulk-constructs a ``MarketStructureHelper`` from a complete
OHLCV DataFrame.  The resulting helper must be in the same state as if
every row had been fed through ``register_candle`` one at a time.
"""

import pandas as pd

from market_structure import MarketStructureHelper
from market_structure.hydrate import hydrate
from market_structure.types import Candle

# ---------------------------------------------------------------------------
# Test data
# ---------------------------------------------------------------------------

# 7 rows, 3 sign-flips → 3 confirmed waves + 1 forming wave.
# Wave 0 (up):   rows 0-1   hist=0.4, 0.2
# Wave 1 (down): rows 2-3   hist=-0.1, -0.3
# Wave 2 (up):   rows 4-5   hist=0.2, 0.5
# Forming (down): row 6     hist=-0.4
ROWS = [
    # (open_time, open, high, low, close, volume, tsi_hist)
    (1_000, 100.0, 105.0, 98.0, 103.0, 1.0, 0.4),
    (2_000, 102.0, 110.0, 100.0, 108.0, 1.0, 0.2),
    (3_000, 107.0, 109.0, 101.0, 106.0, 1.0, -0.1),
    (4_000, 104.0, 106.0, 97.0, 98.0, 1.0, -0.3),
    (5_000, 99.0, 103.0, 96.0, 101.0, 1.0, 0.2),
    (6_000, 100.0, 108.0, 99.0, 107.0, 1.0, 0.5),
    (7_000, 106.0, 107.0, 95.0, 96.0, 1.0, -0.4),
]

COLUMNS = ["open_time", "open", "high", "low", "close", "volume", "tsi_hist"]


def _make_df(rows: list[tuple[float, ...]] | None = None) -> pd.DataFrame:
    return pd.DataFrame(rows or ROWS, columns=COLUMNS)


def _make_candle(row: tuple[float, ...]) -> Candle:
    return Candle(
        open_time=int(row[0]),
        open=float(row[1]),
        high=float(row[2]),
        low=float(row[3]),
        close=float(row[4]),
        volume=float(row[5]),
        histogram_value=float(row[6]),
    )


# ---------------------------------------------------------------------------
# Empty / trivial inputs
# ---------------------------------------------------------------------------


class TestHydrateEmpty:
    def test_empty_dataframe(self) -> None:
        h = hydrate(pd.DataFrame(columns=COLUMNS))
        assert h.wave_registry == ()
        assert h.get_current_wave() is None
        assert h.total_candles_registered == 0

    def test_single_candle_no_flips(self) -> None:
        """One row → no confirmed waves, forming wave exists."""
        df = _make_df([(1_000, 100.0, 105.0, 98.0, 103.0, 1.0, 0.5)])
        h = hydrate(df)
        assert h.wave_registry == ()
        wave = h.get_current_wave()
        assert wave is not None
        assert wave.side == "up"
        assert len(wave.candles) == 1


# ---------------------------------------------------------------------------
# Basic hydration
# ---------------------------------------------------------------------------


class TestHydrateBasic:
    def test_single_flip_one_confirmed_wave(self) -> None:
        """Two groups → one confirmed wave + forming wave."""
        df = _make_df(
            [
                (1_000, 100.0, 105.0, 98.0, 103.0, 1.0, 0.5),
                (2_000, 102.0, 110.0, 100.0, 108.0, 1.0, 0.3),
                (3_000, 107.0, 109.0, 101.0, 106.0, 1.0, -0.2),  # flip
            ]
        )
        h = hydrate(df)
        assert len(h.wave_registry) == 1
        assert h.wave_registry[0].side == "up"
        assert h.wave_registry[0].id == "w-0"

        wave = h.get_current_wave()
        assert wave is not None
        assert wave.side == "down"

    def test_multiple_flips(self) -> None:
        """3 flips → 3 confirmed waves."""
        h = hydrate(_make_df())
        assert len(h.wave_registry) == 3
        assert h.wave_registry[0].side == "up"
        assert h.wave_registry[1].side == "down"
        assert h.wave_registry[2].side == "up"

    def test_wave_ids_sequential(self) -> None:
        h = hydrate(_make_df())
        assert h.wave_registry[0].id == "w-0"
        assert h.wave_registry[1].id == "w-1"
        assert h.wave_registry[2].id == "w-2"

    def test_no_flips_all_forming(self) -> None:
        """All same-sign histogram → no confirmed waves."""
        df = _make_df(
            [
                (1_000, 100.0, 105.0, 98.0, 103.0, 1.0, 0.5),
                (2_000, 102.0, 110.0, 100.0, 108.0, 1.0, 0.3),
                (3_000, 107.0, 109.0, 101.0, 106.0, 1.0, 0.1),
            ]
        )
        h = hydrate(df)
        assert h.wave_registry == ()
        wave = h.get_current_wave()
        assert wave is not None
        assert len(wave.candles) == 3


# ---------------------------------------------------------------------------
# Wave properties
# ---------------------------------------------------------------------------


class TestHydrateWaveProperties:
    def test_wave_extremes(self) -> None:
        """Wave 0 (rows 0-1): high=110 at row 1, low=98 at row 0."""
        h = hydrate(_make_df())
        w0 = h.wave_registry[0]
        assert w0.high.high == 110.0
        assert w0.high.open_time == 2_000
        assert w0.low.low == 98.0
        assert w0.low.open_time == 1_000

    def test_wave_candle_count(self) -> None:
        h = hydrate(_make_df())
        assert len(h.wave_registry[0].candles) == 2  # rows 0-1
        assert len(h.wave_registry[1].candles) == 2  # rows 2-3
        assert len(h.wave_registry[2].candles) == 2  # rows 4-5

    def test_formation_bar_index(self) -> None:
        """formation_bar_index = index of the flip candle (first of next wave)."""
        h = hydrate(_make_df())
        assert h.wave_registry[0].formation_bar_index == 2  # flip at row 2
        assert h.wave_registry[1].formation_bar_index == 4  # flip at row 4
        assert h.wave_registry[2].formation_bar_index == 6  # flip at row 6

    def test_extremum_indices(self) -> None:
        h = hydrate(_make_df())
        w0 = h.wave_registry[0]
        # Wave 0 spans rows 0-1. high at row 1, low at row 0.
        assert w0.high_idx == 1
        assert w0.low_idx == 0

    def test_high_since_computed(self) -> None:
        """Wave 2 (up) should have high_since computed from backward scan."""
        h = hydrate(_make_df())
        w2 = h.wave_registry[2]
        assert w2.side == "up"
        # high_since should be > 0 (HCO at some position, prior waves scanned)
        assert w2.high_since >= 0

    def test_pullback_computed_for_second_wave(self) -> None:
        """Wave 1 (down) has a prior top (wave 0) → pullback is not None."""
        h = hydrate(_make_df())
        w1 = h.wave_registry[1]
        assert w1.side == "down"
        assert w1.pullback is not None
        assert w1.pullback.price_diff < 0  # down-wave: negative

    def test_pullback_none_for_first_wave(self) -> None:
        """First wave has no prior opposite wave."""
        h = hydrate(_make_df())
        assert h.wave_registry[0].pullback is None

    def test_highest_close_or_open(self) -> None:
        h = hydrate(_make_df())
        w0 = h.wave_registry[0]
        # Row 0: max(100, 103) = 103. Row 1: max(102, 108) = 108. HCO = row 1.
        assert w0.highest_close_or_open.open_time == 2_000

    def test_lowest_close_or_open(self) -> None:
        h = hydrate(_make_df())
        w0 = h.wave_registry[0]
        # Row 0: min(100, 103) = 100. Row 1: min(102, 108) = 102. LCO = row 0.
        assert w0.lowest_close_or_open.open_time == 1_000


# ---------------------------------------------------------------------------
# Forming wave state
# ---------------------------------------------------------------------------


class TestHydrateFormingWave:
    def test_forming_wave_exists(self) -> None:
        h = hydrate(_make_df())
        wave = h.get_current_wave()
        assert wave is not None
        assert wave.side == "down"
        assert wave.id == "forming-3"  # 3 confirmed waves → next id = 3

    def test_forming_wave_candles(self) -> None:
        h = hydrate(_make_df())
        wave = h.get_current_wave()
        assert wave is not None
        assert len(wave.candles) == 1
        assert wave.candles[0].open_time == 7_000

    def test_total_candles_registered(self) -> None:
        h = hydrate(_make_df())
        assert h.total_candles_registered == 7

    def test_subsequent_register_candle_deduplicates(self) -> None:
        """Re-registering the last candle is silently ignored."""
        h = hydrate(_make_df())
        last_row = ROWS[-1]
        h.register_candle(_make_candle(last_row))
        assert h.total_candles_registered == 7  # no change

    def test_subsequent_register_candle_extends(self) -> None:
        """A new candle after hydrate is processed normally."""
        h = hydrate(_make_df())
        h.register_candle(
            Candle(
                open_time=8_000,
                open=95.0,
                high=98.0,
                low=93.0,
                close=94.0,
                volume=1.0,
                histogram_value=-0.5,
            ),
        )
        assert h.total_candles_registered == 8
        wave = h.get_current_wave()
        assert wave is not None
        assert len(wave.candles) == 2

    def test_subsequent_register_candle_can_flip(self) -> None:
        """A flip after hydrate confirms the forming wave."""
        h = hydrate(_make_df())
        assert len(h.wave_registry) == 3  # before

        h.register_candle(
            Candle(
                open_time=8_000,
                open=95.0,
                high=98.0,
                low=93.0,
                close=94.0,
                volume=1.0,
                histogram_value=0.3,
            ),
        )
        assert len(h.wave_registry) == 4
        assert h.wave_registry[3].side == "down"
        assert h.wave_registry[3].id == "w-3"


# ---------------------------------------------------------------------------
# Parity with incremental path
# ---------------------------------------------------------------------------


class TestHydrateParity:
    """Feed the same data through both construction paths and compare."""

    def test_wave_registry_matches(self) -> None:
        """hydrate(df) produces the same wave_registry as register_candle."""
        # Incremental path
        h_inc = MarketStructureHelper()
        for row in ROWS:
            h_inc.register_candle(_make_candle(row))

        # Hydrate path
        h_hyd = hydrate(_make_df())

        # Compare confirmed wave registries
        assert len(h_inc.wave_registry) == len(h_hyd.wave_registry)
        for w_inc, w_hyd in zip(h_inc.wave_registry, h_hyd.wave_registry, strict=True):
            assert w_inc == w_hyd, f"Wave mismatch: {w_inc.id}"

    def test_forming_wave_matches(self) -> None:
        """Forming wave from both paths is identical."""
        h_inc = MarketStructureHelper()
        for row in ROWS:
            h_inc.register_candle(_make_candle(row))

        h_hyd = hydrate(_make_df())

        w_inc = h_inc.get_current_wave()
        w_hyd = h_hyd.get_current_wave()
        assert w_inc is not None
        assert w_hyd is not None
        assert w_inc == w_hyd

    def test_parity_with_varied_wave_lengths(self) -> None:
        """Parity test with waves of different lengths (1, 3, 2, 1 candles)."""
        rows = [
            (1_000, 100.0, 105.0, 98.0, 103.0, 1.0, 0.5),  # wave 0 (up, 1 candle)
            (2_000, 107.0, 109.0, 101.0, 106.0, 1.0, -0.1),  # flip
            (3_000, 104.0, 106.0, 97.0, 98.0, 1.0, -0.3),  # wave 1 (down, 3 candles)
            (4_000, 99.0, 103.0, 96.0, 101.0, 1.0, -0.2),
            (5_000, 100.0, 108.0, 99.0, 107.0, 1.0, 0.4),  # flip
            (6_000, 102.0, 110.0, 100.0, 109.0, 1.0, 0.5),  # wave 2 (up, 2 candles)
            (7_000, 106.0, 107.0, 95.0, 96.0, 1.0, -0.4),  # flip
            (8_000, 95.0, 98.0, 93.0, 94.0, 1.0, -0.3),  # forming (down, 1 candle)
        ]

        h_inc = MarketStructureHelper()
        for row in rows:
            h_inc.register_candle(_make_candle(row))

        h_hyd = hydrate(_make_df(rows))

        assert h_inc.wave_registry == h_hyd.wave_registry

        w_inc = h_inc.get_current_wave()
        w_hyd = h_hyd.get_current_wave()
        assert w_inc is not None
        assert w_hyd is not None
        assert w_inc == w_hyd


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestHydrateEdgeCases:
    def test_zero_histogram_is_up_side(self) -> None:
        """Histogram = 0.0 is classified as 'up', matching _sign_flipped."""
        df = _make_df(
            [
                (1_000, 100.0, 105.0, 98.0, 103.0, 1.0, 0.0),
                (2_000, 102.0, 110.0, 100.0, 108.0, 1.0, -0.1),  # flip
            ]
        )
        h = hydrate(df)
        assert len(h.wave_registry) == 1
        assert h.wave_registry[0].side == "up"

    def test_max_waves_respected(self) -> None:
        """Hydrate with max_waves=2 evicts old waves."""
        rows = []
        sign = 0.5
        for i in range(10):
            rows.append((1_000 * (i + 1), 100.0, 105.0, 98.0, 103.0, 1.0, sign))
            sign = -sign

        h = hydrate(_make_df(rows), max_waves=2)
        assert len(h.wave_registry) <= 2

    def test_non_standard_index_handled(self) -> None:
        """DataFrame with non-0-based index is handled via reset_index."""
        df = _make_df()
        df.index = range(10, 17)  # non-standard index
        h = hydrate(df)
        assert len(h.wave_registry) == 3

    def test_directional_arrays_populated(self) -> None:
        """get_last_top / get_last_bottom work after hydrate."""
        h = hydrate(_make_df())
        assert h.get_last_top() is not None
        assert h.get_last_top() is h.wave_registry[2]  # last up wave
        assert h.get_last_bottom() is not None
        assert h.get_last_bottom() is h.wave_registry[1]  # last down wave
