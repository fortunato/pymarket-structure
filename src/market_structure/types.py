"""Public dataclasses for the market-structure library.

All types here are immutable (``frozen=True``) and use ``slots=True`` to
avoid per-instance ``__dict__`` overhead — we create thousands of Wave
instances during a backtest hydrate.
"""

from dataclasses import dataclass, field
from typing import Literal

Direction = Literal["up", "down"]


@dataclass(frozen=True, slots=True)
class Candle:
    """Minimal OHLCV candle shape the helper consumes.

    Mirrors the fields MarketStructureHelper reads off each row. Extensions
    (ATR, TSI histogram) live in the DataFrame column; the helper pulls the
    ``tsi_hist`` value out at construction time, not off the Candle itself.
    """

    open_time: int
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass(frozen=True, slots=True)
class Pullback:
    """Retracement metrics from the prior opposite-direction wave."""

    length: int
    breakout_level: float
    price_diff: float
    correction_factor: float | None
    atr_factor: float | None


@dataclass(frozen=True, slots=True)
class Wave:
    """One confirmed swing leg in the market-structure sequence.

    Immutable. All extremum candles are stored alongside their row index
    into the originating frame so ``_determine_high_since`` does not have
    to rescan the wave's candles to find them.
    """

    id: str
    side: Direction
    formation_bar_index: int
    high: Candle
    low: Candle
    highest_close: Candle
    lowest_close: Candle
    highest_close_or_open: Candle
    lowest_close_or_open: Candle
    high_idx: int
    low_idx: int
    highest_close_or_open_idx: int
    lowest_close_or_open_idx: int
    high_since: int = 0
    low_since: int = 0
    pullback: Pullback | None = None
    candles: tuple[Candle, ...] = field(default_factory=tuple)


@dataclass(frozen=True, slots=True)
class Zone:
    """Support or resistance zone anchored to a specific wave."""

    range: tuple[float, float]
    anchor_wave_id: str
    overlapping_low_wave_ids: tuple[str, ...]
    overlapping_high_wave_ids: tuple[str, ...]
    is_double: bool
    side: Direction
