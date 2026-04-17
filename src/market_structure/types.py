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
    """OHLCV candle with the histogram reading that drives wave detection.

    The ``histogram_value`` field carries the sign-flipping oscillator
    value (TSI, MACD, etc.) for this bar. Which DataFrame column maps
    here is controlled by ``histogram_key`` on the helper — the Candle
    itself is agnostic to the indicator name.
    """

    open_time: int  # epoch ms — unique identifier, used for dedup
    open: float
    high: float
    low: float
    close: float
    volume: float
    histogram_value: float = 0.0  # oscillator reading at this bar


@dataclass(frozen=True, slots=True)
class Pullback:
    """Retracement metrics from the prior opposite-direction wave."""

    length: int  # bar count from prior wave's extreme to this wave's extreme
    breakout_level: float  # close-or-open of the prior wave's extreme candle
    price_diff: float  # signed: positive for up-waves, negative for down-waves
    correction_factor: float | None  # fraction of prior run retraced (0..1+), None during warm-up
    atr_factor: float | None  # retracement depth in ATR multiples, None if no ATR available


@dataclass(frozen=True, slots=True)
class Wave:
    """One confirmed swing leg in the market-structure sequence.

    Immutable. All extremum candles are stored alongside their row index
    into the originating frame so ``_determine_high_since`` does not have
    to rescan the wave's candles to find them.
    """

    id: str  # "w-0", "w-1", ... for confirmed; "forming-N" for in-flight
    side: Direction  # "up" = histogram >= 0; "down" = histogram < 0
    formation_bar_index: int  # bar index of the flip candle that confirmed this wave

    # Extremum candles — the candle objects where each extreme occurred.
    high: Candle  # candle with the highest high in this wave
    low: Candle  # candle with the lowest low in this wave
    highest_close: Candle  # candle with the highest close
    lowest_close: Candle  # candle with the lowest close
    highest_close_or_open: Candle  # HCO — candle with the highest max(close, open)
    lowest_close_or_open: Candle  # LCO — candle with the lowest min(close, open)

    # Row indices into the source DataFrame for each extremum above.
    high_idx: int
    low_idx: int
    highest_close_or_open_idx: int  # HCO index
    lowest_close_or_open_idx: int  # LCO index

    # Backward scan results — how far back (in bars) to a prior wave that
    # exceeded this wave's HCO (for up-waves) or LCO (for down-waves).
    # Used by pick_long_term_top/bottom to identify significant swings.
    high_since: int = 0  # bars since a prior wave had a higher HCO (up-waves only)
    low_since: int = 0  # bars since a prior wave had a lower LCO (down-waves only)

    pullback: Pullback | None = None  # retracement from prior opposite wave; None during warm-up
    candles: tuple[Candle, ...] = field(default_factory=tuple)  # all candles in this wave


@dataclass(frozen=True, slots=True)
class Zone:
    """Support or resistance zone anchored to a specific wave."""

    range: tuple[float, float]  # (low_price, high_price) — body-anchored inclusive bounds
    wick_range: tuple[float, float]  # (low_price, high_price) — wick-based extrema
    anchor_wave_id: str  # wave that defines the primary zone boundary
    overlapping_low_wave_ids: tuple[str, ...]  # down-waves whose body ranges overlap this zone
    overlapping_high_wave_ids: tuple[str, ...]  # up-waves whose body ranges overlap this zone
    is_double: bool  # double bottom (support) or double top (resistance)
    side: Direction  # "down" = support zone; "up" = resistance zone


@dataclass(frozen=True, slots=True)
class ZoneLifecycleState:
    """Tracks a zone's lifecycle through break/retest/flip transitions.

    Immutable — each state transition creates a new instance.
    """

    state: str  # "intact", "broken", "retested", "flipped", "failed_retest"
    break_bar_index: int | None = None
    retest_bar_indices: tuple[int, ...] = ()
    retest_count: int = 0
    flip_bar_index: int | None = None
    failed_retest_bar_index: int | None = None


@dataclass(frozen=True, slots=True)
class LongTermSwing:
    """Result of ``pick_long_term_top`` / ``pick_long_term_bottom``.

    Carries an ``age`` (candle distance from the current bar to the
    extreme) and the ``wave`` that holds the significant extreme.
    """

    age: int  # candle distance from the current bar to the extreme
    wave: Wave  # the wave containing the significant extreme
