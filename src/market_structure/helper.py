"""MarketStructureHelper — swing/trend/zone detector for OHLCV frames.

Two construction paths converge on the same instance state:

- ``register_candle(candle, histogram_value=...)`` — incremental, used live
- ``hydrate(df)`` — vectorized bulk path, used in backtest

See ``docs/porting-market-structure-helper.md`` for the architectural
rationale (option (c) hybrid).
"""

from market_structure.types import Candle, Wave


class MarketStructureHelper:
    """Stateful market-structure detector.

    A fresh helper has no waves and returns ``None`` from every ``get_*``
    accessor. Waves are pushed into the registry by either the incremental
    ingest path (``register_candle``) or the bulk ``hydrate`` path —
    landing in Stages 3 and 8 respectively.
    """

    def __init__(
        self,
        *,
        histogram_key: str = "tsi_hist",
        max_waves: int = 200,
    ) -> None:
        """Initialize an empty helper.

        Args:
            histogram_key: Name of the DataFrame column (or row attribute)
                whose sign-flips demarcate wave boundaries. Matches the TS
                original (``tsi_histogram``), shortened here to align with
                the pandas column naming used in this project.
            max_waves: Maximum number of confirmed waves to retain in the
                registry. Older waves are evicted FIFO when exceeded.
                Actual eviction lands in Stage 4 alongside wave construction.
        """
        self.histogram_key: str = histogram_key
        self.max_waves: int = max_waves

        # Mutable state. These MUST be assigned on ``self`` inside ``__init__``
        # — never declared at class body with a default like
        # ``_wave_registry: list[Wave] = []`` — that would create a single
        # list shared across every instance (a classic Python footgun).
        # ``test_helper_skeleton.TestInstanceIsolation`` locks this in.
        self._wave_registry: list[Wave] = []
        self._top_waves: list[Wave] = []
        self._bottom_waves: list[Wave] = []

        # Forming-wave buffer. Accumulates candles since the last histogram
        # sign-flip; cleared on each flip. Stage 4 will turn a finalized
        # buffer into a ``Wave`` and push it into ``_wave_registry``.
        self._wave_candles: list[Candle] = []

        # Dedup + sign-flip bookkeeping. All ``None`` during warm-up so we
        # can distinguish "never seen a candle" from "just saw one at t=0"
        # without overloading falsy values. A ``0.0`` histogram reading is
        # perfectly valid and MUST NOT be confused with "no prior value".
        self._last_registered_open_time: int | None = None
        self._previous_histogram_value: float | None = None
        self._total_candles_registered: int = 0

    # ------------------------------------------------------------------
    # Read-only state accessors
    # ------------------------------------------------------------------

    @property
    def wave_registry(self) -> tuple[Wave, ...]:
        """All confirmed waves, oldest to newest, as an immutable snapshot."""
        return tuple(self._wave_registry)

    @property
    def total_candles_registered(self) -> int:
        """Count of unique candles seen by ``register_candle`` (post-dedup).

        Duplicate ``open_time`` values — which Freqtrade emits on every
        tick while a candle is still forming — are counted exactly once.
        """
        return self._total_candles_registered

    def get_last_top(self) -> Wave | None:
        """Return the most recently confirmed up-wave, or ``None`` during warm-up."""
        return self._top_waves[-1] if self._top_waves else None

    def get_last_bottom(self) -> Wave | None:
        """Return the most recently confirmed down-wave, or ``None`` during warm-up."""
        return self._bottom_waves[-1] if self._bottom_waves else None

    def get_current_wave(self) -> Wave | None:
        """Return the in-flight wave currently being constructed, or ``None``.

        Stages 2 and 3 always return ``None`` — even while ``_wave_candles``
        accumulates, no ``Wave`` object exists yet. Stage 4 builds the
        forming wave from the buffered candles of the current swing leg.
        """
        return None

    # ------------------------------------------------------------------
    # Ingest API
    # ------------------------------------------------------------------

    def register_candle(
        self,
        candle: Candle,
        *,
        histogram_value: float,
    ) -> None:
        """Ingest a single candle and advance internal state.

        Stage 3 behavior: dedup by ``open_time``, then — if the histogram
        has crossed the zero line since the last call — clear the forming
        wave buffer. Finally, append the new candle to the buffer and
        remember the current histogram value for next time.

        Wave construction (turning a finalized buffer into a ``Wave`` and
        pushing it onto the registry) lands in Stage 4. Until then,
        ``wave_registry`` stays empty and ``get_current_wave()`` returns
        ``None`` even while ``_wave_candles`` fills up.

        Args:
            candle: The OHLCV candle to ingest.
            histogram_value: Current value of the configured histogram
                indicator at this candle. Carried separately so ``Candle``
                stays minimal and indicator-agnostic — the helper can be
                driven by any sign-flipping oscillator (TSI, MACD, custom)
                by wiring the right column here at the call site.
        """
        # Freqtrade re-emits the forming candle on every tick, so the same
        # ``open_time`` can arrive many times. Skip silently — the caller
        # does not need to dedup on its side.
        if candle.open_time == self._last_registered_open_time:
            return
        self._last_registered_open_time = candle.open_time
        self._total_candles_registered += 1

        # Use ``is not None`` (not ``if prev``): a 0.0 reading is a valid
        # histogram value, but ``if prev`` would treat it as "unseen".
        prev = self._previous_histogram_value
        if prev is not None and self._sign_flipped(prev, histogram_value):
            # Prior wave is done. Drop its candles — Stage 4 will instead
            # finalize them into a Wave at this point.
            self._wave_candles.clear()

        self._wave_candles.append(candle)
        self._previous_histogram_value = histogram_value

    @staticmethod
    def _sign_flipped(prev: float, curr: float) -> bool:
        """Return True iff ``prev`` and ``curr`` straddle the zero line.

        Matches the TS original's semantics: ``>= 0`` is classified as
        the "up" side and ``< 0`` as the "down" side. A flip occurs iff
        the up/down classification of ``curr`` differs from ``prev``.

        Exact zero intentionally sits on the "up" side so that a reading
        of ``0.0`` followed by any negative value registers as a flip.
        """
        return (prev >= 0) != (curr >= 0)
