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

    # ------------------------------------------------------------------
    # Read-only state accessors
    # ------------------------------------------------------------------

    @property
    def wave_registry(self) -> tuple[Wave, ...]:
        """All confirmed waves, oldest to newest, as an immutable snapshot."""
        return tuple(self._wave_registry)

    def get_last_top(self) -> Wave | None:
        """Return the most recently confirmed up-wave, or ``None`` during warm-up."""
        return self._top_waves[-1] if self._top_waves else None

    def get_last_bottom(self) -> Wave | None:
        """Return the most recently confirmed down-wave, or ``None`` during warm-up."""
        return self._bottom_waves[-1] if self._bottom_waves else None

    def get_current_wave(self) -> Wave | None:
        """Return the in-flight wave currently being constructed, or ``None``.

        Stage 2 always returns ``None``. Stage 4 builds the forming wave
        from the buffered candles of the current (unfinished) swing leg.
        """
        return None

    # ------------------------------------------------------------------
    # Ingest API (stub — real implementation in Stages 3-4)
    # ------------------------------------------------------------------

    def register_candle(
        self,
        candle: Candle,
        *,
        histogram_value: float,
    ) -> None:
        """Ingest a single candle and advance internal state.

        Args:
            candle: The OHLCV candle to ingest.
            histogram_value: Current value of the configured histogram
                indicator at this candle. Carried separately so ``Candle``
                stays minimal and indicator-agnostic — the helper can be
                driven by any sign-flipping oscillator (TSI, MACD, custom)
                by wiring the right column here at the call site.

        Raises:
            NotImplementedError: Always, in Stage 2. Sign-flip detection
                arrives in Stage 3, wave construction in Stage 4.
        """
        raise NotImplementedError("register_candle is implemented in stages 3-4 of the port")
