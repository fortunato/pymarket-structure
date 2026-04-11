"""MarketStructureHelper — swing/trend/zone detector for OHLCV frames.

Two construction paths converge on the same instance state:

- ``register_candle(candle)`` — incremental, used live
- ``hydrate(df)`` — vectorized bulk path, used in backtest

See ``docs/porting-market-structure-helper.md`` for the architectural
rationale (option (c) hybrid).
"""

from market_structure.types import Candle, Direction, Pullback, Wave


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

        # Wave construction bookkeeping.
        self._next_wave_id: int = 0
        self._wave_start_index: int = 0

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

    def get_last_top(self, *, include_forming_wave: bool = False) -> Wave | None:
        """Return the most recently confirmed up-wave, or ``None`` during warm-up.

        When ``include_forming_wave`` is True and the forming wave is an
        up-wave, it is returned instead — matching the TS semantics where
        ``getLastTop(true)`` treats the forming wave as "last".
        """
        if include_forming_wave:
            current = self.get_current_wave()
            if current is not None and current.side == "up":
                return current
        return self._top_waves[-1] if self._top_waves else None

    def get_last_bottom(self, *, include_forming_wave: bool = False) -> Wave | None:
        """Return the most recently confirmed down-wave, or ``None`` during warm-up.

        When ``include_forming_wave`` is True and the forming wave is a
        down-wave, it is returned instead.
        """
        if include_forming_wave:
            current = self.get_current_wave()
            if current is not None and current.side == "down":
                return current
        return self._bottom_waves[-1] if self._bottom_waves else None

    def get_previous_top(self, *, include_forming_wave: bool = False) -> Wave | None:
        """Return the second-to-last confirmed up-wave, or ``None``.

        When ``include_forming_wave`` is True and the forming wave is an
        up-wave, the forming wave is treated as "last" — so "previous"
        returns the most recent *confirmed* up-wave (one slot back).
        """
        if include_forming_wave:
            current = self.get_current_wave()
            if current is not None and current.side == "up":
                return self._top_waves[-1] if self._top_waves else None
        return self._top_waves[-2] if len(self._top_waves) >= 2 else None

    def get_previous_bottom(self, *, include_forming_wave: bool = False) -> Wave | None:
        """Return the second-to-last confirmed down-wave, or ``None``.

        When ``include_forming_wave`` is True and the forming wave is a
        down-wave, the forming wave is treated as "last" — so "previous"
        returns the most recent *confirmed* down-wave.
        """
        if include_forming_wave:
            current = self.get_current_wave()
            if current is not None and current.side == "down":
                return self._bottom_waves[-1] if self._bottom_waves else None
        return self._bottom_waves[-2] if len(self._bottom_waves) >= 2 else None

    def get_current_wave(self) -> Wave | None:
        """Return the in-flight wave currently being constructed, or ``None``.

        Builds a fresh ``Wave`` from ``_wave_candles`` on each call. The
        forming wave uses a temporary ID (``"forming-N"``) that will become
        ``"w-N"`` once a sign-flip confirms it. Calling this method never
        increments the wave counter — the ID is reserved, not consumed.

        Returns ``None`` only when no candles have been registered yet.
        """
        if not self._wave_candles:
            return None
        # Invariant: _previous_histogram_value is set whenever _wave_candles
        # is non-empty — both are updated together at the end of register_candle.
        assert self._previous_histogram_value is not None
        side: Direction = "up" if self._previous_histogram_value >= 0 else "down"
        return self._construct_wave(side, wave_id=f"forming-{self._next_wave_id}")

    # ------------------------------------------------------------------
    # Ingest API
    # ------------------------------------------------------------------

    def register_candle(self, candle: Candle) -> None:
        """Ingest a single candle and advance internal state.

        Dedup by ``open_time``, then — if the histogram has crossed the
        zero line since the last call — finalize the forming wave and
        start a new one. Finally, append the candle to the forming buffer.

        The histogram value is read from ``candle.histogram_value``.
        Which DataFrame column maps there is controlled by
        ``histogram_key`` on the helper — the caller wires the right
        column when constructing the Candle.
        """
        # Freqtrade re-emits the forming candle on every tick, so the same
        # ``open_time`` can arrive many times. Skip silently — the caller
        # does not need to dedup on its side.
        if candle.open_time == self._last_registered_open_time:
            return
        self._last_registered_open_time = candle.open_time
        self._total_candles_registered += 1

        histogram_value = candle.histogram_value

        # Use ``is not None`` (not ``if prev``): a 0.0 reading is a valid
        # histogram value, but ``if prev`` would treat it as "unseen".
        prev = self._previous_histogram_value
        if prev is not None and self._sign_flipped(prev, histogram_value):
            side: Direction = "up" if prev >= 0 else "down"
            wave_id = f"w-{self._next_wave_id}"
            self._next_wave_id += 1
            wave = self._construct_wave(side, wave_id=wave_id)
            self._push_wave(wave)
            self._wave_candles.clear()
            self._wave_start_index = self._total_candles_registered - 1

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

    # ------------------------------------------------------------------
    # Wave construction
    # ------------------------------------------------------------------

    def _construct_wave(self, side: Direction, *, wave_id: str) -> Wave:
        """Build a ``Wave`` from the current ``_wave_candles`` buffer.

        Finds the six extremum candles via ``max`` / ``min`` with a
        ``key=`` function — the Pythonic replacement for the TS
        ``.reduce()`` pattern. No explicit seed needed: the buffer is
        guaranteed non-empty when we reach a sign-flip.

        For the four extremes that carry a stored index (``high_idx``,
        ``low_idx``, ``highest_close_or_open_idx``,
        ``lowest_close_or_open_idx``), we use ``enumerate`` to track
        the buffer position and translate to a global candle index via
        ``_wave_start_index + offset``.

        The ``wave_id`` is passed in by the caller — ``register_candle``
        assigns ``"w-N"`` (permanent, counter-incrementing) while
        ``get_current_wave`` assigns ``"forming-N"`` (temporary, no
        counter side-effect).
        """
        candles = self._wave_candles
        base = self._wave_start_index

        high_pos, high_c = max(enumerate(candles), key=lambda ic: ic[1].high)
        low_pos, low_c = min(enumerate(candles), key=lambda ic: ic[1].low)
        highest_close_c = max(candles, key=lambda c: c.close)
        lowest_close_c = min(candles, key=lambda c: c.close)
        hco_pos, hco_c = max(enumerate(candles), key=lambda ic: max(ic[1].close, ic[1].open))
        lco_pos, lco_c = min(enumerate(candles), key=lambda ic: min(ic[1].close, ic[1].open))

        high_since = self._determine_high_since(hco_c, hco_pos) if side == "up" else 0
        low_since = self._determine_low_since(lco_c, lco_pos) if side == "down" else 0

        if side == "up":
            pullback = self._determine_pullback_from_bottom(hco_c, hco_pos)
        else:
            pullback = self._determine_pullback_from_top(lco_c, lco_pos)

        return Wave(
            id=wave_id,
            side=side,
            formation_bar_index=self._total_candles_registered - 1,
            high=high_c,
            low=low_c,
            highest_close=highest_close_c,
            lowest_close=lowest_close_c,
            highest_close_or_open=hco_c,
            lowest_close_or_open=lco_c,
            high_idx=base + high_pos,
            low_idx=base + low_pos,
            highest_close_or_open_idx=base + hco_pos,
            lowest_close_or_open_idx=base + lco_pos,
            high_since=high_since,
            low_since=low_since,
            pullback=pullback,
            candles=tuple(candles),
        )

    # ------------------------------------------------------------------
    # Backward scans
    # ------------------------------------------------------------------

    def _determine_high_since(self, hco_candle: Candle, hco_pos: int) -> int:
        """Count bars backward from the HCO extreme to the last time price exceeded it.

        Starting from ``hco_pos`` (the position of the highest-close-or-open
        candle within the forming wave's buffer), walks backward through
        ``_wave_registry`` newest-to-oldest. For each prior wave, compares
        its ``highest_close_or_open`` level against the forming wave's.

        Returns the total candle distance — used by ``pick_long_term_top``
        (Stage 11) to identify significant swing highs. Only meaningful
        for ``"up"`` waves.
        """
        forming_top = max(hco_candle.close, hco_candle.open)
        candle_count = hco_pos

        for wave in reversed(self._wave_registry):
            older_top = max(
                wave.highest_close_or_open.close,
                wave.highest_close_or_open.open,
            )
            if older_top > forming_top:
                local_idx = next(
                    i for i, c in enumerate(wave.candles) if c is wave.highest_close_or_open
                )
                return candle_count + len(wave.candles) - 1 - local_idx
            candle_count += len(wave.candles)

        return candle_count

    def _determine_low_since(self, lco_candle: Candle, lco_pos: int) -> int:
        """Count bars backward from the LCO extreme to the last time price went lower.

        Mirror of ``_determine_high_since`` for down-waves: walks backward
        looking for a prior wave whose ``lowest_close_or_open`` is *below*
        the forming wave's level.
        """
        forming_bottom = min(lco_candle.close, lco_candle.open)
        candle_count = lco_pos

        for wave in reversed(self._wave_registry):
            older_bottom = min(
                wave.lowest_close_or_open.close,
                wave.lowest_close_or_open.open,
            )
            if older_bottom < forming_bottom:
                local_idx = next(
                    i for i, c in enumerate(wave.candles) if c is wave.lowest_close_or_open
                )
                return candle_count + len(wave.candles) - 1 - local_idx
            candle_count += len(wave.candles)

        return candle_count

    # ------------------------------------------------------------------
    # Pullback computation
    # ------------------------------------------------------------------

    def _determine_pullback_from_bottom(self, hco_candle: Candle, hco_pos: int) -> Pullback | None:
        """Compute pullback metrics for an up-wave from the last confirmed bottom.

        Measures how far price has risen from the bottom wave's
        ``lowest_close_or_open`` to the forming wave's
        ``highest_close_or_open``. The ``correction_factor`` expresses
        this move as a fraction of the prior run (previous top → bottom).

        Returns ``None`` during warm-up when no confirmed bottom exists.
        """
        bottom = self.get_last_bottom()
        if bottom is None:
            return None

        # Distance from end of bottom wave to its LCO candle.
        lco_local = next(
            i for i, c in enumerate(bottom.candles) if c is bottom.lowest_close_or_open
        )
        bottom_candle_distance = len(bottom.candles) - 1 - lco_local

        top_close_or_open = max(hco_candle.close, hco_candle.open)
        bottom_close_or_open = min(
            bottom.lowest_close_or_open.close,
            bottom.lowest_close_or_open.open,
        )

        previous_top = self._get_top_before(bottom)
        correction_factor: float | None = None
        if previous_top is not None:
            previous_top_high = max(
                previous_top.highest_close_or_open.close,
                previous_top.highest_close_or_open.open,
            )
            denominator = previous_top_high - bottom_close_or_open
            if denominator != 0:
                correction_factor = (top_close_or_open - bottom_close_or_open) / denominator

        return Pullback(
            length=hco_pos + bottom_candle_distance,
            breakout_level=bottom_close_or_open,
            price_diff=top_close_or_open - bottom_close_or_open,
            correction_factor=correction_factor,
            atr_factor=None,
        )

    def _determine_pullback_from_top(self, lco_candle: Candle, lco_pos: int) -> Pullback | None:
        """Compute pullback metrics for a down-wave from the last confirmed top.

        Mirror of ``_determine_pullback_from_bottom``: measures how far
        price has fallen from the top wave's ``highest_close_or_open`` to
        the forming wave's ``lowest_close_or_open``.
        """
        top = self.get_last_top()
        if top is None:
            return None

        # Distance from end of top wave to its HCO candle.
        hco_local = next(i for i, c in enumerate(top.candles) if c is top.highest_close_or_open)
        top_candle_distance = len(top.candles) - 1 - hco_local

        top_close_or_open = max(
            top.highest_close_or_open.close,
            top.highest_close_or_open.open,
        )
        bottom_close_or_open = min(lco_candle.close, lco_candle.open)

        previous_bottom = self._get_bottom_before(top)
        correction_factor: float | None = None
        if previous_bottom is not None:
            previous_bottom_low = min(
                previous_bottom.lowest_close_or_open.close,
                previous_bottom.lowest_close_or_open.open,
            )
            denominator = top_close_or_open - previous_bottom_low
            if denominator != 0:
                correction_factor = (top_close_or_open - bottom_close_or_open) / denominator

        return Pullback(
            length=lco_pos + top_candle_distance,
            breakout_level=top_close_or_open,
            price_diff=bottom_close_or_open - top_close_or_open,
            correction_factor=correction_factor,
            atr_factor=None,
        )

    # ------------------------------------------------------------------
    # Wave registry lookups
    # ------------------------------------------------------------------

    def _get_wave_index(self, wave: Wave) -> int | None:
        """Return the index of ``wave`` in the registry, or ``None``.

        Uses identity (``is``) not equality — the wave objects in the
        registry are the canonical instances, and callers always pass
        references obtained from ``get_last_top()`` etc.
        """
        for i, w in enumerate(self._wave_registry):
            if w is wave:
                return i
        return None

    def _get_top_before(self, wave: Wave) -> Wave | None:
        """Return the up-wave immediately preceding ``wave`` in the registry.

        Waves alternate in the registry (up, down, up, …). If ``wave``
        is a down-wave at index *i*, the top before it sits at *i - 1*.
        If ``wave`` is itself an up-wave, the previous up-wave is two
        slots back at *i - 2*.

        Returns ``None`` when the computed index is out of bounds —
        unlike JS, Python's negative indices wrap, so we guard explicitly.
        """
        idx = self._get_wave_index(wave)
        if idx is None:
            return None
        top_idx = idx - 1 if wave.side == "down" else idx - 2
        if top_idx < 0:
            return None
        return self._wave_registry[top_idx]

    def _get_bottom_before(self, wave: Wave) -> Wave | None:
        """Return the down-wave immediately preceding ``wave`` in the registry.

        Mirror of ``_get_top_before``: if ``wave`` is an up-wave, the
        bottom before it is one slot back; if down, two slots back.
        """
        idx = self._get_wave_index(wave)
        if idx is None:
            return None
        bottom_idx = idx - 1 if wave.side == "up" else idx - 2
        if bottom_idx < 0:
            return None
        return self._wave_registry[bottom_idx]

    # ------------------------------------------------------------------
    # Wave registry management
    # ------------------------------------------------------------------

    def _push_wave(self, wave: Wave) -> None:
        """Append ``wave`` to the registry and the appropriate directional array.

        Shared by the incremental path (``register_candle``) and the bulk
        ``hydrate`` path (Stage 8). After pushing, evict oldest waves if
        the registry exceeds ``max_waves``.
        """
        self._wave_registry.append(wave)
        if wave.side == "up":
            self._top_waves.append(wave)
        else:
            self._bottom_waves.append(wave)
        self._evict_old_waves()

    def _evict_old_waves(self) -> None:
        """Remove the oldest waves when the registry exceeds ``max_waves``.

        FIFO eviction via ``list.pop(0)`` — O(n) per eviction but
        ``max_waves`` is typically 200, so the constant is negligible.
        ``collections.deque`` would give O(1) popleft but complicates
        indexed access elsewhere; not worth it until profiling says so.

        Evicted waves are also removed from the head of the matching
        directional array (``_top_waves`` or ``_bottom_waves``) to keep
        all three lists consistent.
        """
        while len(self._wave_registry) > self.max_waves:
            evicted = self._wave_registry.pop(0)
            if evicted.side == "up" and self._top_waves and self._top_waves[0] is evicted:
                self._top_waves.pop(0)
            elif evicted.side == "down" and self._bottom_waves and self._bottom_waves[0] is evicted:
                self._bottom_waves.pop(0)

    # ------------------------------------------------------------------
    # Wave comparison helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _hco_value(wave: Wave) -> float:
        """Extract the numeric HCO level from a wave.

        HCO = highest close-or-open: ``max(close, open)`` of the candle
        stored in ``wave.highest_close_or_open``. This is the price level
        used when comparing tops across waves.
        """
        c = wave.highest_close_or_open
        return max(c.close, c.open)

    @staticmethod
    def _lco_value(wave: Wave) -> float:
        """Extract the numeric LCO level from a wave.

        LCO = lowest close-or-open: ``min(close, open)`` of the candle
        stored in ``wave.lowest_close_or_open``. This is the price level
        used when comparing bottoms across waves.
        """
        c = wave.lowest_close_or_open
        return min(c.close, c.open)

    @staticmethod
    def made_higher_high(last: Wave, previous: Wave) -> bool:
        """True if ``last`` wave's HCO exceeds ``previous`` wave's HCO."""
        return MarketStructureHelper._hco_value(last) > MarketStructureHelper._hco_value(previous)

    @staticmethod
    def made_higher_low(last: Wave, previous: Wave) -> bool:
        """True if ``last`` wave's LCO exceeds ``previous`` wave's LCO."""
        return MarketStructureHelper._lco_value(last) > MarketStructureHelper._lco_value(previous)

    @staticmethod
    def made_lower_low(last: Wave, previous: Wave) -> bool:
        """True if ``last`` wave's LCO is below ``previous`` wave's LCO."""
        return MarketStructureHelper._lco_value(last) < MarketStructureHelper._lco_value(previous)

    @staticmethod
    def made_lower_high(last: Wave, previous: Wave) -> bool:
        """True if ``last`` wave's HCO is below ``previous`` wave's HCO."""
        return MarketStructureHelper._hco_value(last) < MarketStructureHelper._hco_value(previous)

    @staticmethod
    def is_diverging(last: Wave, previous: Wave) -> bool:
        """True if price made a new extreme but histogram momentum did not.

        For up-waves (bearish divergence): price made a higher close but the
        peak histogram reading within the wave was lower.

        For down-waves (bullish divergence): price made a lower close but the
        trough histogram reading within the wave was higher (less negative).

        Histogram extremes are computed from ``wave.candles`` at query time,
        matching the TS ``getHighestHistogramReading`` /
        ``getLowestHistogramReading`` pattern.
        """
        if last.side == "up":
            last_hist_high = max(c.histogram_value for c in last.candles)
            prev_hist_high = max(c.histogram_value for c in previous.candles)
            return (
                last.highest_close.close > previous.highest_close.close
                and last_hist_high < prev_hist_high
            )
        last_hist_low = min(c.histogram_value for c in last.candles)
        prev_hist_low = min(c.histogram_value for c in previous.candles)
        return (
            last.lowest_close.close < previous.lowest_close.close and last_hist_low > prev_hist_low
        )

    # ------------------------------------------------------------------
    # Between scans
    # ------------------------------------------------------------------

    def made_lower_low_between(self, wave: Wave, preceding: Wave) -> bool:
        """True if any wave between ``preceding`` and ``wave`` has a lower low.

        Scans all registry entries between the two given waves (exclusive)
        checking if any intermediate wave's ``low.low`` undercuts either
        endpoint. Used by zone detection to rule out double-bottom patterns
        when an intervening wave made a deeper low.
        """
        wave_idx = self._get_wave_index(wave)
        preceding_idx = self._get_wave_index(preceding)
        if wave_idx is None or preceding_idx is None:
            return False
        for i in range(wave_idx - 1, preceding_idx, -1):
            w = self._wave_registry[i]
            if w.low.low < wave.low.low or w.low.low < preceding.low.low:
                return True
        return False

    def made_higher_high_between(self, wave: Wave, preceding: Wave) -> bool:
        """True if any wave between ``preceding`` and ``wave`` has a higher high.

        Mirror of ``made_lower_low_between`` for tops — used by zone
        detection to rule out double-top patterns.
        """
        wave_idx = self._get_wave_index(wave)
        preceding_idx = self._get_wave_index(preceding)
        if wave_idx is None or preceding_idx is None:
            return False
        for i in range(wave_idx - 1, preceding_idx, -1):
            w = self._wave_registry[i]
            if w.high.high > wave.high.high or w.high.high > preceding.high.high:
                return True
        return False

    # ------------------------------------------------------------------
    # Trend state
    # ------------------------------------------------------------------

    def is_trending_up(self) -> bool:
        """True if market structure confirms an uptrend.

        Requires four confirmed waves (two tops, two bottoms) showing
        higher highs and higher lows. The forming wave must not have
        broken structure — if it's a down-wave, its low must stay above
        the last bottom's LCO level.
        """
        last_top = self.get_last_top()
        last_bottom = self.get_last_bottom()
        previous_top = self.get_previous_top()
        previous_bottom = self.get_previous_bottom()
        current = self.get_current_wave()

        if (
            last_top is None
            or last_bottom is None
            or previous_top is None
            or previous_bottom is None
            or current is None
        ):
            return False

        return (
            self.made_higher_high(last_top, previous_top)
            and self.made_higher_low(last_bottom, previous_bottom)
            and (current.side == "up" or current.low.low > self._lco_value(last_bottom))
        )

    def is_trending_down(self) -> bool:
        """True if market structure confirms a downtrend.

        Requires four confirmed waves (two tops, two bottoms) showing
        lower highs and lower lows. The forming wave must not have
        broken structure — if it's an up-wave, its high must stay below
        the last top's HCO level.
        """
        last_top = self.get_last_top()
        last_bottom = self.get_last_bottom()
        previous_top = self.get_previous_top()
        previous_bottom = self.get_previous_bottom()
        current = self.get_current_wave()

        if (
            last_top is None
            or last_bottom is None
            or previous_top is None
            or previous_bottom is None
            or current is None
        ):
            return False

        return (
            self.made_lower_high(last_top, previous_top)
            and self.made_lower_low(last_bottom, previous_bottom)
            and (current.side == "down" or current.high.high < self._hco_value(last_top))
        )
