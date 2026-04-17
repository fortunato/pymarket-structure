"""MarketStructureHelper — swing/trend/zone detector for OHLCV frames.

Two construction paths converge on the same instance state:

- ``register_candle(candle)`` — incremental, used live
- ``hydrate(df)`` — vectorized bulk path, used in backtest

See ``docs/porting-market-structure-helper.md`` for the architectural
rationale (option (c) hybrid).
"""

import numpy as np

from market_structure.types import Candle, Direction, LongTermSwing, Pullback, Wave, Zone


def _double_pattern_tolerance(
    atr_arr: np.ndarray | None,
    anchor: Wave,
    side: Direction,
    atr_multiple: float,
    pct_fallback: float,
) -> float:
    """Return the price-proximity tolerance for double-pattern qualification.

    Reads ATR at the anchor wave's **own swing bar** — ``anchor.low_idx`` for
    support (``side="down"``), ``anchor.high_idx`` for resistance
    (``side="up"``). Do NOT use ``anchor.formation_bar_index``: that is the
    flip candle of the *next* wave (see ``hydrate.py:131-132`` and
    ``types.py:55``), which can equal ``len(atr_arr)`` for the most recent
    in-progress wave and would silently force every freshest-anchor call
    into the percentage fallback — the opposite of the intended behaviour.

    Falls back to ``pct_fallback * anchor.low.low`` (support) or
    ``pct_fallback * anchor.high.high`` (resistance) when ``atr_arr`` is
    ``None``, the per-anchor lookup is out of bounds, or the ATR value is
    not finite and positive (including negative values). The fallback is
    intentionally tighter than a typical ATR-path tolerance so the
    fallback→ATR transition is monotone-conservative.
    """
    anchor_price = anchor.low.low if side == "down" else anchor.high.high

    if atr_arr is not None:
        idx = anchor.low_idx if side == "down" else anchor.high_idx
        if 0 <= idx < len(atr_arr):
            atr_val = float(atr_arr[idx])
            if np.isfinite(atr_val) and atr_val > 0:
                return atr_multiple * atr_val

    return pct_fallback * anchor_price


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

        # Zone cache — (params_key, cached_result), single-entry per zone type,
        # invalidated on _push_wave.
        self._zone_cache_support: tuple[tuple[object, ...], list[Zone]] | None = None
        self._zone_cache_resistance: tuple[tuple[object, ...], list[Zone]] | None = None

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
        up-wave, it is returned instead.
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

        ``>= 0`` is classified as the "up" side and ``< 0`` as the
        "down" side. A flip occurs iff
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
        self._invalidate_zone_cache()
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

    def _invalidate_zone_cache(self) -> None:
        """Clear both zone caches. Called from ``_push_wave``."""
        self._zone_cache_support = None
        self._zone_cache_resistance = None

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

        Histogram extremes are computed from ``wave.candles`` at query time.
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

    # ------------------------------------------------------------------
    # Zone range helpers
    # ------------------------------------------------------------------

    @staticmethod
    def get_bottom_range(wave: Wave) -> tuple[float, float]:
        """Body-anchored price range for a bottom (support) zone.

        The anchor candle is the one with the lowest *close* in the wave
        (close = settlement, the only consensus price per Auction Market
        Theory — Steidlmayer / Dalton, *Mind Over Markets*).  The zone
        spans the anchor candle's body:
        ``(min(open, close), max(open, close))``.

        Use :meth:`get_bottom_wick_range` for wick-based geometry
        (Wyckoff stop placement beyond the spring wick).
        """
        anchor = wave.lowest_close
        return (min(anchor.open, anchor.close), max(anchor.open, anchor.close))

    @staticmethod
    def get_top_range(wave: Wave) -> tuple[float, float]:
        """Body-anchored price range for a top (resistance) zone.

        The anchor candle is the one with the highest *close* in the wave.
        The zone spans the anchor candle's body:
        ``(min(open, close), max(open, close))``.

        Use :meth:`get_top_wick_range` for wick-based geometry
        (Wyckoff stop placement beyond the wick extreme).
        """
        anchor = wave.highest_close
        return (min(anchor.open, anchor.close), max(anchor.open, anchor.close))

    @staticmethod
    def get_bottom_wick_range(wave: Wave) -> tuple[float, float]:
        """Wick-based price range for a bottom (support) zone.

        Returns ``(wave.low.low, min(anchor.close, anchor.open))``
        where anchor is ``wave.lowest_close``.  This is the wick-extended
        geometry — the four ``ms_*_zone_wick_*`` DataFrame columns carry
        these values so consumers can place stops beyond the spring wick
        (textbook Wyckoff stop placement).
        """
        anchor = wave.lowest_close
        return (wave.low.low, min(anchor.close, anchor.open))

    @staticmethod
    def get_top_wick_range(wave: Wave) -> tuple[float, float]:
        """Wick-based price range for a top (resistance) zone.

        Returns ``(max(anchor.close, anchor.open), wave.high.high)``
        where anchor is ``wave.highest_close``.  This is the wick-extended
        geometry — the four ``ms_*_zone_wick_*`` DataFrame columns carry
        these values so consumers can place stops beyond the wick extreme.
        """
        anchor = wave.highest_close
        return (max(anchor.close, anchor.open), wave.high.high)

    @staticmethod
    def range_overlaps(a: tuple[float, float], b: tuple[float, float]) -> bool:
        """True if two ``(min, max)`` price ranges overlap.

        Three cases: ``a[0]`` falls within ``b``, ``a[1]`` falls within
        ``b``, or ``a`` completely contains ``b``.
        """
        return (b[0] <= a[0] <= b[1]) or (b[0] <= a[1] <= b[1]) or (a[0] <= b[0] and a[1] >= b[1])

    def _assert_alternation(self, anchor: Wave, preceding: Wave) -> None:
        """Alternation invariant for the double-pattern body.

        TSI-driven wave emission guarantees that same-side waves are
        separated by at least one opposite-side wave in
        ``_wave_registry``. An assertion failure here indicates a bug
        somewhere in the emission path, not a recoverable runtime
        condition. Under ``python -O`` the assert is stripped — this
        is intentional.
        """
        assert anchor.side == preceding.side, (
            f"double-pattern pair with mismatched sides: "
            f"{preceding.id}={preceding.side} vs {anchor.id}={anchor.side}"
        )
        reg = self._wave_registry
        pre_pos = next((i for i, w in enumerate(reg) if w is preceding), -1)
        # Forming-wave anchors aren't in ``_wave_registry``; treat the
        # anchor position as one past the end for those.
        anc_pos = next((i for i, w in enumerate(reg) if w is anchor), len(reg))
        assert any(reg[j].side != anchor.side for j in range(pre_pos + 1, anc_pos)), (
            f"registry alternation broken: no opposite-side wave between "
            f"{preceding.id} (pos {pre_pos}) and {anchor.id} (pos {anc_pos})"
        )

    # ------------------------------------------------------------------
    # Zone detection
    # ------------------------------------------------------------------

    def get_support_zones(
        self,
        include_forming_wave: bool = False,
        double_bottom_proximity: int = 2,
        filter_if_not_overlapping: bool = False,
        only_include_most_recent_zone: bool = True,
        *,
        atr_arr: np.ndarray | None = None,
        tolerance_atr_multiple: float = 0.3,
        tolerance_pct_fallback: float = 0.004,
    ) -> list[Zone]:
        """Identify support zones from overlapping bottom-wave body ranges.

        Walks bottom waves newest-to-oldest. For each, builds a
        body-anchored range via ``get_bottom_range`` (the anchor
        candle's body — accepted price per Auction Market Theory) and
        collects older bottoms whose body ranges overlap (the
        ``overlapping_low_wave_ids`` geometry list).

        A separate, price-proximity test decides whether the candidate
        is a double bottom: the anchor's low must be within
        ``tolerance`` of a preceding same-side wave's low and no
        intervening low may undercut the pair (``made_lower_low_between``).
        When a candidate is qualified AND the two bodies overlap, the
        zone range is extended to include the preceding body bottom.
        Wick extrema are unioned across the pair into
        ``Zone.wick_range`` for Wyckoff stop placement.

        Args:
            include_forming_wave: Include the in-flight wave if it's a
                down-wave.
            double_bottom_proximity: How many preceding same-side waves
                to consider for double-bottom labelling. Default ``2``
                so the canonical W-pattern with one intermediate
                non-violating higher low is admitted out of the box.
            filter_if_not_overlapping: If True, drop zones with zero
                overlapping lows (no confirmation from older bottoms).
            only_include_most_recent_zone: If True, skip a wave whose
                range overlaps an already-registered zone.
            atr_arr: Optional ATR values aligned 1:1 with the DataFrame
                the helper was hydrated from. When provided, the
                double-bottom qualification predicate uses
                ``tolerance_atr_multiple * atr_arr[anchor.low_idx]`` as
                the price-proximity tolerance; when ``None`` (or when
                the per-anchor lookup yields NaN / zero / out-of-bounds),
                the ``tolerance_pct_fallback`` path is used instead.

                **Cache-key note**: the zone cache keys on
                ``id(atr_arr)`` (object identity), not content equality.
                This is correct within a single ``attach_market_structure``
                call tree (the same array is passed to both zone methods).
                Callers who reuse a helper across multiple separate ATR
                computations in the same process — and allow the first
                array to be garbage-collected before the second call —
                may hit a stale cache entry because CPython can reuse the
                freed memory address. In that scenario, construct a fresh
                helper or explicitly invalidate the cache by calling with
                a different ``include_forming_wave`` toggle.
            tolerance_atr_multiple: Multiplier applied to the anchor
                wave's formation-bar ATR to compute the "same level"
                tolerance band. Default 0.3.
            tolerance_pct_fallback: Fraction of the anchor's low used as
                tolerance when the ATR value is missing, zero, or out of
                bounds. Default 0.004 (0.4 %).

        The double-pattern body is also guarded by an alternation
        ``assert`` (``_assert_alternation``) that verifies the
        preceding and anchor waves are same-sided and have at least one
        opposite-side wave between them in ``_wave_registry``. The assert
        is a correctness tripwire for the TSI-driven emission path — under
        ``python -O`` it is stripped; if you rely on this invariant as a
        production guardrail, do not run with ``-O``.
        """
        params_key = (
            include_forming_wave,
            double_bottom_proximity,
            filter_if_not_overlapping,
            only_include_most_recent_zone,
            id(atr_arr),
            tolerance_atr_multiple,
            tolerance_pct_fallback,
        )
        if self._zone_cache_support is not None and self._zone_cache_support[0] == params_key:
            return self._zone_cache_support[1]

        # Build working list: newest bottom first.
        down_waves: list[Wave] = list(reversed(self._bottom_waves))
        if include_forming_wave:
            current = self.get_current_wave()
            if current is not None and current.side == "down":
                down_waves.insert(0, current)

        up_waves: list[Wave] = list(reversed(self._top_waves))

        zones: list[Zone] = []

        for idx, wave in enumerate(down_waves):
            current_range = list(self.get_bottom_range(wave))
            current_wick_range = list(self.get_bottom_wick_range(wave))
            overlapping_lows: list[str] = []
            is_double = False

            # Skip if this range overlaps an already-registered zone.
            if only_include_most_recent_zone and any(
                self.range_overlaps((current_range[0], current_range[1]), z.range) for z in zones
            ):
                continue

            # Match against all preceding (older) bottoms.
            for preceding_idx, preceding_wave in enumerate(down_waves[idx + 1 :]):
                bottom_range = self.get_bottom_range(preceding_wave)
                bodies_overlap = self.range_overlaps(
                    bottom_range, (current_range[0], current_range[1])
                )

                if bodies_overlap:
                    overlapping_lows.append(preceding_wave.id)

                # Tolerance-based qualification predicate.
                if preceding_idx < double_bottom_proximity and not self.made_lower_low_between(
                    wave, preceding_wave
                ):
                    self._assert_alternation(wave, preceding_wave)
                    tolerance = _double_pattern_tolerance(
                        atr_arr,
                        wave,
                        "down",
                        tolerance_atr_multiple,
                        tolerance_pct_fallback,
                    )
                    if abs(preceding_wave.low.low - wave.low.low) <= tolerance:
                        is_double = True
                        if bodies_overlap and bottom_range[0] < current_range[0]:
                            current_range[0] = bottom_range[0]
                        # Union wick extrema across all qualifying predecessors
                        # so the Wyckoff stop sits below the deepest spring.
                        prec_wick = self.get_bottom_wick_range(preceding_wave)
                        if prec_wick[0] < current_wick_range[0]:
                            current_wick_range[0] = prec_wick[0]
                        if prec_wick[1] > current_wick_range[1]:
                            current_wick_range[1] = prec_wick[1]

            # Find overlapping top waves.
            overlapping_highs: list[str] = [
                w.id
                for w in up_waves
                if self.range_overlaps(self.get_top_range(w), (current_range[0], current_range[1]))
            ]

            zones.append(
                Zone(
                    range=(current_range[0], current_range[1]),
                    wick_range=(current_wick_range[0], current_wick_range[1]),
                    anchor_wave_id=wave.id,
                    overlapping_low_wave_ids=tuple(overlapping_lows),
                    overlapping_high_wave_ids=tuple(overlapping_highs),
                    is_double=is_double,
                    side="down",
                )
            )

        result = [
            z
            for z in zones
            if not filter_if_not_overlapping or len(z.overlapping_low_wave_ids) >= 1
        ]
        self._zone_cache_support = (params_key, result)
        return result

    def get_resistance_zones(
        self,
        include_forming_wave: bool = False,
        double_top_proximity: int = 2,
        filter_if_not_overlapping: bool = False,
        only_include_most_recent_zone: bool = True,
        *,
        atr_arr: np.ndarray | None = None,
        tolerance_atr_multiple: float = 0.3,
        tolerance_pct_fallback: float = 0.004,
    ) -> list[Zone]:
        """Identify resistance zones from overlapping top-wave body ranges.

        Mirror of ``get_support_zones`` for tops / resistance. See that
        method's docstring for the full parameter contract (including
        the double-pattern tolerance semantics, the raised
        ``double_top_proximity`` default, the alternation ``assert``,
        and the ``python -O`` caveat). Differences: the ATR lookup
        here uses ``anchor.high_idx`` instead of ``anchor.low_idx``,
        and the percentage fallback is applied against
        ``anchor.high.high``.
        """
        params_key = (
            include_forming_wave,
            double_top_proximity,
            filter_if_not_overlapping,
            only_include_most_recent_zone,
            id(atr_arr),
            tolerance_atr_multiple,
            tolerance_pct_fallback,
        )
        if self._zone_cache_resistance is not None and self._zone_cache_resistance[0] == params_key:
            return self._zone_cache_resistance[1]

        up_waves: list[Wave] = list(reversed(self._top_waves))
        if include_forming_wave:
            current = self.get_current_wave()
            if current is not None and current.side == "up":
                up_waves.insert(0, current)

        down_waves: list[Wave] = list(reversed(self._bottom_waves))

        zones: list[Zone] = []

        for idx, wave in enumerate(up_waves):
            current_range = list(self.get_top_range(wave))
            current_wick_range = list(self.get_top_wick_range(wave))
            overlapping_highs: list[str] = []
            is_double = False

            if only_include_most_recent_zone and any(
                self.range_overlaps((current_range[0], current_range[1]), z.range) for z in zones
            ):
                continue

            for preceding_idx, preceding_wave in enumerate(up_waves[idx + 1 :]):
                top_range = self.get_top_range(preceding_wave)
                bodies_overlap = self.range_overlaps(
                    top_range, (current_range[0], current_range[1])
                )

                if bodies_overlap:
                    overlapping_highs.append(preceding_wave.id)

                # Tolerance-based qualification predicate.
                # Mirrors the support-side flow in ``get_support_zones``.
                if preceding_idx < double_top_proximity and not self.made_higher_high_between(
                    wave, preceding_wave
                ):
                    self._assert_alternation(wave, preceding_wave)
                    tolerance = _double_pattern_tolerance(
                        atr_arr,
                        wave,
                        "up",
                        tolerance_atr_multiple,
                        tolerance_pct_fallback,
                    )
                    if abs(preceding_wave.high.high - wave.high.high) <= tolerance:
                        is_double = True
                        if bodies_overlap and top_range[1] > current_range[1]:
                            current_range[1] = top_range[1]
                        # Union wick extrema across all qualifying predecessors.
                        prec_wick = self.get_top_wick_range(preceding_wave)
                        if prec_wick[0] < current_wick_range[0]:
                            current_wick_range[0] = prec_wick[0]
                        if prec_wick[1] > current_wick_range[1]:
                            current_wick_range[1] = prec_wick[1]

            overlapping_lows: list[str] = [
                w.id
                for w in down_waves
                if self.range_overlaps(
                    self.get_bottom_range(w), (current_range[0], current_range[1])
                )
            ]

            zones.append(
                Zone(
                    range=(current_range[0], current_range[1]),
                    wick_range=(current_wick_range[0], current_wick_range[1]),
                    anchor_wave_id=wave.id,
                    overlapping_low_wave_ids=tuple(overlapping_lows),
                    overlapping_high_wave_ids=tuple(overlapping_highs),
                    is_double=is_double,
                    side="up",
                )
            )

        result = [
            z
            for z in zones
            if not filter_if_not_overlapping or len(z.overlapping_high_wave_ids) >= 1
        ]
        self._zone_cache_resistance = (params_key, result)
        return result

    # ------------------------------------------------------------------
    # Long-term swing pickers
    # ------------------------------------------------------------------

    def pick_long_term_top(self, high_since: int = 100, max_age: int = 50) -> LongTermSwing | None:
        """Find a significant long-term swing high in the recent registry.

        Walks ``_wave_registry`` in reverse, accumulating ``age`` (candle
        count from the current bar). Returns the first up-wave whose
        ``high_since`` meets the threshold, or ``None`` if no such wave
        exists within ``max_age`` bars.
        """
        current = self.get_current_wave()
        age = len(current.candles) if current is not None else 0

        for wave in reversed(self._wave_registry):
            if wave.side == "up" and wave.high_since >= high_since:
                # Find the HCO candle's position within the wave (from the end).
                local_idx = next(
                    i
                    for i, c in enumerate(reversed(wave.candles))
                    if c.open_time == wave.highest_close_or_open.open_time
                )
                age += local_idx
                return LongTermSwing(age=age, wave=wave)

            age += len(wave.candles)
            if age > max_age:
                return None

        return None

    def pick_long_term_bottom(
        self, low_since: int = 100, max_age: int = 50
    ) -> LongTermSwing | None:
        """Find a significant long-term swing low in the recent registry.

        Mirror of ``pick_long_term_top`` for down-waves / ``low_since``.
        """
        current = self.get_current_wave()
        age = len(current.candles) if current is not None else 0

        for wave in reversed(self._wave_registry):
            if wave.side == "down" and wave.low_since >= low_since:
                local_idx = next(
                    i
                    for i, c in enumerate(reversed(wave.candles))
                    if c.open_time == wave.lowest_close_or_open.open_time
                )
                age += local_idx
                return LongTermSwing(age=age, wave=wave)

            age += len(wave.candles)
            if age > max_age:
                return None

        return None

    # ------------------------------------------------------------------
    # Wave lookup by ID
    # ------------------------------------------------------------------

    def get_wave_by_id(self, wave_id: str) -> Wave | None:
        """Return the wave with the given ID, or ``None``.

        Linear scan — the registry is small (capped at ``max_waves``).
        Needed by downstream zone utilities.
        """
        for wave in self._wave_registry:
            if wave.id == wave_id:
                return wave
        return None
