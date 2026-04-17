# Market Structure Columns -- Freqtrade Integration Reference

## Overview

The `attach_market_structure` wrapper projects up to 67 columns onto a Freqtrade DataFrame. Each column is prefixed with `ms_` in the DataFrame (e.g., the short name `wave_side` becomes `ms_wave_side`).

All columns use `pd.NA` (nullable) for boolean/integer fields and `np.nan` for float fields when insufficient wave history exists (warm-up period).

---

## Column Reference

### Wave Identity

| DataFrame Column | Short Name | dtype | Updates | Description |
|---|---|---|---|---|
| `ms_wave_side` | `wave_side` | `object` (str) | per wave | Direction of the current wave segment: `"up"` when the histogram is non-negative, `"down"` when negative. Directly reflects the sign of the oscillator histogram driving wave detection. |
| `ms_wave_id` | `wave_id` | `object` (str) | per wave | Unique identifier for the wave this bar belongs to. Confirmed waves use `"w-0"`, `"w-1"`, etc. The still-forming (unconfirmed) wave uses `"forming-N"` where N is the next wave counter. |

### Price Levels & Swing Significance

| DataFrame Column | Short Name | dtype | Updates | Description |
|---|---|---|---|---|
| `ms_last_top_price` | `last_top_price` | `float64` | per boundary | HCO (highest close-or-open) price of the most recently confirmed up-wave. This is `max(close, open)` of the candle with the highest close-or-open value in that wave. Represents the last significant swing high body level. |
| `ms_last_bottom_price` | `last_bottom_price` | `float64` | per boundary | LCO (lowest close-or-open) price of the most recently confirmed down-wave. This is `min(close, open)` of the candle with the lowest close-or-open value in that wave. Represents the last significant swing low body level. |
| `ms_high_since` | `high_since` | `Int32` | per boundary | Number of bars backward from the most recent up-wave's HCO extreme to the last prior wave that had a higher HCO. Larger values indicate the current swing high is more significant (no prior wave exceeded it for a long time). |
| `ms_low_since` | `low_since` | `Int32` | per boundary | Number of bars backward from the most recent down-wave's LCO extreme to the last prior wave that had a lower LCO. Larger values indicate the current swing low is more significant. |
| `ms_bars_since_last_top` | `bars_since_last_top` | `Int32` | per bar | Running counter of bars since the most recent confirmed swing high. Resets to 0 on the boundary bar where the swing is confirmed. NaN before the first confirmed swing. |
| `ms_bars_since_last_bottom` | `bars_since_last_bottom` | `Int32` | per bar | Running counter of bars since the most recent confirmed swing low. Same reset semantics as `bars_since_last_top`. |

### Trend Structure

| DataFrame Column | Short Name | dtype | Updates | Description |
|---|---|---|---|---|
| `ms_made_higher_high` | `made_higher_high` | `boolean` | per boundary | True when the most recent confirmed up-wave's HCO exceeds the previous confirmed up-wave's HCO. Indicates an expanding swing high -- one of the two conditions for an uptrend. |
| `ms_made_higher_low` | `made_higher_low` | `boolean` | per boundary | True when the most recent confirmed down-wave's LCO exceeds the previous confirmed down-wave's LCO. Indicates a rising swing low -- the second condition for an uptrend. |
| `ms_made_lower_high` | `made_lower_high` | `boolean` | per boundary | True when the most recent confirmed up-wave's HCO is below the previous confirmed up-wave's HCO. Indicates a contracting swing high -- one of the two conditions for a downtrend. |
| `ms_made_lower_low` | `made_lower_low` | `boolean` | per boundary | True when the most recent confirmed down-wave's LCO is below the previous confirmed down-wave's LCO. Indicates a falling swing low -- the second condition for a downtrend. |
| `ms_is_trending_up` | `is_trending_up` | `bool` | per bar | True when market structure confirms an uptrend: HH + HL from last two confirmed tops/bottoms. During down-waves, dynamically revoked bar-by-bar if the running low drops to or below the last bottom's LCO level (structure break). |
| `ms_is_trending_down` | `is_trending_down` | `bool` | per bar | True when market structure confirms a downtrend: LH + LL from last two confirmed tops/bottoms. During up-waves, dynamically revoked bar-by-bar if the running high reaches or exceeds the last top's HCO level (structure break). |
| `ms_structure_break_level` | `structure_break_level` | `float64` | per boundary | The price level whose breach would invalidate the current trend. In an uptrend: the LCO of the last confirmed bottom that maintained the higher-low sequence. In a downtrend: the HCO of the last confirmed top that maintained the lower-high sequence. NaN when no valid trend sequence exists. |
| `ms_structure_break_confirmed` | `structure_break_confirmed` | `boolean` | per bar | True on the bar where close crosses the `structure_break_level`. Fires once per structure break event. |
| `ms_trend_wave_count` | `trend_wave_count` | `Int32` | per boundary | Number of consecutive trend-confirming wave pairs (HH+HL for uptrend, LH+LL for downtrend) counted backward from the current position. Higher counts indicate more mature trends. 0 when no trend sequence exists. |
| `ms_trend_duration` | `trend_duration` | `Int32` | per bar | Number of bars since the current trend began. Measured from the bar where `is_trending_up` or `is_trending_down` transitioned from False to True. NaN when not trending. Resets on trend break. |
| `ms_three_push_up` | `three_push_up` | `boolean` | per boundary | True when the last 3 up-waves exhibit a three-push exhaustion pattern: diminishing amplitude, each making a new higher high, and convergent spacing (each extension smaller than the prior). Classic trend exhaustion signal. |
| `ms_three_push_down` | `three_push_down` | `boolean` | per boundary | True when the last 3 down-waves exhibit a three-push exhaustion pattern with analogous conditions in the bearish direction. |

### Wave Metrics

| DataFrame Column | Short Name | dtype | Updates | Description |
|---|---|---|---|---|
| `ms_wave_length` | `wave_length` | `Int32` | per boundary | Number of candles (bars) in the most recently confirmed wave. Useful for gauging momentum exhaustion or comparing wave durations across a trend. |
| `ms_wave_count` | `wave_count` | `Int32` | per boundary | Total number of confirmed waves in the registry at this point. Increases monotonically during backtest. Useful for ensuring enough structure exists before applying wave-based conditions. |
| `ms_wave_amplitude` | `wave_amplitude` | `float64` | per boundary | ATR-normalized amplitude of the most recently confirmed wave: `(wave_high - wave_low) / atr`. NaN if no confirmed wave or ATR is unavailable. |
| `ms_wave_slope` | `wave_slope` | `float64` | per boundary | Rate of amplitude per bar: `wave_amplitude / wave_length`. Higher values indicate faster price movement. NaN if no confirmed wave. |
| `ms_forming_wave_high` | `forming_wave_high` | `float64` | per bar | Running maximum of bar highs within the current wave segment. Resets at each wave boundary (histogram sign-flip). |
| `ms_forming_wave_low` | `forming_wave_low` | `float64` | per bar | Running minimum of bar lows within the current wave segment. Resets at each wave boundary. |
| `ms_wave_volume` | `wave_volume` | `float64` | per boundary | Total volume across all candles in the most recently confirmed wave. |
| `ms_wave_volume_ratio` | `wave_volume_ratio` | `float64` | per boundary | Ratio of current wave volume to the previous same-direction wave's volume. Values above 1.0 indicate increasing participation; below 1.0 suggests exhaustion. NaN when fewer than 2 same-direction waves exist. |
| `ms_wave_amplitude_ratio` | `wave_amplitude_ratio` | `float64` | per boundary | Ratio of the current wave's ATR-normalized amplitude to the previous same-direction wave's amplitude. Values below 1.0 with declining volume suggest trend exhaustion. NaN when insufficient waves. |

### Pullback Metrics

| DataFrame Column | Short Name | dtype | Updates | Description |
|---|---|---|---|---|
| `ms_pullback_length` | `pullback_length` | `Int32` | per boundary | Bar count from the prior opposite-direction wave's extreme to this wave's extreme. Measures how many bars the retracement took. |
| `ms_pullback_correction_factor` | `pullback_correction_factor` | `float64` | per boundary | Fraction of the prior run that was retraced, expressed as a ratio. A value of 1.0 means the retracement fully recovered the prior move; values above 1.0 indicate extension beyond. `NaN` during warm-up (fewer than 3 waves of the relevant sides). |
| `ms_pullback_breakout_level` | `pullback_breakout_level` | `float64` | per boundary | The close-or-open price of the prior opposite wave's extreme candle. For an up-wave, this is the LCO of the last bottom -- the level price must reclaim. For a down-wave, the HCO of the last top. |
| `ms_pullback_price_diff` | `pullback_price_diff` | `float64` | per boundary | Signed price distance of the pullback. Positive for up-waves (price rose from bottom), negative for down-waves (price fell from top). The absolute value is the raw retracement magnitude in price units. |
| `ms_pullback_atr_factor` | `pullback_atr_factor` | `float64` | per boundary | ATR-normalized pullback depth: `abs(pullback_price_diff) / atr_at_formation`. Measures retracement significance relative to volatility. |

### Divergence

| DataFrame Column | Short Name | dtype | Updates | Description |
|---|---|---|---|---|
| `ms_bearish_divergence` | `bearish_divergence` | `boolean` | per boundary | True when the most recent confirmed up-wave made a higher close than the previous up-wave, but the peak histogram reading within the wave was lower. Classic bearish divergence signal: price momentum is weakening at higher prices. |
| `ms_bullish_divergence` | `bullish_divergence` | `boolean` | per boundary | True when the most recent confirmed down-wave made a lower close than the previous down-wave, but the trough histogram reading within the wave was higher (less negative). Classic bullish divergence signal: selling momentum is weakening at lower prices. |

### Support / Resistance Zones

| DataFrame Column | Short Name | dtype | Updates | Description |
|---|---|---|---|---|
| `ms_support_zone_low` | `support_zone_low` | `float64` | per boundary | Lower bound of the nearest support zone. Body-anchored: `min(anchor.open, anchor.close)` where anchor is the candle with the lowest close in the wave (Auction Market Theory — close is settlement/consensus price). Extended to include a preceding same-side body bottom **only when that preceding wave's body range actually overlaps** the anchor's. Price-level proximity alone (i.e. a qualified double bottom with disjoint bodies) does *not* extend the zone — qualification and geometry are decoupled. |
| `ms_support_zone_high` | `support_zone_high` | `float64` | per boundary | Upper bound of the nearest support zone. Body-anchored: `max(anchor.open, anchor.close)` of the anchor candle. |
| `ms_support_is_double` | `support_is_double` | `boolean` | per boundary | True when the nearest support zone qualifies as a double bottom. A pair of lows qualifies when (a) no intervening wave made a deeper low and (b) the absolute price distance is within `tolerance_atr_multiple × atr[anchor.low_idx]` (default 0.3 × ATR) — or within `tolerance_pct_fallback × anchor.low` (default 0.4 %) when ATR is unavailable. The lookback horizon is `double_bottom_proximity` (default 2) preceding same-side waves, which admits the canonical W-pattern with one intermediate non-violating swing. |
| `ms_support_overlap_count` | `support_overlap_count` | `Int32` | per boundary | Number of older down-waves whose bottom body ranges overlap the nearest support zone. Higher counts suggest the zone has been tested more frequently. |
| `ms_support_zone_anchor_time` | `support_zone_anchor_time` | `Int64` | per boundary | Epoch milliseconds of the anchor candle's open_time for the nearest support zone. Useful for tracking zone age. |
| `ms_support_zone_wick_low` | `support_zone_wick_low` | `float64` | per boundary | Lower wick bound of the nearest support zone: the wave's absolute low (`wave.low.low`). Useful for Wyckoff stop placement beyond the spring wick. For a marubozu anchor (body == range), collapses to `support_zone_low`. |
| `ms_support_zone_wick_high` | `support_zone_wick_high` | `float64` | per boundary | Upper wick bound of the nearest support zone: `min(anchor.open, anchor.close)` — the body bottom of the anchor candle. Equals `support_zone_low` for the primary anchor; may differ when unioned across a double-bottom pair. |
| `ms_resistance_zone_low` | `resistance_zone_low` | `float64` | per boundary | Lower bound of the nearest resistance zone. Body-anchored: `min(anchor.open, anchor.close)` of the anchor candle with the highest close in the wave. |
| `ms_resistance_zone_high` | `resistance_zone_high` | `float64` | per boundary | Upper bound of the nearest resistance zone. Body-anchored: `max(anchor.open, anchor.close)`. Extended to include a preceding same-side body top **only when that preceding wave's body range actually overlaps** the anchor's. See the support mirror for the body-vs-qualification decoupling. |
| `ms_resistance_is_double` | `resistance_is_double` | `boolean` | per boundary | True when the nearest resistance zone qualifies as a double top. Mirror of `ms_support_is_double`: qualification is price-proximity based (ATR × multiple, falling back to percentage of `anchor.high`), no intervening higher high is allowed, and `double_top_proximity` (default 2) sets the lookback horizon — admits the M-pattern with one intermediate non-violating lower high. |
| `ms_resistance_zone_wick_low` | `resistance_zone_wick_low` | `float64` | per boundary | Lower wick bound of the nearest resistance zone: `max(anchor.open, anchor.close)` — the body top of the anchor candle. |
| `ms_resistance_zone_wick_high` | `resistance_zone_wick_high` | `float64` | per boundary | Upper wick bound of the nearest resistance zone: the wave's absolute high (`wave.high.high`). Useful for stop placement beyond the wick extreme. |
| `ms_resistance_overlap_count` | `resistance_overlap_count` | `Int32` | per boundary | Number of older up-waves whose top body ranges overlap the nearest resistance zone. Higher counts indicate more frequent tests of the level. |
| `ms_resistance_zone_anchor_time` | `resistance_zone_anchor_time` | `Int64` | per boundary | Epoch milliseconds of the anchor candle's open_time for the nearest resistance zone. |
| `ms_zone_quality_support` | `zone_quality_support` | `float64` | per boundary | Composite quality score [0, 10] for the nearest support zone. Factors: overlap count, double-bottom status, ATR-relative width, recency decay, and touch count. Higher is stronger. |
| `ms_zone_quality_resistance` | `zone_quality_resistance` | `float64` | per boundary | Composite quality score [0, 10] for the nearest resistance zone. Same formula as support quality. |

### Zone Lifecycle Events

| DataFrame Column | Short Name | dtype | Updates | Description |
|---|---|---|---|---|
| `ms_zone_break_support` | `zone_break_support` | `boolean` | per bar | True on the bar where price breaks through a support zone (close or wick crosses zone boundary, depending on `retest_mode`). |
| `ms_zone_break_resistance` | `zone_break_resistance` | `boolean` | per bar | True on the bar where price breaks through a resistance zone. |
| `ms_zone_retest_support` | `zone_retest_support` | `boolean` | per bar | True on the bar where price returns to a previously broken support zone (now acting as resistance). |
| `ms_zone_retest_resistance` | `zone_retest_resistance` | `boolean` | per bar | True on the bar where price returns to a previously broken resistance zone (now acting as support). |
| `ms_zone_retest_count_support` | `zone_retest_count_support` | `Int32` | per bar | Cumulative count of retests for the nearest support zone's lifecycle. Increments each time a retest occurs. |
| `ms_zone_retest_count_resistance` | `zone_retest_count_resistance` | `Int32` | per bar | Cumulative count of retests for the nearest resistance zone's lifecycle. |
| `ms_zone_flip_support` | `zone_flip_support` | `boolean` | per bar | True on the bar where a retested support zone confirms a role flip (support becomes resistance after break + retest + rejection). |
| `ms_zone_flip_resistance` | `zone_flip_resistance` | `boolean` | per bar | True on the bar where a retested resistance zone confirms a role flip (resistance becomes support). |
| `ms_zone_failed_retest_support` | `zone_failed_retest_support` | `boolean` | per bar | True on the bar where price breaks back through a support zone after a retest, reverting the zone to its original role. |
| `ms_zone_failed_retest_resistance` | `zone_failed_retest_resistance` | `boolean` | per bar | True on the bar where price breaks back through a resistance zone after a retest, reverting the zone to its original role. |

### Volatility & Distance

| DataFrame Column | Short Name | dtype | Updates | Description |
|---|---|---|---|---|
| `ms_atr` | `atr` | `float64` | per bar | Wilder's Average True Range with the configured lookback period (default 14). NaN for the first `atr_period - 1` bars. Direct passthrough of the computed ATR array. |
| `ms_distance_to_support` | `distance_to_support` | `float64` | per bar | ATR-normalized distance from close to the nearest support zone upper bound: `(close - support_zone_high) / atr`. Positive when price is above the zone; negative when inside or below. NaN when no zone or ATR is unavailable. |
| `ms_distance_to_resistance` | `distance_to_resistance` | `float64` | per bar | ATR-normalized distance from close to the nearest resistance zone lower bound: `(resistance_zone_low - close) / atr`. Positive when price is below the zone; negative when inside or above. |

### Swing Failure Pattern

| DataFrame Column | Short Name | dtype | Updates | Description |
|---|---|---|---|---|
| `ms_sfp_high` | `sfp_high` | `boolean` | per bar | Swing Failure Pattern (bearish): True when the bar's high exceeds the last confirmed top's HCO but the close rejects below it. Indicates a failed breakout above resistance. No recency filter is applied -- combine with `bars_since_last_top` for recency filtering. |
| `ms_sfp_low` | `sfp_low` | `boolean` | per bar | Swing Failure Pattern (bullish): True when the bar's low undercuts the last confirmed bottom's LCO but the close recovers above it. Indicates a failed breakdown below support. No recency filter -- combine with `bars_since_last_bottom`. |

---

## Key Concepts for Strategy Authors

### Update Frequency

Each column has one of three update frequencies, shown in the **Updates** column of the tables above:

- **per wave**: Value is identical for every bar within a wave segment. Changes only at wave boundaries (histogram sign-flips). Only `wave_side` and `wave_id` have this frequency.
- **per boundary**: Value steps forward when a new wave is confirmed, then holds constant until the next confirmation. Most structural columns (HH/HL flags, zone bounds, wave metrics, pullbacks, divergence) work this way. These are safe to use without `.shift()` because they only change at wave boundaries, which are already one bar old by the time they appear.
- **per bar**: Value can change on every bar, even within the same wave. This includes trend flags (which can break mid-wave), running extremes, ATR, distances, counters, lifecycle events, and SFP signals. Use `.shift(1)` when detecting transitions (e.g., `is_trending_up` flipping from True to False).

### HCO / LCO Pricing

All swing-comparison columns use close-or-open extremes, not wick extremes. HCO = `max(close, open)` of the most extreme candle; LCO = `min(close, open)`. This filters out wick noise and focuses on body-level price commitment.

### Trend Break is Per-Bar

Unlike the HH/HL booleans (per-boundary values constant until the next wave confirms), `is_trending_up` and `is_trending_down` are corrected on every bar within a wave. An uptrend can break mid-wave if a down-wave's low breaches the last bottom's LCO.

### Zone Semantics

Support/resistance zone columns reflect the *nearest* (most recently anchored) zone. Zones are built from the body of the anchor candle — the candle with the lowest close (support) or highest close (resistance) in the wave. This "body-anchored" geometry follows Auction Market Theory (Steidlmayer / Dalton): the body represents accepted price; wicks are rejected excursion.

The four `ms_*_zone_wick_*` columns preserve the wick-extended extrema for Wyckoff stop placement (place stops beyond the spring wick, not just beyond the body zone).

The `is_double` flag and `overlap_count` serve as zone-strength indicators — importantly, they measure *different* things:

- `is_double` is **price-level proximity**: "does a preceding same-side wave's extreme sit within a tolerance band around the anchor's extreme, with no violating swing in between?". Tolerance is ATR-derived (`tolerance_atr_multiple × atr[anchor.{low,high}_idx]`, default 0.3 × ATR) with a percentage-of-price fallback (default 0.4 %).
- `overlap_count` is **body geometry**: the number of preceding same-side body ranges that actually overlap the zone's range. A qualified double bottom with disjoint bodies will have `is_double=True` but not contribute to `overlap_count`, and will not extend the zone bounds.

Internally, zones are queried via `MarketStructureHelper.get_support_zones(atr_arr=...)` / `get_resistance_zones(atr_arr=...)`; the Freqtrade projector passes the column's `atr` array through, so both columns share a single ATR time-series.

When no zone exists, all zone columns are `NaN` / `pd.NA`.

### Warm-Up Behavior

Columns return `pd.NA` or `NaN` until enough waves exist to compute them. Structural comparisons require at least two confirmed waves of the relevant side. Always guard against `pd.NA` in conditions or use `.fillna(False)` for boolean columns.

---

## Usage Examples

### 1. Trend-Following Entry: Uptrend + Pullback Completion

Enter long when market structure confirms an uptrend and the pullback correction factor indicates a meaningful retracement before resuming.

```python
def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
    dataframe.loc[
        (
            (dataframe["ms_is_trending_up"] == True)
            & (dataframe["ms_wave_side"] == "up")
            & (dataframe["ms_pullback_correction_factor"] >= 0.5)
            & (dataframe["ms_pullback_correction_factor"] <= 1.0)
            & (dataframe["ms_wave_count"] >= 6)
        ),
        "enter_long",
    ] = 1
    return dataframe
```

### 2. Mean-Reversion Entry: At Support Zone

Enter long when price dips into a confirmed support zone with double-bottom strength.

```python
def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
    dataframe.loc[
        (
            (dataframe["low"] <= dataframe["ms_support_zone_high"])
            & (dataframe["low"] >= dataframe["ms_support_zone_low"])
            & (dataframe["ms_support_is_double"].fillna(False) == True)
            & (dataframe["ms_is_trending_down"] == False)
        ),
        "enter_long",
    ] = 1
    return dataframe
```

### 3. Divergence-Based Reversal

Enter on bullish/bearish divergence combined with zone proximity.

```python
def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
    # Bullish divergence long entry
    dataframe.loc[
        (
            (dataframe["ms_bullish_divergence"].fillna(False) == True)
            & (dataframe["ms_made_lower_low"].fillna(False) == True)
            & (dataframe["ms_wave_side"] == "up")
            & (dataframe["close"] <= dataframe["ms_support_zone_high"] * 1.01)
        ),
        "enter_long",
    ] = 1

    # Bearish divergence short entry
    dataframe.loc[
        (
            (dataframe["ms_bearish_divergence"].fillna(False) == True)
            & (dataframe["ms_made_higher_high"].fillna(False) == True)
            & (dataframe["ms_wave_side"] == "down")
            & (dataframe["close"] >= dataframe["ms_resistance_zone_low"] * 0.99)
        ),
        "enter_short",
    ] = 1
    return dataframe
```

### 4. Trend Strength Filtering: Wave Count + High/Low Since

Filter for significant swing points with sufficient structural history.

```python
def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
    dataframe.loc[
        (
            (dataframe["ms_is_trending_up"] == True)
            & (dataframe["ms_high_since"].fillna(0) >= 50)
            & (dataframe["ms_wave_count"] >= 8)
            & (dataframe["ms_wave_length"].fillna(0) <= 30)
        ),
        "enter_long",
    ] = 1
    return dataframe
```

### 5. Position Sizing with Pullback Correction Factor

Scale position size based on pullback depth.

```python
def custom_stake_amount(self, current_time, current_rate, proposed_stake,
                        min_stake, max_stake, leverage, entry_tag, side,
                        **kwargs) -> float:
    dataframe, _ = self.dp.get_analyzed_dataframe(
        pair=kwargs["pair"], timeframe=self.timeframe
    )
    last = dataframe.iloc[-1]
    correction = last.get("ms_pullback_correction_factor", float("nan"))

    if pd.isna(correction):
        return proposed_stake

    if 0.5 <= correction <= 0.85:
        scale = 1.0      # Deep retracement with room to run
    elif 0.85 < correction <= 1.0:
        scale = 0.6      # Nearly recovered prior leg
    elif correction < 0.5:
        scale = 0.5      # Very shallow pullback
    else:
        scale = 0.3      # Overextended beyond prior leg

    return max(min_stake, min(proposed_stake * scale, max_stake))
```

### 6. Multi-Condition Entry

Higher-conviction entry requiring structural trend, zone proximity, favorable pullback, and no divergence warning.

```python
def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
    dataframe.loc[
        (
            (dataframe["ms_is_trending_up"] == True)
            & (dataframe["ms_made_higher_high"].fillna(False) == True)
            & (dataframe["ms_made_higher_low"].fillna(False) == True)
            & (dataframe["ms_wave_side"] == "up")
            & (dataframe["ms_pullback_correction_factor"] >= 0.382)
            & (dataframe["ms_pullback_correction_factor"] <= 0.786)
            & (dataframe["ms_forming_wave_low"].shift(1)
               <= dataframe["ms_support_zone_high"].shift(1) * 1.005)
            & (dataframe["ms_bearish_divergence"].fillna(False) == False)
            & (dataframe["ms_wave_count"] >= 6)
        ),
        "enter_long",
    ] = 1
    return dataframe
```

### 7. Exit Signal: Trend Break Detection

Exit when the per-bar trend flag transitions from True to False.

```python
def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
    # Uptrend structure break
    dataframe.loc[
        (
            (dataframe["ms_is_trending_up"].shift(1) == True)
            & (dataframe["ms_is_trending_up"] == False)
        ),
        "exit_long",
    ] = 1

    # Lower high + lower low confirmed with price below support
    dataframe.loc[
        (
            (dataframe["ms_made_lower_high"].fillna(False) == True)
            & (dataframe["ms_made_lower_low"].fillna(False) == True)
            & (dataframe["ms_forming_wave_low"] < dataframe["ms_support_zone_low"])
        ),
        "exit_long",
    ] = 1

    # Downtrend structure break
    dataframe.loc[
        (
            (dataframe["ms_is_trending_down"].shift(1) == True)
            & (dataframe["ms_is_trending_down"] == False)
        ),
        "exit_short",
    ] = 1
    return dataframe
```

---

## Backtest vs. Live Behavior

In **backtest mode** (first call per pair), all columns are projected historically across the entire DataFrame via a forward pass over the wave registry. Per-bar columns are corrected bar-by-bar for accurate historical simulation.

In **live mode** (subsequent calls), only the current helper state is projected. All rows receive the same current-state value -- correct because Freqtrade strategies only inspect the last row for live decisions.

The `columns` parameter accepts a tuple of short names to project only the columns your strategy needs. Omitting unused columns (especially zone columns, which require the most computation) reduces overhead. Pass `None` (the default) to project all 67.

### Additional Parameters

| Parameter | Default | Description |
|---|---|---|
| `atr_period` | `14` | Lookback period for Wilder's ATR computation. |
| `retest_mode` | `"wick"` | Zone lifecycle detection mode: `"wick"` uses high/low, `"close"` uses close price. |
| `auto_histogram` | `False` | Compute TSI histogram automatically from close prices instead of requiring a pre-existing histogram column. |
| `tsi_r` | `12` | TSI long smoothing period (first EMA). |
| `tsi_s` | `8` | TSI short smoothing period (second EMA). |
| `tsi_signal_period` | `4` | TSI signal line EMA period. |

### Multi-Timeframe Wrapper

`attach_market_structure_mtf()` projects HTF (higher-timeframe) structure onto an LTF (lower-timeframe) DataFrame. It:

1. Resamples LTF candles to HTF (e.g. 1h → 4h)
2. Runs `attach_market_structure` on the HTF frame
3. Shifts results forward by one HTF period to prevent lookahead
4. Forward-fills back onto LTF rows

HTF columns are prefixed with `ms_{htf}_` instead of `ms_` (e.g. `ms_4h_wave_side`, `ms_4h_is_trending_up`).

**Lookahead prevention**: Values are always from the most recently *closed* HTF candle. The currently forming (incomplete) HTF candle is never visible to LTF bars. No `.shift()` is needed on MTF columns.

**Constraints**: HTF must be strictly greater than LTF and an even multiple of it (e.g. 4h/1h is valid, 4h/3h is not).

| Parameter | Default | Description |
|---|---|---|
| `htf` | *(required)* | Higher timeframe as a pandas offset alias (e.g. `"4h"`). |
| `ltf` | *(required)* | Lower timeframe as a pandas offset alias (e.g. `"1h"`). |
| `columns` | `None` (all) | Which columns to project — same short names as the main function. |

All other parameters (`hist_col`, `max_waves`, `atr_period`, `retest_mode`, `auto_histogram`, `tsi_r`, `tsi_s`, `tsi_signal_period`) are forwarded to `attach_market_structure` — see the parameter table above for defaults.

**Usage** (Freqtrade strategy):

```python
from market_structure.mtf import attach_market_structure_mtf

# In populate_indicators():
df = attach_market_structure_mtf(
    df, metadata, self.ms_store,
    htf="4h", ltf="1h",
    columns=("wave_side", "is_trending_up", "support_zone_low"),
    auto_histogram=True,
)
# Columns added: ms_4h_wave_side, ms_4h_is_trending_up, ms_4h_support_zone_low
```

The `store` dict is shared with the main function but keyed differently — MTF helpers are stored as `"{pair}_{htf}"` (e.g. `"BTC/USDT_4h"`), so one store can hold both LTF and HTF helpers without collision.

---

## Quick dtype Reference

| dtype | Columns | NA sentinel |
|---|---|---|
| `object` (str) | `wave_side`, `wave_id` | `""` (live) |
| `float64` | `last_top_price`, `last_bottom_price`, `pullback_correction_factor`, `pullback_breakout_level`, `pullback_price_diff`, `support_zone_low`, `support_zone_high`, `support_zone_wick_low`, `support_zone_wick_high`, `resistance_zone_low`, `resistance_zone_high`, `resistance_zone_wick_low`, `resistance_zone_wick_high`, `forming_wave_high`, `forming_wave_low`, `atr`, `structure_break_level`, `distance_to_support`, `distance_to_resistance`, `wave_amplitude`, `wave_slope`, `zone_quality_support`, `zone_quality_resistance`, `pullback_atr_factor`, `wave_volume`, `wave_volume_ratio`, `wave_amplitude_ratio` | `NaN` |
| `boolean` (nullable) | `made_higher_high`, `made_higher_low`, `made_lower_high`, `made_lower_low`, `bearish_divergence`, `bullish_divergence`, `support_is_double`, `resistance_is_double`, `structure_break_confirmed`, `sfp_high`, `sfp_low`, `zone_break_support`, `zone_break_resistance`, `zone_retest_support`, `zone_retest_resistance`, `zone_flip_support`, `zone_flip_resistance`, `zone_failed_retest_support`, `zone_failed_retest_resistance`, `three_push_up`, `three_push_down` | `pd.NA` |
| `Int32` (nullable) | `high_since`, `low_since`, `pullback_length`, `wave_length`, `wave_count`, `support_overlap_count`, `resistance_overlap_count`, `bars_since_last_top`, `bars_since_last_bottom`, `zone_retest_count_support`, `zone_retest_count_resistance`, `trend_wave_count`, `trend_duration` | `pd.NA` |
| `Int64` (nullable) | `support_zone_anchor_time`, `resistance_zone_anchor_time` | `pd.NA` |
| `bool` (non-nullable) | `is_trending_up`, `is_trending_down` | N/A (defaults to `False`) |

Note the distinction between nullable `boolean` (structural flags that genuinely lack a value during warm-up) and non-nullable `bool` (trend flags that default to `False` when structure is insufficient). When building conditions with nullable boolean columns, always use `.fillna(False)` to avoid `pd.NA` propagation in bitwise `&` / `|` chains.
