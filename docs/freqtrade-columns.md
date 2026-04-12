# Market Structure Columns -- Freqtrade Integration Reference

## Overview

The `attach_market_structure` wrapper projects up to 30 columns onto a Freqtrade DataFrame. Each column is prefixed with `ms_` in the DataFrame (e.g., the short name `wave_side` becomes `ms_wave_side`). Columns are organized into three tiers that reflect how frequently their values change:

- **Tier 1 -- Wave-constant**: Value is fixed for every bar within a single wave segment. Changes only at wave boundaries (histogram sign-flips).
- **Tier 2 -- Wave-boundary**: Value updates when a new wave is confirmed. Remains constant across all bars within a wave, but is computed from the cumulative wave history up to that point.
- **Tier 3 -- Per-bar**: Value can change on every bar, even within the same wave.

All columns use `pd.NA` (nullable) for boolean/integer fields and `np.nan` for float fields when insufficient wave history exists (warm-up period).

---

## Column Reference

### Tier 1 -- Wave-Constant

| DataFrame Column | Short Name | dtype | Description |
|---|---|---|---|
| `ms_wave_side` | `wave_side` | `object` (str) | Direction of the current wave segment: `"up"` when the histogram is non-negative, `"down"` when negative. Directly reflects the sign of the oscillator histogram driving wave detection. |
| `ms_wave_id` | `wave_id` | `object` (str) | Unique identifier for the wave this bar belongs to. Confirmed waves use `"w-0"`, `"w-1"`, etc. The still-forming (unconfirmed) wave uses `"forming-N"` where N is the next wave counter. |

### Tier 2 -- Wave-Boundary

#### Price Levels

| DataFrame Column | Short Name | dtype | Description |
|---|---|---|---|
| `ms_last_top_price` | `last_top_price` | `float64` | HCO (highest close-or-open) price of the most recently confirmed up-wave. This is `max(close, open)` of the candle with the highest close-or-open value in that wave. Represents the last significant swing high body level. |
| `ms_last_bottom_price` | `last_bottom_price` | `float64` | LCO (lowest close-or-open) price of the most recently confirmed down-wave. This is `min(close, open)` of the candle with the lowest close-or-open value in that wave. Represents the last significant swing low body level. |

#### Higher-High / Lower-Low Structure

| DataFrame Column | Short Name | dtype | Description |
|---|---|---|---|
| `ms_made_higher_high` | `made_higher_high` | `boolean` | True when the most recent confirmed up-wave's HCO exceeds the previous confirmed up-wave's HCO. Indicates an expanding swing high -- one of the two conditions for an uptrend. |
| `ms_made_higher_low` | `made_higher_low` | `boolean` | True when the most recent confirmed down-wave's LCO exceeds the previous confirmed down-wave's LCO. Indicates a rising swing low -- the second condition for an uptrend. |
| `ms_made_lower_high` | `made_lower_high` | `boolean` | True when the most recent confirmed up-wave's HCO is below the previous confirmed up-wave's HCO. Indicates a contracting swing high -- one of the two conditions for a downtrend. |
| `ms_made_lower_low` | `made_lower_low` | `boolean` | True when the most recent confirmed down-wave's LCO is below the previous confirmed down-wave's LCO. Indicates a falling swing low -- the second condition for a downtrend. |

#### Swing Significance

| DataFrame Column | Short Name | dtype | Description |
|---|---|---|---|
| `ms_high_since` | `high_since` | `Int32` | Number of bars backward from the most recent up-wave's HCO extreme to the last prior wave that had a higher HCO. Larger values indicate the current swing high is more significant (no prior wave exceeded it for a long time). |
| `ms_low_since` | `low_since` | `Int32` | Number of bars backward from the most recent down-wave's LCO extreme to the last prior wave that had a lower LCO. Larger values indicate the current swing low is more significant. |

#### Wave Metrics

| DataFrame Column | Short Name | dtype | Description |
|---|---|---|---|
| `ms_wave_length` | `wave_length` | `Int32` | Number of candles (bars) in the most recently confirmed wave. Useful for gauging momentum exhaustion or comparing wave durations across a trend. |
| `ms_wave_count` | `wave_count` | `Int32` | Total number of confirmed waves in the registry at this point. Increases monotonically during backtest. Useful for ensuring enough structure exists before applying wave-based conditions. |

#### Pullback Metrics

| DataFrame Column | Short Name | dtype | Description |
|---|---|---|---|
| `ms_pullback_length` | `pullback_length` | `Int32` | Bar count from the prior opposite-direction wave's extreme to this wave's extreme. Measures how many bars the retracement took. |
| `ms_pullback_correction_factor` | `pullback_correction_factor` | `float64` | Fraction of the prior run that was retraced, expressed as a ratio. A value of 1.0 means the retracement fully recovered the prior move; values above 1.0 indicate extension beyond. `NaN` during warm-up (fewer than 3 waves of the relevant sides). |
| `ms_pullback_breakout_level` | `pullback_breakout_level` | `float64` | The close-or-open price of the prior opposite wave's extreme candle. For an up-wave, this is the LCO of the last bottom -- the level price must reclaim. For a down-wave, the HCO of the last top. |
| `ms_pullback_price_diff` | `pullback_price_diff` | `float64` | Signed price distance of the pullback. Positive for up-waves (price rose from bottom), negative for down-waves (price fell from top). The absolute value is the raw retracement magnitude in price units. |

#### Divergence

| DataFrame Column | Short Name | dtype | Description |
|---|---|---|---|
| `ms_bearish_divergence` | `bearish_divergence` | `boolean` | True when the most recent confirmed up-wave made a higher close than the previous up-wave, but the peak histogram reading within the wave was lower. Classic bearish divergence signal: price momentum is weakening at higher prices. |
| `ms_bullish_divergence` | `bullish_divergence` | `boolean` | True when the most recent confirmed down-wave made a lower close than the previous down-wave, but the trough histogram reading within the wave was higher (less negative). Classic bullish divergence signal: selling momentum is weakening at lower prices. |

#### Support / Resistance Zones

| DataFrame Column | Short Name | dtype | Description |
|---|---|---|---|
| `ms_support_zone_low` | `support_zone_low` | `float64` | Lower bound of the nearest support zone. Derived from the lowest wick (low) of the anchor down-wave, extended if a double-bottom pattern exists. |
| `ms_support_zone_high` | `support_zone_high` | `float64` | Upper bound of the nearest support zone. Corresponds to the LCO body level of the anchor down-wave. |
| `ms_support_is_double` | `support_is_double` | `boolean` | True when the nearest support zone qualifies as a double-bottom pattern -- a preceding bottom wave's wick range overlaps the current zone and no intervening wave made a deeper low. |
| `ms_support_overlap_count` | `support_overlap_count` | `Int32` | Number of older down-waves whose bottom wick ranges overlap the nearest support zone. Higher counts suggest the zone has been tested more frequently. |
| `ms_resistance_zone_low` | `resistance_zone_low` | `float64` | Lower bound of the nearest resistance zone. Corresponds to the HCO body level of the anchor up-wave. |
| `ms_resistance_zone_high` | `resistance_zone_high` | `float64` | Upper bound of the nearest resistance zone. Derived from the highest wick (high) of the anchor up-wave, extended if a double-top pattern exists. |
| `ms_resistance_is_double` | `resistance_is_double` | `boolean` | True when the nearest resistance zone qualifies as a double-top pattern -- a preceding top wave's wick range overlaps the current zone and no intervening wave made a higher high. |
| `ms_resistance_overlap_count` | `resistance_overlap_count` | `Int32` | Number of older up-waves whose top wick ranges overlap the nearest resistance zone. Higher counts indicate more frequent tests of the level. |

### Tier 3 -- Per-Bar

| DataFrame Column | Short Name | dtype | Description |
|---|---|---|---|
| `ms_is_trending_up` | `is_trending_up` | `bool` | True when market structure confirms an uptrend: HH + HL from last two confirmed tops/bottoms. During down-waves, dynamically revoked bar-by-bar if the running low drops to or below the last bottom's LCO level (structure break). |
| `ms_is_trending_down` | `is_trending_down` | `bool` | True when market structure confirms a downtrend: LH + LL from last two confirmed tops/bottoms. During up-waves, dynamically revoked bar-by-bar if the running high reaches or exceeds the last top's HCO level (structure break). |
| `ms_forming_wave_high` | `forming_wave_high` | `float64` | Running maximum of bar highs within the current wave segment. Resets at each wave boundary (histogram sign-flip). |
| `ms_forming_wave_low` | `forming_wave_low` | `float64` | Running minimum of bar lows within the current wave segment. Resets at each wave boundary. |

---

## Key Concepts for Strategy Authors

**HCO / LCO pricing**: All swing-comparison columns use close-or-open extremes, not wick extremes. HCO = `max(close, open)` of the most extreme candle; LCO = `min(close, open)`. This filters out wick noise and focuses on body-level price commitment.

**Trend break is per-bar**: Unlike the HH/HL booleans (wave-boundary values constant until the next wave confirms), `is_trending_up` and `is_trending_down` are corrected on every bar within a wave. An uptrend can break mid-wave if a down-wave's low breaches the last bottom's LCO.

**Zone semantics**: Support/resistance zone columns reflect the *nearest* (most recently anchored) zone. Zones are built from overlapping wick ranges across bottom-waves (support) or top-waves (resistance). The `is_double` flag and `overlap_count` serve as zone-strength indicators. When no zone exists, all zone columns are `NaN` / `pd.NA`.

**Warm-up behavior**: Columns return `pd.NA` or `NaN` until enough waves exist to compute them. Structural comparisons require at least two confirmed waves of the relevant side. Always guard against `pd.NA` in conditions or use `.fillna(False)` for boolean columns.

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

In **backtest mode** (first call per pair), all 30 columns are projected historically across the entire DataFrame via a forward pass over the wave registry. Tier 3 columns are corrected per-bar for accurate historical simulation.

In **live mode** (subsequent calls), only the current helper state is projected. All rows receive the same current-state value -- correct because Freqtrade strategies only inspect the last row for live decisions.

The `columns` parameter accepts a tuple of short names to project only the columns your strategy needs. Omitting unused columns (especially zone columns, which require the most computation) reduces overhead. Pass `None` (the default) to project all 30.

---

## Quick dtype Reference

| dtype | Columns | NA sentinel |
|---|---|---|
| `object` (str) | `wave_side`, `wave_id` | `""` (live) |
| `float64` | `last_top_price`, `last_bottom_price`, `pullback_correction_factor`, `pullback_breakout_level`, `pullback_price_diff`, `support_zone_low`, `support_zone_high`, `resistance_zone_low`, `resistance_zone_high`, `forming_wave_high`, `forming_wave_low` | `NaN` |
| `boolean` (nullable) | `made_higher_high`, `made_higher_low`, `made_lower_high`, `made_lower_low`, `bearish_divergence`, `bullish_divergence`, `support_is_double`, `resistance_is_double` | `pd.NA` |
| `Int32` (nullable) | `high_since`, `low_since`, `pullback_length`, `wave_length`, `wave_count`, `support_overlap_count`, `resistance_overlap_count` | `pd.NA` |
| `bool` (non-nullable) | `is_trending_up`, `is_trending_down` | N/A (defaults to `False`) |

Note the distinction between nullable `boolean` (Tier 2 structural flags that genuinely lack a value during warm-up) and non-nullable `bool` (Tier 3 trend flags that default to `False` when structure is insufficient). When building conditions with nullable boolean columns, always use `.fillna(False)` to avoid `pd.NA` propagation in bitwise `&` / `|` chains.
