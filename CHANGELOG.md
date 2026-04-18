# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-04-18

### Added

- Initial release of market-structure library
- Swing detection and wave construction
- Trend identification (higher highs/lows, lower highs/lows)
- Support and resistance zone detection with quality scoring
- Body-anchored zone geometry (anchor candle's body — accepted price per Auction Market Theory) rather than wick-extended geometry. Zones are narrower and not dominated by single-candle flash-crash wicks.
- `get_bottom_wick_range` / `get_top_wick_range` helpers returning wick-based geometry for Wyckoff stop placement beyond the spring wick
- Four wick auxiliary DataFrame columns: `ms_support_zone_wick_low`, `ms_support_zone_wick_high`, `ms_resistance_zone_wick_low`, `ms_resistance_zone_wick_high`
- `Zone.wick_range` field carrying wick-based extrema alongside the body-based `range`
- Double-pattern extension using body-coordinate overlap gate and body bottom/top extension values
- Zone lifecycle tracking (break, retest, flip)
- SFP (Swing Failure Pattern) detection
- Three-push exhaustion pattern detection
- Wave amplitude and pullback metrics
- ATR-normalized distance-to-zone calculations
- Multi-timeframe (MTF) analysis support
- Freqtrade integration via `attach_market_structure()`
- TSI (True Strength Index) histogram calculation
- Frontend info panel with wick bounds alongside body zone range
