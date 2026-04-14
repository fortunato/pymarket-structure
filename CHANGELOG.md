# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-04-14

### Added

- Initial release of market-structure library
- Swing detection and wave construction
- Trend identification (higher highs/lows, lower highs/lows)
- Support and resistance zone detection with quality scoring
- Zone lifecycle tracking (break, retest, flip)
- SFP (Swing Failure Pattern) detection
- Three-push exhaustion pattern detection
- Wave amplitude and pullback metrics
- ATR-normalized distance-to-zone calculations
- Multi-timeframe (MTF) analysis support
- Freqtrade integration via `attach_market_structure()`
- TSI (True Strength Index) histogram calculation
