export interface EnrichedBar {
	// OHLCV — time is Unix seconds for Lightweight Charts
	time: number;
	open: number;
	high: number;
	low: number;
	close: number;
	volume: number;
	tsi_histogram: number;
	tsi: number;
	tsi_signal: number;

	// Wave identity
	ms_wave_side: 'up' | 'down' | '';
	ms_wave_id: string;

	// Price levels
	ms_last_top_price: number | null;
	ms_last_bottom_price: number | null;

	// Structure
	ms_made_higher_high: boolean | null;
	ms_made_higher_low: boolean | null;
	ms_made_lower_high: boolean | null;
	ms_made_lower_low: boolean | null;

	// Swing significance
	ms_high_since: number | null;
	ms_low_since: number | null;

	// Support zones
	ms_support_zone_low: number | null;
	ms_support_zone_high: number | null;
	ms_support_is_double: boolean | null;
	ms_support_overlap_count: number | null;
	ms_support_zone_anchor_time: number | null;

	// Resistance zones
	ms_resistance_zone_low: number | null;
	ms_resistance_zone_high: number | null;
	ms_resistance_is_double: boolean | null;
	ms_resistance_overlap_count: number | null;
	ms_resistance_zone_anchor_time: number | null;

	// Trends
	ms_is_trending_up: boolean;
	ms_is_trending_down: boolean;

	// Forming wave
	ms_forming_wave_high: number | null;
	ms_forming_wave_low: number | null;

	// Divergence
	ms_bearish_divergence: boolean | null;
	ms_bullish_divergence: boolean | null;

	// Pullback metrics
	ms_pullback_length: number | null;
	ms_pullback_correction_factor: number | null;
	ms_pullback_breakout_level: number | null;
	ms_pullback_price_diff: number | null;

	// Wave metrics
	ms_wave_length: number | null;
	ms_wave_count: number | null;
}
