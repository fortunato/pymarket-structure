import { EnrichedBar } from '../models/candle-bar.model';
import {
	DivergenceMarker,
	LifecycleEvent,
	PatternMarker,
	StructureBreakSpan,
	TrendSpan,
	WaveTransition,
	ZoneSpan,
} from '../models/chart-overlay.model';

/**
 * Detect contiguous zone spans from consecutive bar transitions.
 * A new zone starts when the zone price values change or appear from null.
 * A zone ends when values change again or become null.
 */
export function detectZoneSpans(bars: EnrichedBar[], type: 'support' | 'resistance'): ZoneSpan[] {
	if (bars.length === 0) return [];

	const lowKey = type === 'support' ? 'ms_support_zone_low' : 'ms_resistance_zone_low';
	const highKey = type === 'support' ? 'ms_support_zone_high' : 'ms_resistance_zone_high';
	const doubleKey = type === 'support' ? 'ms_support_is_double' : 'ms_resistance_is_double';
	const overlapKey =
		type === 'support' ? 'ms_support_overlap_count' : 'ms_resistance_overlap_count';
	const anchorKey =
		type === 'support' ? 'ms_support_zone_anchor_time' : 'ms_resistance_zone_anchor_time';

	const spans: ZoneSpan[] = [];
	let currentLow: number | null = null;
	let currentHigh: number | null = null;
	let startTime = 0;
	let isDouble = false;
	let overlapCount = 0;

	for (const bar of bars) {
		const lo = bar[lowKey];
		const hi = bar[highKey];

		if (lo !== currentLow || hi !== currentHigh) {
			// Close previous span if it was active
			if (currentLow !== null && currentHigh !== null) {
				spans.push({
					startTime,
					endTime: bar.time,
					priceLow: currentLow,
					priceHigh: currentHigh,
					isDouble,
					overlapCount,
					type,
				});
			}
			// Open new span — use anchor time (defining candle) if available,
			// otherwise fall back to the current bar's time
			currentLow = lo;
			currentHigh = hi;
			startTime = bar[anchorKey] ?? bar.time;
			isDouble = bar[doubleKey] ?? false;
			overlapCount = bar[overlapKey] ?? 0;
		}
	}

	// Close final span
	if (currentLow !== null && currentHigh !== null && bars.length > 0) {
		spans.push({
			startTime,
			endTime: bars[bars.length - 1].time,
			priceLow: currentLow,
			priceHigh: currentHigh,
			isDouble,
			overlapCount,
			type,
		});
	}

	return spans;
}

/**
 * Detect bars where ms_wave_id changes — these are wave transitions.
 */
export function detectWaveTransitions(bars: EnrichedBar[]): WaveTransition[] {
	const transitions: WaveTransition[] = [];
	for (let i = 1; i < bars.length; i++) {
		if (bars[i].ms_wave_id !== bars[i - 1].ms_wave_id && bars[i].ms_wave_side) {
			transitions.push({
				time: bars[i].time,
				waveId: bars[i].ms_wave_id,
				newSide: bars[i].ms_wave_side as 'up' | 'down',
			});
		}
	}
	return transitions;
}

/**
 * Detect bars where divergence transitions to true.
 */
export function detectDivergences(bars: EnrichedBar[]): DivergenceMarker[] {
	const markers: DivergenceMarker[] = [];
	for (let i = 1; i < bars.length; i++) {
		if (bars[i].ms_bearish_divergence && !bars[i - 1].ms_bearish_divergence) {
			markers.push({ time: bars[i].time, type: 'bearish' });
		}
		if (bars[i].ms_bullish_divergence && !bars[i - 1].ms_bullish_divergence) {
			markers.push({ time: bars[i].time, type: 'bullish' });
		}
	}
	return markers;
}

/**
 * Detect contiguous spans where trending is active.
 */
export function detectTrendSpans(bars: EnrichedBar[]): TrendSpan[] {
	if (bars.length === 0) return [];

	const spans: TrendSpan[] = [];
	let currentDir: 'up' | 'down' | null = null;
	let startTime = 0;

	for (const bar of bars) {
		const dir = bar.ms_is_trending_up ? 'up' : bar.ms_is_trending_down ? 'down' : null;

		if (dir !== currentDir) {
			if (currentDir !== null) {
				spans.push({ startTime, endTime: bar.time, direction: currentDir });
			}
			currentDir = dir;
			startTime = bar.time;
		}
	}

	if (currentDir !== null && bars.length > 0) {
		spans.push({ startTime, endTime: bars[bars.length - 1].time, direction: currentDir });
	}

	return spans;
}

/**
 * Detect bars where zone lifecycle events fire (break, retest, flip, failed_retest).
 */
export function detectLifecycleEvents(bars: EnrichedBar[]): LifecycleEvent[] {
	const events: LifecycleEvent[] = [];
	const sides = ['support', 'resistance'] as const;
	const types = ['break', 'retest', 'flip', 'failed_retest'] as const;

	for (const bar of bars) {
		for (const side of sides) {
			for (const eventType of types) {
				const key = `ms_zone_${eventType}_${side}` as keyof EnrichedBar;
				if (bar[key] === true) {
					events.push({ time: bar.time, eventType, zoneSide: side });
				}
			}
		}
	}
	return events;
}

/**
 * Detect contiguous spans of the same structure_break_level.
 */
export function detectStructureBreakSpans(bars: EnrichedBar[]): StructureBreakSpan[] {
	const spans: StructureBreakSpan[] = [];
	let currentLevel: number | null = null;
	let startTime = 0;
	let isUptrend = false;

	for (const bar of bars) {
		const level = bar.ms_structure_break_level;
		if (level !== currentLevel) {
			if (currentLevel !== null) {
				spans.push({ startTime, endTime: bar.time, level: currentLevel, isUptrend });
			}
			currentLevel = level;
			if (level !== null) {
				startTime = bar.time;
				isUptrend = bar.ms_is_trending_up;
			}
		}
	}

	if (currentLevel !== null && bars.length > 0) {
		spans.push({
			startTime,
			endTime: bars[bars.length - 1].time,
			level: currentLevel,
			isUptrend,
		});
	}

	return spans;
}

/**
 * Detect bars where SFP or three-push patterns are active.
 */
export function detectPatternMarkers(bars: EnrichedBar[]): PatternMarker[] {
	const markers: PatternMarker[] = [];
	const patternKeys = [
		['ms_sfp_high', 'sfp_high'],
		['ms_sfp_low', 'sfp_low'],
		['ms_three_push_up', 'three_push_up'],
		['ms_three_push_down', 'three_push_down'],
	] as const;

	for (const bar of bars) {
		for (const [key, patternType] of patternKeys) {
			if (bar[key] === true) {
				markers.push({ time: bar.time, patternType });
			}
		}
	}
	return markers;
}
