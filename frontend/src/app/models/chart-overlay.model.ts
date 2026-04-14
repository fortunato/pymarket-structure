export interface OverlayState {
	supportZones: boolean;
	resistanceZones: boolean;
	waveTransitions: boolean;
	trendBackground: boolean;
	priceLines: boolean;
	divergenceMarkers: boolean;
	trades: boolean;
	zoneLifecycle: boolean;
	structureBreak: boolean;
	patterns: boolean;
}

export interface ZoneSpan {
	startTime: number;
	endTime: number;
	priceLow: number;
	priceHigh: number;
	isDouble: boolean;
	overlapCount: number;
	type: 'support' | 'resistance';
}

export interface TrendSpan {
	startTime: number;
	endTime: number;
	direction: 'up' | 'down';
}

export interface WaveTransition {
	time: number;
	waveId: string;
	newSide: 'up' | 'down';
}

export interface DivergenceMarker {
	time: number;
	type: 'bullish' | 'bearish';
}

export interface LifecycleEvent {
	time: number;
	eventType: 'break' | 'retest' | 'flip' | 'failed_retest';
	zoneSide: 'support' | 'resistance';
}

export interface StructureBreakSpan {
	startTime: number;
	endTime: number;
	level: number;
	isUptrend: boolean;
}

export interface PatternMarker {
	time: number;
	patternType: 'sfp_high' | 'sfp_low' | 'three_push_up' | 'three_push_down';
}

export const OVERLAY_LABELS: Record<keyof OverlayState, string> = {
	supportZones: 'Support Zones',
	resistanceZones: 'Resistance Zones',
	waveTransitions: 'Wave Transitions',
	trendBackground: 'Trend Background',
	priceLines: 'Price Lines',
	divergenceMarkers: 'Divergence',
	trades: 'Trades',
	zoneLifecycle: 'Zone Lifecycle',
	structureBreak: 'Structure Break',
	patterns: 'Patterns',
};
