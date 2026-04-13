export interface OverlayState {
	supportZones: boolean;
	resistanceZones: boolean;
	waveTransitions: boolean;
	trendBackground: boolean;
	priceLines: boolean;
	divergenceMarkers: boolean;
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

export const OVERLAY_LABELS: Record<keyof OverlayState, string> = {
	supportZones: 'Support Zones',
	resistanceZones: 'Resistance Zones',
	waveTransitions: 'Wave Transitions',
	trendBackground: 'Trend Background',
	priceLines: 'Price Lines',
	divergenceMarkers: 'Divergence',
};
