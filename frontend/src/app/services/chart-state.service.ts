import { Injectable, signal } from '@angular/core';

import { EnrichedBar } from '../models/candle-bar.model';
import { OverlayState } from '../models/chart-overlay.model';
import { Trade } from '../models/trade.model';

@Injectable({ providedIn: 'root' })
export class ChartStateService {
	readonly selectedPair = signal('BTCUSDT');
	readonly selectedStrategy = signal('MsFilterV6');

	readonly overlays = signal<OverlayState>({
		supportZones: true,
		resistanceZones: true,
		waveTransitions: true,
		trendBackground: true,
		priceLines: true,
		divergenceMarkers: true,
		trades: true,
		zoneLifecycle: false,
		structureBreak: false,
		patterns: false,
	});

	readonly activeBar = signal<EnrichedBar | null>(null);
	readonly hoveredTrade = signal<Trade | null>(null);
	readonly pinnedTrade = signal<Trade | null>(null);

	toggleOverlay(key: keyof OverlayState): void {
		this.overlays.update((s) => ({ ...s, [key]: !s[key] }));
	}
}
