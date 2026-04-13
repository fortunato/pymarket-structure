import { Injectable, signal } from '@angular/core';

import { EnrichedBar } from '../models/candle-bar.model';
import { OverlayState } from '../models/chart-overlay.model';

@Injectable({ providedIn: 'root' })
export class ChartStateService {
	readonly overlays = signal<OverlayState>({
		supportZones: true,
		resistanceZones: true,
		waveTransitions: true,
		trendBackground: true,
		priceLines: true,
		divergenceMarkers: true,
	});

	readonly activeBar = signal<EnrichedBar | null>(null);

	toggleOverlay(key: keyof OverlayState): void {
		this.overlays.update((s) => ({ ...s, [key]: !s[key] }));
	}
}
