import { computed, effect, inject, Injectable, signal } from '@angular/core';
import { HttpClient } from '@angular/common/http';

import { EnrichedBar } from '../models/candle-bar.model';
import { ChartStateService } from './chart-state.service';
import {
	detectDivergences,
	detectLifecycleEvents,
	detectPatternMarkers,
	detectStructureBreakSpans,
	detectTrendSpans,
	detectWaveTransitions,
	detectZoneSpans,
} from '../utils/zone-detector.util';

@Injectable({ providedIn: 'root' })
export class MarketDataService {
	private readonly http = inject(HttpClient);
	private readonly chartState = inject(ChartStateService);

	/** Internal writable signal updated when HTTP completes. */
	private readonly _bars = signal<EnrichedBar[]>([]);

	/** Public read-only bars signal. */
	readonly bars = this._bars.asReadonly();

	readonly supportZones = computed(() => detectZoneSpans(this.bars(), 'support'));
	readonly resistanceZones = computed(() => detectZoneSpans(this.bars(), 'resistance'));
	readonly waveTransitions = computed(() => detectWaveTransitions(this.bars()));
	readonly divergences = computed(() => detectDivergences(this.bars()));
	readonly trendSpans = computed(() => detectTrendSpans(this.bars()));
	readonly lifecycleEvents = computed(() => detectLifecycleEvents(this.bars()));
	readonly structureBreakSpans = computed(() => detectStructureBreakSpans(this.bars()));
	readonly patternMarkers = computed(() => detectPatternMarkers(this.bars()));

	constructor() {
		// Reload bars whenever selectedPair changes
		effect(() => {
			const pair = this.chartState.selectedPair();
			this.loadBars(pair);
		});
	}

	private loadBars(pair: string): void {
		this.http.get<EnrichedBar[]>(`assets/data/${pair}-4h.json`).subscribe((bars) => {
			this._bars.set(bars);
		});
	}
}
