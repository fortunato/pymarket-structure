import { computed, inject, Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { toSignal } from '@angular/core/rxjs-interop';

import { EnrichedBar } from '../models/candle-bar.model';
import {
	detectDivergences,
	detectTrendSpans,
	detectWaveTransitions,
	detectZoneSpans,
} from '../utils/zone-detector.util';

@Injectable({ providedIn: 'root' })
export class MarketDataService {
	private readonly http = inject(HttpClient);

	private readonly bars$ = this.http.get<EnrichedBar[]>('assets/data/LTCUSDT-4h.json');
	readonly bars = toSignal(this.bars$, { initialValue: [] as EnrichedBar[] });

	readonly supportZones = computed(() => detectZoneSpans(this.bars(), 'support'));
	readonly resistanceZones = computed(() => detectZoneSpans(this.bars(), 'resistance'));
	readonly waveTransitions = computed(() => detectWaveTransitions(this.bars()));
	readonly divergences = computed(() => detectDivergences(this.bars()));
	readonly trendSpans = computed(() => detectTrendSpans(this.bars()));
}
