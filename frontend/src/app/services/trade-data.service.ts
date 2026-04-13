import { computed, effect, inject, Injectable, signal } from '@angular/core';
import { HttpClient } from '@angular/common/http';

import { BacktestFile, Trade } from '../models/trade.model';
import { ChartStateService } from './chart-state.service';
import { computeStats } from '../utils/trade-stats.util';

@Injectable({ providedIn: 'root' })
export class TradeDataService {
	private readonly http = inject(HttpClient);
	private readonly chartState = inject(ChartStateService);

	private readonly _trades = signal<Trade[]>([]);
	readonly trades = this._trades.asReadonly();

	readonly stats = computed(() => computeStats(this.trades()));

	constructor() {
		// Reload trades whenever pair or strategy changes
		effect(() => {
			const pair = this.chartState.selectedPair();
			const strategy = this.chartState.selectedStrategy();
			this.loadTrades(strategy, pair);
		});
	}

	private loadTrades(strategy: string, pair: string): void {
		const url = `assets/data/trades/${strategy}-${pair}-4h.json`;
		this.http.get<BacktestFile>(url).subscribe({
			next: (data) => this._trades.set(data.trades),
			error: () => this._trades.set([]),
		});
	}
}
