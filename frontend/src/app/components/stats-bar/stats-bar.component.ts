import { ChangeDetectionStrategy, Component, computed, inject } from '@angular/core';
import { DecimalPipe } from '@angular/common';

import { TradeDataService } from '../../services/trade-data.service';

@Component({
	selector: 'app-stats-bar',
	imports: [DecimalPipe],
	templateUrl: './stats-bar.component.html',
	styleUrl: './stats-bar.component.scss',
	changeDetection: ChangeDetectionStrategy.OnPush,
})
export class StatsBarComponent {
	private readonly tradeData = inject(TradeDataService);
	readonly stats = computed(() => this.tradeData.stats());
	readonly hasTrades = computed(() => this.stats().totalTrades > 0);

	readonly formattedDuration = computed(() => {
		const h = this.stats().avgDurationHours;
		if (h >= 168) return `${(h / 24).toFixed(0)}d`;
		return `${h.toFixed(0)}h`;
	});
}
