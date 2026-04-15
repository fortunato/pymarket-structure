import { ChangeDetectionStrategy, Component, computed, input } from '@angular/core';
import { DecimalPipe } from '@angular/common';

import { Trade } from '../../models/trade.model';

@Component({
	selector: 'app-trade-tooltip',
	imports: [DecimalPipe],
	templateUrl: './trade-tooltip.component.html',
	styleUrl: './trade-tooltip.component.scss',
	changeDetection: ChangeDetectionStrategy.OnPush,
	host: {
		'[style.left.px]': 'x() + 16',
		'[style.top.px]': 'y() - 40',
	},
})
export class TradeTooltipComponent {
	readonly trade = input.required<Trade>();
	readonly x = input.required<number>();
	readonly y = input.required<number>();
	readonly pinned = input(false);

	readonly isWin = computed(() => this.trade().profit_ratio > 0);
	readonly direction = computed(() => (this.trade().is_short ? 'SHORT' : 'LONG'));
	readonly profitPct = computed(() => this.trade().profit_ratio * 100);

	readonly entryDate = computed(() => this.formatDate(this.trade().open_time));
	readonly exitDate = computed(() => this.formatDate(this.trade().close_time));

	readonly durationHours = computed(() => {
		const secs = this.trade().close_time - this.trade().open_time;
		return Math.round(secs / 3600);
	});

	readonly durationBars = computed(() => {
		const secs = this.trade().close_time - this.trade().open_time;
		return Math.round(secs / (4 * 3600)); // 4h bars
	});

	private formatDate(epochSeconds: number): string {
		const d = new Date(epochSeconds * 1000);
		const months = [
			'Jan',
			'Feb',
			'Mar',
			'Apr',
			'May',
			'Jun',
			'Jul',
			'Aug',
			'Sep',
			'Oct',
			'Nov',
			'Dec',
		];
		return `${months[d.getUTCMonth()]} ${d.getUTCDate()} ${String(d.getUTCHours()).padStart(2, '0')}:${String(d.getUTCMinutes()).padStart(2, '0')}`;
	}
}
