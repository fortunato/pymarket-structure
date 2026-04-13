import { ChangeDetectionStrategy, Component, computed, inject } from '@angular/core';
import { DecimalPipe } from '@angular/common';

import { ChartStateService } from '../../services/chart-state.service';

@Component({
	selector: 'app-info-panel',
	imports: [DecimalPipe],
	templateUrl: './info-panel.component.html',
	styleUrl: './info-panel.component.scss',
	changeDetection: ChangeDetectionStrategy.OnPush,
})
export class InfoPanelComponent {
	private readonly chartState = inject(ChartStateService);

	readonly bar = computed(() => this.chartState.activeBar());

	readonly formattedTime = computed(() => {
		const b = this.bar();
		if (!b) return '';
		return new Date(b.time * 1000).toLocaleString('en-US', {
			month: 'short',
			day: 'numeric',
			year: 'numeric',
			hour: '2-digit',
			minute: '2-digit',
			hour12: false,
		});
	});
}
