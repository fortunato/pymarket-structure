import { ChangeDetectionStrategy, Component, inject } from '@angular/core';

import { ChartStateService } from '../../services/chart-state.service';
import { AVAILABLE_PAIRS, AVAILABLE_STRATEGIES, STRATEGY_LABELS } from '../../models/trade.model';

@Component({
	selector: 'app-pair-strategy-selector',
	templateUrl: './pair-strategy-selector.component.html',
	styleUrl: './pair-strategy-selector.component.scss',
	changeDetection: ChangeDetectionStrategy.OnPush,
})
export class PairStrategySelectorComponent {
	readonly chartState = inject(ChartStateService);
	readonly pairs = AVAILABLE_PAIRS;
	readonly strategies = AVAILABLE_STRATEGIES;
	readonly strategyLabels = STRATEGY_LABELS;

	onPairChange(event: Event): void {
		const value = (event.target as HTMLSelectElement).value;
		this.chartState.selectedPair.set(value);
	}

	onStrategyChange(event: Event): void {
		const value = (event.target as HTMLSelectElement).value;
		this.chartState.selectedStrategy.set(value);
	}
}
