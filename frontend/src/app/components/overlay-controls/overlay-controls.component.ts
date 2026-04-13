import { ChangeDetectionStrategy, Component, computed, inject } from '@angular/core';

import { ChartStateService } from '../../services/chart-state.service';
import { OVERLAY_LABELS, type OverlayState } from '../../models/chart-overlay.model';

@Component({
	selector: 'app-overlay-controls',
	templateUrl: './overlay-controls.component.html',
	styleUrl: './overlay-controls.component.scss',
	changeDetection: ChangeDetectionStrategy.OnPush,
})
export class OverlayControlsComponent {
	private readonly chartState = inject(ChartStateService);

	readonly entries = computed(() => {
		const state = this.chartState.overlays();
		return (Object.keys(state) as (keyof OverlayState)[]).map((key) => ({
			key,
			label: OVERLAY_LABELS[key],
			active: state[key],
		}));
	});

	toggle(key: keyof OverlayState): void {
		this.chartState.toggleOverlay(key);
	}
}
