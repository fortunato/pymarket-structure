import { ChangeDetectionStrategy, Component, computed, inject } from '@angular/core';
import { DecimalPipe } from '@angular/common';

import { ChartStateService } from '../../services/chart-state.service';
import { EnrichedBar } from '../../models/candle-bar.model';
import { HintComponent } from '../hint/hint.component';

@Component({
	selector: 'app-info-panel',
	imports: [DecimalPipe, HintComponent],
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

	fmtPrice(v: number | null): string {
		return v === null || v === undefined ? '--' : v.toFixed(2);
	}

	fmtInt(v: number | null): string {
		return v === null || v === undefined ? '--' : Math.round(v).toString();
	}

	fmtAtr(v: number | null): string {
		return v === null || v === undefined ? '--' : `${v.toFixed(1)} ATR`;
	}

	fmtRatio(v: number | null): string {
		return v === null || v === undefined ? '--' : `${v.toFixed(1)}x`;
	}

	fmtAtrRaw(v: number | null): string {
		return v === null || v === undefined ? '--' : v.toFixed(2);
	}

	fmtScore(v: number | null): string {
		return v === null || v === undefined ? '--' : v.toFixed(1);
	}

	qualityLabel(v: number | null): string {
		if (v === null || v === undefined) return '';
		if (v < 3) return 'Weak';
		if (v < 6) return 'Fair';
		if (v < 8) return 'Strong';
		return 'Very Strong';
	}

	qualityClass(v: number | null): string {
		if (v === null || v === undefined) return '';
		if (v < 3) return 'quality-weak';
		if (v < 6) return 'quality-fair';
		if (v < 8) return 'quality-strong';
		return 'quality-very-strong';
	}

	fmtDistance(v: number | null, side: 'support' | 'resistance'): string {
		if (v === null || v === undefined) return '--';
		const dir = side === 'support' ? 'above' : 'below';
		return `${v.toFixed(1)} ATR ${dir}`;
	}

	hasLifecycleEvent(b: EnrichedBar): boolean {
		return !!(
			b.ms_zone_break_support ||
			b.ms_zone_break_resistance ||
			b.ms_zone_retest_support ||
			b.ms_zone_retest_resistance ||
			b.ms_zone_flip_support ||
			b.ms_zone_flip_resistance ||
			b.ms_zone_failed_retest_support ||
			b.ms_zone_failed_retest_resistance
		);
	}
}
