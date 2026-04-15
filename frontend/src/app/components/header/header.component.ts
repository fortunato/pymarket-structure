import { ChangeDetectionStrategy, Component } from '@angular/core';

import { PairStrategySelectorComponent } from '../pair-strategy-selector/pair-strategy-selector.component';

@Component({
	selector: 'app-header',
	imports: [PairStrategySelectorComponent],
	templateUrl: './header.component.html',
	styleUrl: './header.component.scss',
	changeDetection: ChangeDetectionStrategy.OnPush,
})
export class HeaderComponent {}
