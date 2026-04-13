import { Component } from '@angular/core';

import { HeaderComponent } from './components/header/header.component';
import { ChartComponent } from './components/chart/chart.component';
import { InfoPanelComponent } from './components/info-panel/info-panel.component';
import { OverlayControlsComponent } from './components/overlay-controls/overlay-controls.component';
import { StatsBarComponent } from './components/stats-bar/stats-bar.component';

@Component({
	selector: 'app-root',
	imports: [
		HeaderComponent,
		ChartComponent,
		InfoPanelComponent,
		OverlayControlsComponent,
		StatsBarComponent,
	],
	templateUrl: './app.html',
	styleUrl: './app.scss',
})
export class App {}
