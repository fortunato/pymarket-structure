import {
	type ISeriesPrimitive,
	type IPrimitivePaneRenderer,
	type IPrimitivePaneView,
	type SeriesAttachedParameter,
	type Time,
} from 'lightweight-charts';
import type { CanvasRenderingTarget2D } from 'fancy-canvas';

import { TrendSpan } from '../models/chart-overlay.model';

class TrendBackgroundRenderer implements IPrimitivePaneRenderer {
	private _rects: { x1: number; x2: number; height: number; direction: 'up' | 'down' }[] = [];

	update(rects: typeof this._rects): void {
		this._rects = rects;
	}

	draw(target: CanvasRenderingTarget2D): void {
		target.useBitmapCoordinateSpace((scope) => {
			const ctx = scope.context;
			const hRatio = scope.horizontalPixelRatio;
			const vRatio = scope.verticalPixelRatio;

			for (const rect of this._rects) {
				ctx.fillStyle =
					rect.direction === 'up'
						? 'rgba(38, 166, 154, 0.05)'
						: 'rgba(239, 83, 80, 0.05)';
				ctx.fillRect(
					Math.round(rect.x1 * hRatio),
					0,
					Math.round((rect.x2 - rect.x1) * hRatio),
					Math.round(rect.height * vRatio),
				);
			}
		});
	}
}

class TrendBackgroundPaneView implements IPrimitivePaneView {
	private _renderer = new TrendBackgroundRenderer();
	private _spans: TrendSpan[] = [];
	private _attached: SeriesAttachedParameter<Time> | null = null;

	setData(spans: TrendSpan[]): void {
		this._spans = spans;
	}

	setAttached(params: SeriesAttachedParameter<Time>): void {
		this._attached = params;
	}

	renderer(): IPrimitivePaneRenderer {
		if (!this._attached) return this._renderer;

		const timeScale = this._attached.chart.timeScale();
		const rects: { x1: number; x2: number; height: number; direction: 'up' | 'down' }[] = [];

		const chartEl = this._attached.chart.chartElement();
		const height = chartEl?.clientHeight ?? 600;

		for (const span of this._spans) {
			const x1 = timeScale.timeToCoordinate(span.startTime as unknown as Time);
			const x2 = timeScale.timeToCoordinate(span.endTime as unknown as Time);
			if (x1 === null || x2 === null) continue;
			rects.push({ x1, x2, height, direction: span.direction });
		}

		this._renderer.update(rects);
		return this._renderer;
	}

	zOrder(): 'bottom' {
		return 'bottom';
	}
}

export class TrendBackgroundPrimitive implements ISeriesPrimitive<Time> {
	private _paneView = new TrendBackgroundPaneView();
	private _requestUpdate?: () => void;

	attached(params: SeriesAttachedParameter<Time>): void {
		this._requestUpdate = params.requestUpdate;
		this._paneView.setAttached(params);
	}

	detached(): void {
		this._requestUpdate = undefined;
	}

	updateAllViews(): void {
		// Called by LWC before rendering
	}

	paneViews(): IPrimitivePaneView[] {
		return [this._paneView];
	}

	setData(spans: TrendSpan[]): void {
		this._paneView.setData(spans);
		this._requestUpdate?.();
	}
}
