import {
	type ISeriesPrimitive,
	type IPrimitivePaneRenderer,
	type IPrimitivePaneView,
	type SeriesAttachedParameter,
	type Time,
} from 'lightweight-charts';
import type { CanvasRenderingTarget2D } from 'fancy-canvas';

import { StructureBreakSpan } from '../models/chart-overlay.model';

interface LineSegment {
	x1: number;
	x2: number;
	y: number;
	isUptrend: boolean;
}

const UPTREND_COLOR = 'rgba(38, 166, 154, 0.6)';
const DOWNTREND_COLOR = 'rgba(239, 83, 80, 0.6)';

class StructureBreakRenderer implements IPrimitivePaneRenderer {
	private _segments: LineSegment[] = [];

	update(segments: LineSegment[]): void {
		this._segments = segments;
	}

	draw(target: CanvasRenderingTarget2D): void {
		target.useBitmapCoordinateSpace((scope) => {
			const ctx = scope.context;
			const hRatio = scope.horizontalPixelRatio;
			const vRatio = scope.verticalPixelRatio;

			for (const seg of this._segments) {
				const x1 = Math.round(seg.x1 * hRatio);
				const x2 = Math.round(seg.x2 * hRatio);
				const y = Math.round(seg.y * vRatio);

				if (x2 <= x1) continue;

				ctx.strokeStyle = seg.isUptrend ? UPTREND_COLOR : DOWNTREND_COLOR;
				ctx.lineWidth = Math.max(1, 1.5 * hRatio);
				ctx.setLineDash([6 * hRatio, 4 * hRatio]);
				ctx.beginPath();
				ctx.moveTo(x1, y);
				ctx.lineTo(x2, y);
				ctx.stroke();
				ctx.setLineDash([]);
			}
		});
	}
}

class StructureBreakPaneView implements IPrimitivePaneView {
	private _renderer = new StructureBreakRenderer();
	private _spans: StructureBreakSpan[] = [];
	private _attached: SeriesAttachedParameter<Time> | null = null;

	setData(spans: StructureBreakSpan[]): void {
		this._spans = spans;
	}

	setAttached(params: SeriesAttachedParameter<Time>): void {
		this._attached = params;
	}

	renderer(): IPrimitivePaneRenderer {
		if (!this._attached) return this._renderer;

		const timeScale = this._attached.chart.timeScale();
		const series = this._attached.series;
		const segments: LineSegment[] = [];

		for (const span of this._spans) {
			const x1 = timeScale.timeToCoordinate(span.startTime as unknown as Time);
			const x2 = timeScale.timeToCoordinate(span.endTime as unknown as Time);
			const y = series.priceToCoordinate(span.level);

			if (x1 === null || x2 === null || y === null) continue;

			segments.push({ x1, x2, y, isUptrend: span.isUptrend });
		}

		this._renderer.update(segments);
		return this._renderer;
	}
}

export class StructureBreakPrimitive implements ISeriesPrimitive<Time> {
	private _paneView = new StructureBreakPaneView();
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

	setData(spans: StructureBreakSpan[]): void {
		this._paneView.setData(spans);
		this._requestUpdate?.();
	}
}
