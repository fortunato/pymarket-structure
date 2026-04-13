import {
	type ISeriesPrimitive,
	type IPrimitivePaneRenderer,
	type IPrimitivePaneView,
	type SeriesAttachedParameter,
	type Time,
} from 'lightweight-charts';
import type { CanvasRenderingTarget2D } from 'fancy-canvas';

import { ZoneSpan } from '../models/chart-overlay.model';

interface ZoneRect {
	x1: number;
	x2: number;
	y1: number; // priceHigh coordinate (top of zone on screen)
	y2: number; // priceLow coordinate (bottom of zone on screen)
	isDouble: boolean;
	overlapCount: number;
	type: 'support' | 'resistance';
}

// Color base values
const SUPPORT_RGB = '38, 166, 154';
const RESISTANCE_RGB = '239, 83, 80';

/**
 * Graduated fill opacity based on overlap count.
 * More overlapping wave extremes = more structural confirmation.
 */
function bandOpacity(overlapCount: number): number {
	if (overlapCount === 0) return 0.04;
	if (overlapCount === 1) return 0.08;
	if (overlapCount === 2) return 0.14;
	return 0.2; // 3+
}

/**
 * Key-level line width (the prominent body-edge line).
 */
function keyLineWidth(overlapCount: number, isDouble: boolean, pixelRatio: number): number {
	if (isDouble || overlapCount >= 3) return 2.5 * pixelRatio;
	if (overlapCount >= 1) return 1.5 * pixelRatio;
	return 1 * pixelRatio;
}

/**
 * Key-level line opacity.
 */
function keyLineOpacity(overlapCount: number): number {
	if (overlapCount === 0) return 0.25;
	if (overlapCount === 1) return 0.4;
	if (overlapCount === 2) return 0.55;
	return 0.7; // 3+
}

/**
 * Create a diagonal hatch pattern for double-bottom/top zones.
 */
function createHatchPattern(
	ctx: CanvasRenderingContext2D,
	rgb: string,
	opacity: number,
	pixelRatio: number,
): CanvasPattern | null {
	const size = Math.round(8 * pixelRatio);
	const canvas = new OffscreenCanvas(size, size);
	const pctx = canvas.getContext('2d');
	if (!pctx) return null;

	pctx.strokeStyle = `rgba(${rgb}, ${opacity})`;
	pctx.lineWidth = Math.max(1, pixelRatio);

	// Diagonal lines at 45 degrees
	pctx.beginPath();
	pctx.moveTo(0, size);
	pctx.lineTo(size, 0);
	pctx.moveTo(-size / 2, size / 2);
	pctx.lineTo(size / 2, -size / 2);
	pctx.moveTo(size / 2, size + size / 2);
	pctx.lineTo(size + size / 2, size / 2);
	pctx.stroke();

	return ctx.createPattern(canvas, 'repeat');
}

class ZoneRectangleRenderer implements IPrimitivePaneRenderer {
	private _rects: ZoneRect[] = [];

	update(rects: ZoneRect[]): void {
		this._rects = rects;
	}

	draw(target: CanvasRenderingTarget2D): void {
		target.useBitmapCoordinateSpace((scope) => {
			const ctx = scope.context;
			const hRatio = scope.horizontalPixelRatio;
			const vRatio = scope.verticalPixelRatio;

			for (const rect of this._rects) {
				const x1 = Math.round(rect.x1 * hRatio);
				const x2 = Math.round(rect.x2 * hRatio);
				const y1 = Math.round(rect.y1 * vRatio);
				const y2 = Math.round(rect.y2 * vRatio);
				const w = x2 - x1;
				const h = y2 - y1;

				if (w <= 0 || h <= 0) continue;

				const rgb = rect.type === 'support' ? SUPPORT_RGB : RESISTANCE_RGB;
				const fillAlpha = bandOpacity(rect.overlapCount);

				// 1. Thin band fill (faint background for wick range)
				if (rect.isDouble) {
					// Hatched pattern for doubles — qualitatively different signal
					const pattern = createHatchPattern(ctx, rgb, fillAlpha + 0.08, hRatio);
					if (pattern) {
						ctx.fillStyle = pattern;
						ctx.fillRect(x1, y1, w, h);
					}
				} else {
					ctx.fillStyle = `rgba(${rgb}, ${fillAlpha})`;
					ctx.fillRect(x1, y1, w, h);
				}

				// 2. Prominent key-level line at the body edge
				//    Support: body edge is at priceHigh (y1 = top)
				//    Resistance: body edge is at priceLow (y2 = bottom)
				const lineY = rect.type === 'support' ? y1 : y2;
				const lineAlpha = keyLineOpacity(rect.overlapCount);
				ctx.strokeStyle = `rgba(${rgb}, ${lineAlpha})`;
				ctx.lineWidth = keyLineWidth(rect.overlapCount, rect.isDouble, hRatio);
				ctx.beginPath();
				ctx.moveTo(x1, lineY);
				ctx.lineTo(x2, lineY);
				ctx.stroke();

				// 3. Overlap count label at the leading edge
				if (rect.overlapCount > 0) {
					const fontSize = Math.round(10 * hRatio);
					ctx.font = `600 ${fontSize}px Inter, system-ui, sans-serif`;
					ctx.fillStyle = `rgba(${rgb}, ${lineAlpha})`;
					const label = `x${rect.overlapCount}`;
					const textX = x1 + Math.round(4 * hRatio);
					const textY =
						lineY +
						(rect.type === 'support'
							? -Math.round(4 * vRatio)
							: Math.round(fontSize + 2 * vRatio));
					ctx.fillText(label, textX, textY);
				}
			}
		});
	}
}

class ZoneRectanglePaneView implements IPrimitivePaneView {
	private _renderer = new ZoneRectangleRenderer();
	private _zones: ZoneSpan[] = [];
	private _attached: SeriesAttachedParameter<Time> | null = null;

	setData(zones: ZoneSpan[]): void {
		this._zones = zones;
	}

	setAttached(params: SeriesAttachedParameter<Time>): void {
		this._attached = params;
	}

	renderer(): IPrimitivePaneRenderer {
		if (!this._attached) return this._renderer;

		const timeScale = this._attached.chart.timeScale();
		const series = this._attached.series;
		const rects: ZoneRect[] = [];

		for (const zone of this._zones) {
			const x1 = timeScale.timeToCoordinate(zone.startTime as unknown as Time);
			const x2 = timeScale.timeToCoordinate(zone.endTime as unknown as Time);
			const y1 = series.priceToCoordinate(zone.priceHigh);
			const y2 = series.priceToCoordinate(zone.priceLow);

			if (x1 === null || x2 === null || y1 === null || y2 === null) continue;

			rects.push({
				x1,
				x2,
				y1,
				y2,
				isDouble: zone.isDouble,
				overlapCount: zone.overlapCount,
				type: zone.type,
			});
		}

		this._renderer.update(rects);
		return this._renderer;
	}
}

export class ZoneRectanglePrimitive implements ISeriesPrimitive<Time> {
	private _paneView = new ZoneRectanglePaneView();
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

	setData(zones: ZoneSpan[]): void {
		this._paneView.setData(zones);
		this._requestUpdate?.();
	}
}
