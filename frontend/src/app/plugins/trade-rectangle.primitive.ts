import {
	type ISeriesPrimitive,
	type IPrimitivePaneRenderer,
	type IPrimitivePaneView,
	type SeriesAttachedParameter,
	type Time,
} from 'lightweight-charts';
import type { CanvasRenderingTarget2D } from 'fancy-canvas';

import { Trade, TradeRect } from '../models/trade.model';

// ── Color palette ──────────────────────────────────────────────────
const WIN_RGB = '38, 166, 154';
const LOSS_RGB = '239, 83, 80';

function tradeColor(isWin: boolean): string {
	return isWin ? WIN_RGB : LOSS_RGB;
}

// ── Renderer ───────────────────────────────────────────────────────

class TradeRectangleRenderer implements IPrimitivePaneRenderer {
	private _rects: TradeRect[] = [];
	private _hoveredIndex: number | null = null;

	update(rects: TradeRect[], hoveredIndex: number | null): void {
		this._rects = rects;
		this._hoveredIndex = hoveredIndex;
	}

	draw(target: CanvasRenderingTarget2D): void {
		target.useBitmapCoordinateSpace((scope) => {
			const ctx = scope.context;
			const hRatio = scope.horizontalPixelRatio;
			const vRatio = scope.verticalPixelRatio;
			const hasHover = this._hoveredIndex !== null;

			for (let i = 0; i < this._rects.length; i++) {
				const rect = this._rects[i];
				const isHovered = i === this._hoveredIndex;

				const x1 = Math.round(rect.x1 * hRatio);
				const x2 = Math.round(rect.x2 * hRatio);
				const y1 = Math.round(rect.y1 * vRatio);
				const y2 = Math.round(rect.y2 * vRatio);
				const w = x2 - x1;
				const h = y2 - y1;

				if (w <= 0) continue;
				// Allow very thin trades (single candle) by using min height
				const drawH = Math.max(h, Math.round(2 * vRatio));

				const rgb = tradeColor(rect.isWin);

				// Fill opacity: hovered=30%, normal=15%, dimmed=6%
				const fillAlpha = isHovered ? 0.3 : hasHover ? 0.06 : 0.15;
				ctx.fillStyle = `rgba(${rgb}, ${fillAlpha})`;
				ctx.fillRect(x1, y1, w, drawH);

				// Border: hovered=full, normal=60%, dimmed=20%
				const borderAlpha = isHovered ? 1.0 : hasHover ? 0.2 : 0.6;
				ctx.strokeStyle = `rgba(${rgb}, ${borderAlpha})`;
				ctx.lineWidth = Math.max(1, (isHovered ? 2 : 1) * hRatio);

				// Short trades get dashed border
				if (rect.trade.is_short) {
					ctx.setLineDash([6 * hRatio, 3 * hRatio]);
				} else {
					ctx.setLineDash([]);
				}

				ctx.strokeRect(x1, y1, w, drawH);
				ctx.setLineDash([]); // reset

				// ── Entry marker: arrow + direction label ──────────────
				if (!hasHover || isHovered) {
					const entryY = Math.round(
						(rect.trade.is_short
							? Math.min(rect.y1, rect.y2)
							: Math.max(rect.y1, rect.y2)) * vRatio,
					);
					const markerSize = Math.round(5 * hRatio);
					const markerAlpha = isHovered ? 1.0 : 0.8;
					ctx.fillStyle = `rgba(${rgb}, ${markerAlpha})`;
					ctx.beginPath();
					if (rect.trade.is_short) {
						// Down-pointing triangle for shorts
						ctx.moveTo(x1 - markerSize, entryY - markerSize);
						ctx.lineTo(x1 + markerSize, entryY - markerSize);
						ctx.lineTo(x1, entryY + markerSize);
					} else {
						// Up-pointing triangle for longs
						ctx.moveTo(x1 - markerSize, entryY + markerSize);
						ctx.lineTo(x1 + markerSize, entryY + markerSize);
						ctx.lineTo(x1, entryY - markerSize);
					}
					ctx.closePath();
					ctx.fill();

					// Direction label at entry edge
					const labelFontSize = Math.round(10 * hRatio);
					ctx.font = `700 ${labelFontSize}px Inter, system-ui, sans-serif`;
					ctx.fillStyle = `rgba(${rgb}, ${markerAlpha})`;
					const dirLabel = rect.trade.is_short ? 'S' : 'L';
					const labelX = x1 + Math.round(8 * hRatio);
					const labelY =
						entryY +
						(rect.trade.is_short
							? -Math.round(5 * vRatio)
							: Math.round(labelFontSize + 2 * vRatio));
					ctx.fillText(dirLabel, labelX, labelY);
				}

				// ── Exit marker: small circle with label ───────────────
				if (!hasHover || isHovered) {
					const exitY = Math.round(
						(rect.trade.is_short
							? Math.max(rect.y1, rect.y2)
							: Math.min(rect.y1, rect.y2)) * vRatio,
					);
					const circleR = Math.round(3.5 * hRatio);
					const markerAlpha = isHovered ? 1.0 : 0.8;
					ctx.fillStyle = `rgba(${rgb}, ${markerAlpha})`;
					ctx.beginPath();
					ctx.arc(x2, exitY, circleR, 0, Math.PI * 2);
					ctx.fill();

					// Exit reason label (only on hover or when not dimmed)
					if (isHovered || !hasHover) {
						const fontSize = Math.round(9 * hRatio);
						ctx.font = `500 ${fontSize}px Inter, system-ui, sans-serif`;
						ctx.fillStyle = `rgba(${rgb}, ${markerAlpha * 0.9})`;
						const label = _shortExitReason(rect.trade.exit_reason);
						ctx.fillText(
							label,
							x2 + Math.round(5 * hRatio),
							exitY + Math.round(3 * vRatio),
						);
					}
				}
			}
		});
	}
}

function _shortExitReason(reason: string): string {
	const map: Record<string, string> = {
		stop_loss: 'SL',
		stoploss: 'SL',
		trailing_stop_loss: 'TSL',
		exit_signal: 'sig',
		roi: 'ROI',
		force_exit: 'force',
	};
	return map[reason] ?? reason;
}

// ── Pane View ──────────────────────────────────────────────────────

class TradeRectanglePaneView implements IPrimitivePaneView {
	private _renderer = new TradeRectangleRenderer();
	private _trades: Trade[] = [];
	private _hoveredIndex: number | null = null;
	private _attached: SeriesAttachedParameter<Time> | null = null;

	/** Last computed rects, exposed for hit-testing. */
	lastRects: TradeRect[] = [];

	setData(trades: Trade[]): void {
		this._trades = trades;
	}

	setHoveredIndex(index: number | null): void {
		this._hoveredIndex = index;
	}

	setAttached(params: SeriesAttachedParameter<Time>): void {
		this._attached = params;
	}

	renderer(): IPrimitivePaneRenderer {
		if (!this._attached) return this._renderer;

		const timeScale = this._attached.chart.timeScale();
		const series = this._attached.series;
		const rects: TradeRect[] = [];

		for (const trade of this._trades) {
			const x1 = timeScale.timeToCoordinate(trade.open_time as unknown as Time);
			const x2 = timeScale.timeToCoordinate(trade.close_time as unknown as Time);
			const yOpen = series.priceToCoordinate(trade.open_rate);
			const yClose = series.priceToCoordinate(trade.close_rate);

			if (x1 === null || x2 === null || yOpen === null || yClose === null) continue;

			rects.push({
				trade,
				x1,
				x2,
				y1: Math.min(yOpen, yClose), // top of rect on screen
				y2: Math.max(yOpen, yClose), // bottom of rect on screen
				isWin: trade.profit_ratio > 0,
			});
		}

		this.lastRects = rects;
		this._renderer.update(rects, this._hoveredIndex);
		return this._renderer;
	}
}

// ── Primitive ──────────────────────────────────────────────────────

export class TradeRectanglePrimitive implements ISeriesPrimitive<Time> {
	private _paneView = new TradeRectanglePaneView();
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

	setData(trades: Trade[]): void {
		this._paneView.setData(trades);
		this._requestUpdate?.();
	}

	setHoveredIndex(index: number | null): void {
		this._paneView.setHoveredIndex(index);
		this._requestUpdate?.();
	}

	/**
	 * Find which trade (if any) contains the given logical coordinates.
	 * Uses the last rendered rects from the pane view.
	 */
	findTradeAtCoordinate(x: number, y: number): { trade: Trade; index: number } | null {
		for (let i = 0; i < this._paneView.lastRects.length; i++) {
			const r = this._paneView.lastRects[i];
			if (x >= r.x1 && x <= r.x2 && y >= r.y1 && y <= r.y2) {
				return { trade: r.trade, index: i };
			}
		}
		return null;
	}
}
