import {
	AfterViewInit,
	ChangeDetectionStrategy,
	Component,
	effect,
	ElementRef,
	inject,
	OnDestroy,
	signal,
	viewChild,
} from '@angular/core';
import {
	CandlestickSeries,
	ColorType,
	CrosshairMode,
	type IChartApi,
	type ISeriesApi,
	type ISeriesMarkersPluginApi,
	LineStyle,
	type IPriceLine,
	createChart,
	createSeriesMarkers,
	type SeriesMarker,
	type Time,
	HistogramSeries,
	LineSeries,
} from 'lightweight-charts';

import { EnrichedBar } from '../../models/candle-bar.model';
import { Trade } from '../../models/trade.model';
import { MarketDataService } from '../../services/market-data.service';
import { TradeDataService } from '../../services/trade-data.service';

interface DivergenceTooltip {
	x: number;
	y: number;
	type: 'bullish' | 'bearish';
	title: string;
	body: string;
}

export interface TradeTooltipState {
	trade: Trade;
	x: number;
	y: number;
	pinned: boolean;
}

const DIVERGENCE_TEXT = {
	bullish: {
		title: 'Selling Pressure Fading Into Lower Lows',
		body: 'Price printed a lower low but the TSI histogram trough was shallower \u2014 selling momentum is decelerating. Watch for a wave direction change and histogram crossing above zero as confirmation.',
	},
	bearish: {
		title: 'Momentum Weakening Into Higher Highs',
		body: 'Price printed a higher high but the TSI histogram peaked lower \u2014 buying momentum is decelerating. Watch for a wave direction change and histogram crossing below zero as confirmation.',
	},
} as const;
import { ChartStateService } from '../../services/chart-state.service';
import { ZoneRectanglePrimitive } from '../../plugins/zone-rectangle.primitive';
import { TrendBackgroundPrimitive } from '../../plugins/trend-background.primitive';
import { TradeRectanglePrimitive } from '../../plugins/trade-rectangle.primitive';
import { StructureBreakPrimitive } from '../../plugins/structure-break.primitive';
import { TradeTooltipComponent } from '../trade-tooltip/trade-tooltip.component';

@Component({
	selector: 'app-chart',
	imports: [TradeTooltipComponent],
	templateUrl: './chart.component.html',
	styleUrl: './chart.component.scss',
	changeDetection: ChangeDetectionStrategy.OnPush,
})
export class ChartComponent implements AfterViewInit, OnDestroy {
	private readonly marketData = inject(MarketDataService);
	private readonly tradeData = inject(TradeDataService);
	private readonly chartState = inject(ChartStateService);

	readonly chartContainer = viewChild.required<ElementRef<HTMLDivElement>>('chartContainer');
	readonly tooltip = signal<DivergenceTooltip | null>(null);
	readonly tradeTooltip = signal<TradeTooltipState | null>(null);

	private chart!: IChartApi;
	private candleSeries!: ISeriesApi<'Candlestick'>;
	private volumeSeries!: ISeriesApi<'Histogram'>;
	private tsiHistSeries!: ISeriesApi<'Histogram'>;
	private tsiLineSeries!: ISeriesApi<'Line'>;
	private tsiSignalSeries!: ISeriesApi<'Line'>;

	// Primitives
	private supportZonePrimitive = new ZoneRectanglePrimitive();
	private resistanceZonePrimitive = new ZoneRectanglePrimitive();
	private trendBackgroundPrimitive = new TrendBackgroundPrimitive();
	private tradeRectanglePrimitive = new TradeRectanglePrimitive();
	private structureBreakPrimitive = new StructureBreakPrimitive();

	// Markers plugin
	private markersPlugin!: ISeriesMarkersPluginApi<Time>;

	// Price lines (managed manually)
	private topPriceLine: IPriceLine | null = null;
	private bottomPriceLine: IPriceLine | null = null;
	private pinnedOpenLine: IPriceLine | null = null;
	private pinnedCloseLine: IPriceLine | null = null;

	private resizeObserver!: ResizeObserver;

	constructor() {
		// Data update effect
		effect(() => {
			const bars = this.marketData.bars();
			if (bars.length === 0 || !this.chart) return;

			this.setCandlestickData(bars);
			this.setVolumeData(bars);
			this.setTsiData(bars);

			// Clear state on pair switch
			this.chartState.activeBar.set(null);
			this.chartState.hoveredTrade.set(null);
			this.chartState.pinnedTrade.set(null);
			this.tradeTooltip.set(null);
			this.clearPinnedTradeLines();
		});

		// Overlay update effect
		effect(() => {
			const overlays = this.chartState.overlays();
			const bars = this.marketData.bars();
			if (bars.length === 0 || !this.chart) return;

			// Support zones
			if (overlays.supportZones) {
				this.supportZonePrimitive.setData(this.marketData.supportZones());
			} else {
				this.supportZonePrimitive.setData([]);
			}

			// Resistance zones
			if (overlays.resistanceZones) {
				this.resistanceZonePrimitive.setData(this.marketData.resistanceZones());
			} else {
				this.resistanceZonePrimitive.setData([]);
			}

			// Trend background
			if (overlays.trendBackground) {
				this.trendBackgroundPrimitive.setData(this.marketData.trendSpans());
			} else {
				this.trendBackgroundPrimitive.setData([]);
			}

			// Structure break level primitive
			if (overlays.structureBreak) {
				this.structureBreakPrimitive.setData(this.marketData.structureBreakSpans());
			} else {
				this.structureBreakPrimitive.setData([]);
			}

			// All markers merged into one array (sorted by time)
			const markers: SeriesMarker<Time>[] = [];

			if (overlays.waveTransitions) {
				for (const t of this.marketData.waveTransitions()) {
					markers.push({
						time: t.time as unknown as Time,
						position: t.newSide === 'up' ? 'belowBar' : 'aboveBar',
						color: t.newSide === 'up' ? '#26a69a' : '#ef5350',
						shape: 'circle',
						text: t.waveId,
					});
				}
			}

			if (overlays.divergenceMarkers) {
				for (const d of this.marketData.divergences()) {
					markers.push({
						time: d.time as unknown as Time,
						position: d.type === 'bullish' ? 'belowBar' : 'aboveBar',
						color: d.type === 'bullish' ? '#26a69a' : '#ef5350',
						shape: 'square',
						text: d.type === 'bullish' ? 'Bull Div' : 'Bear Div',
					});
				}
			}

			// Zone lifecycle event markers
			if (overlays.zoneLifecycle) {
				for (const e of this.marketData.lifecycleEvents()) {
					const isSup = e.zoneSide === 'support';
					const pos = isSup ? 'belowBar' : 'aboveBar';
					switch (e.eventType) {
						case 'break':
							markers.push({
								time: e.time as unknown as Time,
								position: pos,
								color: isSup ? '#e74c3c' : '#2ecc71',
								shape: isSup ? 'arrowDown' : 'arrowUp',
								text: 'ZB',
							});
							break;
						case 'retest':
							markers.push({
								time: e.time as unknown as Time,
								position: pos,
								color: '#f39c12',
								shape: 'circle',
								text: 'RT',
							});
							break;
						case 'flip':
							markers.push({
								time: e.time as unknown as Time,
								position: pos,
								color: '#9b59b6',
								shape: 'square',
								text: 'FL',
							});
							break;
						case 'failed_retest':
							markers.push({
								time: e.time as unknown as Time,
								position: pos,
								color: '#95a5a6',
								shape: 'circle',
								text: 'FR',
							});
							break;
					}
				}
			}

			// Structure break confirmed markers
			if (overlays.structureBreak) {
				for (const bar of bars) {
					if (bar.ms_structure_break_confirmed) {
						markers.push({
							time: bar.time as unknown as Time,
							position: 'inBar',
							color: '#e67e22',
							shape: bar.ms_is_trending_up ? 'arrowDown' : 'arrowUp',
							text: 'SB',
						});
					}
				}
			}

			// SFP and three-push pattern markers
			if (overlays.patterns) {
				for (const p of this.marketData.patternMarkers()) {
					switch (p.patternType) {
						case 'sfp_high':
							markers.push({
								time: p.time as unknown as Time,
								position: 'aboveBar',
								color: '#e74c3c',
								shape: 'square',
								text: 'SFP',
							});
							break;
						case 'sfp_low':
							markers.push({
								time: p.time as unknown as Time,
								position: 'belowBar',
								color: '#2ecc71',
								shape: 'square',
								text: 'SFP',
							});
							break;
						case 'three_push_up':
							markers.push({
								time: p.time as unknown as Time,
								position: 'aboveBar',
								color: '#e74c3c',
								shape: 'arrowDown',
								text: '3P',
							});
							break;
						case 'three_push_down':
							markers.push({
								time: p.time as unknown as Time,
								position: 'belowBar',
								color: '#2ecc71',
								shape: 'arrowUp',
								text: '3P',
							});
							break;
					}
				}
			}

			markers.sort((a, b) => (a.time as number) - (b.time as number));
			this.markersPlugin.setMarkers(markers);

			// Price lines
			this.updatePriceLines(overlays.priceLines ? this.chartState.activeBar() : null);
		});

		// Trade overlay effect
		effect(() => {
			const overlays = this.chartState.overlays();
			const trades = this.tradeData.trades();
			if (!this.chart) return;

			if (overlays.trades) {
				this.tradeRectanglePrimitive.setData(trades);
			} else {
				this.tradeRectanglePrimitive.setData([]);
			}
		});

		// Update price lines on active bar change
		effect(() => {
			const bar = this.chartState.activeBar();
			const overlays = this.chartState.overlays();
			if (!this.chart || !overlays.priceLines) return;
			this.updatePriceLines(bar);
		});

		// Pinned trade price lines
		effect(() => {
			const pinned = this.chartState.pinnedTrade();
			if (!this.chart) return;
			this.clearPinnedTradeLines();
			if (pinned) {
				this.pinnedOpenLine = this.candleSeries.createPriceLine({
					price: pinned.open_rate,
					color: 'rgba(66, 165, 245, 0.7)',
					lineWidth: 1,
					lineStyle: LineStyle.Dashed,
					axisLabelVisible: true,
					title: 'Entry',
				});
				this.pinnedCloseLine = this.candleSeries.createPriceLine({
					price: pinned.close_rate,
					color: 'rgba(66, 165, 245, 0.7)',
					lineWidth: 1,
					lineStyle: LineStyle.Dashed,
					axisLabelVisible: true,
					title: 'Exit',
				});
			}
		});
	}

	ngAfterViewInit(): void {
		this.initChart();
	}

	ngOnDestroy(): void {
		this.resizeObserver?.disconnect();
		this.chart?.remove();
	}

	private initChart(): void {
		const container = this.chartContainer().nativeElement;

		this.chart = createChart(container, {
			layout: {
				background: { type: ColorType.Solid, color: '#1a1a2e' },
				textColor: '#a0a0b0',
				fontFamily: "'Inter', system-ui, sans-serif",
			},
			grid: {
				vertLines: { color: '#2a2a3e' },
				horzLines: { color: '#2a2a3e' },
			},
			crosshair: {
				mode: CrosshairMode.Normal,
			},
			timeScale: {
				timeVisible: true,
				secondsVisible: false,
				borderColor: '#2a2a4a',
			},
			rightPriceScale: {
				borderColor: '#2a2a4a',
			},
		});

		// Candlestick series
		this.candleSeries = this.chart.addSeries(CandlestickSeries, {
			upColor: '#26a69a',
			downColor: '#ef5350',
			borderUpColor: '#26a69a',
			borderDownColor: '#ef5350',
			wickUpColor: '#26a69a',
			wickDownColor: '#ef5350',
		});

		// Markers plugin (v5 API)
		this.markersPlugin = createSeriesMarkers(this.candleSeries);

		// Volume histogram (overlay on same pane, scaled down)
		this.volumeSeries = this.chart.addSeries(HistogramSeries, {
			priceFormat: { type: 'volume' },
			priceScaleId: 'volume',
		});
		this.chart.priceScale('volume').applyOptions({
			scaleMargins: { top: 0.85, bottom: 0 },
		});

		// TSI pane (separate pane below candles)
		const tsiPane = this.chart.addPane();

		this.tsiHistSeries = tsiPane.addSeries(HistogramSeries, {
			priceFormat: { type: 'price', precision: 2, minMove: 0.01 },
			priceLineVisible: false,
			lastValueVisible: false,
		});

		this.tsiLineSeries = tsiPane.addSeries(LineSeries, {
			color: '#42a5f5',
			lineWidth: 1,
			priceLineVisible: false,
			lastValueVisible: false,
		});

		this.tsiSignalSeries = tsiPane.addSeries(LineSeries, {
			color: '#ff9800',
			lineWidth: 1,
			lineStyle: LineStyle.Dashed,
			priceLineVisible: false,
			lastValueVisible: false,
		});

		// Zero line on TSI pane
		this.tsiHistSeries.createPriceLine({
			price: 0,
			color: 'rgba(160, 160, 176, 0.3)',
			lineWidth: 1,
			lineStyle: LineStyle.Solid,
			axisLabelVisible: false,
			title: '',
		});

		// Attach primitives
		this.candleSeries.attachPrimitive(this.supportZonePrimitive);
		this.candleSeries.attachPrimitive(this.resistanceZonePrimitive);
		this.candleSeries.attachPrimitive(this.trendBackgroundPrimitive);
		this.candleSeries.attachPrimitive(this.tradeRectanglePrimitive);
		this.candleSeries.attachPrimitive(this.structureBreakPrimitive);

		// Crosshair handler — active bar + trade hover detection
		// When cursor leaves the chart, keep the last bar sticky so the info panel
		// remains interactive (collapsible sections can be clicked).
		this.chart.subscribeCrosshairMove((param) => {
			if (!param.time) {
				this.tooltip.set(null);
				// Don't clear hovered trade here — let mouse detection handle it
				if (!this.chartState.pinnedTrade()) {
					this.chartState.hoveredTrade.set(null);
					this.tradeRectanglePrimitive.setHoveredIndex(null);
					this.tradeTooltip.set(null);
				}
				return;
			}
			const bars = this.marketData.bars();
			const bar = bars.find((b) => b.time === param.time);
			this.chartState.activeBar.set(bar ?? null);
			this.updateDivergenceTooltip(bar ?? null, param);

			// Trade hit testing
			if (param.point && this.chartState.overlays().trades) {
				const hit = this.tradeRectanglePrimitive.findTradeAtCoordinate(
					param.point.x,
					param.point.y,
				);
				if (hit) {
					this.chartState.hoveredTrade.set(hit.trade);
					this.tradeRectanglePrimitive.setHoveredIndex(hit.index);
					if (!this.chartState.pinnedTrade()) {
						this.tradeTooltip.set({
							trade: hit.trade,
							x: param.point.x,
							y: param.point.y,
							pinned: false,
						});
					}
				} else {
					this.chartState.hoveredTrade.set(null);
					this.tradeRectanglePrimitive.setHoveredIndex(null);
					if (!this.chartState.pinnedTrade()) {
						this.tradeTooltip.set(null);
					}
				}
			}
		});

		// Click handler — pin/unpin trades
		this.chart.subscribeClick((param) => {
			if (!param.point || !this.chartState.overlays().trades) return;

			const hit = this.tradeRectanglePrimitive.findTradeAtCoordinate(
				param.point.x,
				param.point.y,
			);

			const currentPinned = this.chartState.pinnedTrade();

			if (hit) {
				// Toggle: unpin if same trade, otherwise pin new trade
				if (
					currentPinned &&
					currentPinned.open_time === hit.trade.open_time &&
					currentPinned.close_time === hit.trade.close_time
				) {
					this.chartState.pinnedTrade.set(null);
					this.tradeTooltip.set(null);
				} else {
					this.chartState.pinnedTrade.set(hit.trade);
					this.tradeTooltip.set({
						trade: hit.trade,
						x: param.point.x,
						y: param.point.y,
						pinned: true,
					});
				}
			} else if (currentPinned) {
				// Click outside any trade → unpin
				this.chartState.pinnedTrade.set(null);
				this.tradeTooltip.set(null);
			}
		});

		// Responsive resize
		this.resizeObserver = new ResizeObserver((entries) => {
			for (const entry of entries) {
				const { width, height } = entry.contentRect;
				this.chart.applyOptions({ width, height });
			}
		});
		this.resizeObserver.observe(container);
	}

	private setCandlestickData(bars: EnrichedBar[]): void {
		this.candleSeries.setData(
			bars.map((b) => ({
				time: b.time as unknown as Time,
				open: b.open,
				high: b.high,
				low: b.low,
				close: b.close,
			})),
		);
		this.chart.timeScale().fitContent();
	}

	private setVolumeData(bars: EnrichedBar[]): void {
		this.volumeSeries.setData(
			bars.map((b) => ({
				time: b.time as unknown as Time,
				value: b.volume,
				color:
					b.ms_wave_side === 'up' ? 'rgba(38, 166, 154, 0.3)' : 'rgba(239, 83, 80, 0.3)',
			})),
		);
	}

	private setTsiData(bars: EnrichedBar[]): void {
		this.tsiHistSeries.setData(
			bars.map((b) => ({
				time: b.time as unknown as Time,
				value: b.tsi_histogram,
				color: b.tsi_histogram >= 0 ? 'rgba(38, 166, 154, 0.5)' : 'rgba(239, 83, 80, 0.5)',
			})),
		);

		this.tsiLineSeries.setData(
			bars.map((b) => ({
				time: b.time as unknown as Time,
				value: b.tsi,
			})),
		);

		this.tsiSignalSeries.setData(
			bars.map((b) => ({
				time: b.time as unknown as Time,
				value: b.tsi_signal,
			})),
		);
	}

	private updateDivergenceTooltip(
		bar: EnrichedBar | null,
		param: { point?: { x: number; y: number } },
	): void {
		if (!bar || !param.point || (!bar.ms_bearish_divergence && !bar.ms_bullish_divergence)) {
			this.tooltip.set(null);
			return;
		}
		const type = bar.ms_bearish_divergence ? 'bearish' : 'bullish';
		const text = DIVERGENCE_TEXT[type];
		this.tooltip.set({
			x: param.point.x,
			y: param.point.y,
			type,
			title: text.title,
			body: text.body,
		});
	}

	private updatePriceLines(bar: EnrichedBar | null): void {
		if (this.topPriceLine) {
			this.candleSeries.removePriceLine(this.topPriceLine);
			this.topPriceLine = null;
		}
		if (this.bottomPriceLine) {
			this.candleSeries.removePriceLine(this.bottomPriceLine);
			this.bottomPriceLine = null;
		}

		if (!bar) return;

		if (bar.ms_last_top_price !== null) {
			this.topPriceLine = this.candleSeries.createPriceLine({
				price: bar.ms_last_top_price,
				color: 'rgba(239, 83, 80, 0.6)',
				lineWidth: 1,
				lineStyle: LineStyle.Dashed,
				axisLabelVisible: true,
				title: 'Last Top',
			});
		}

		if (bar.ms_last_bottom_price !== null) {
			this.bottomPriceLine = this.candleSeries.createPriceLine({
				price: bar.ms_last_bottom_price,
				color: 'rgba(38, 166, 154, 0.6)',
				lineWidth: 1,
				lineStyle: LineStyle.Dashed,
				axisLabelVisible: true,
				title: 'Last Bottom',
			});
		}
	}

	private clearPinnedTradeLines(): void {
		if (this.pinnedOpenLine) {
			this.candleSeries.removePriceLine(this.pinnedOpenLine);
			this.pinnedOpenLine = null;
		}
		if (this.pinnedCloseLine) {
			this.candleSeries.removePriceLine(this.pinnedCloseLine);
			this.pinnedCloseLine = null;
		}
	}
}
