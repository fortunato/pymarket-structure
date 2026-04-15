export interface Trade {
	open_time: number; // epoch seconds
	close_time: number;
	is_short: boolean;
	open_rate: number;
	close_rate: number;
	profit_ratio: number;
	profit_abs: number;
	exit_reason: string;
	enter_tag: string;
	stake_amount: number;
}

export interface BacktestFile {
	strategy: string;
	pair: string;
	trades: Trade[];
}

/** Rendering-ready trade with pixel coordinates and derived fields. */
export interface TradeRect {
	trade: Trade;
	x1: number;
	x2: number;
	y1: number; // min(open_rate, close_rate) coordinate (top on screen)
	y2: number; // max(open_rate, close_rate) coordinate (bottom on screen)
	isWin: boolean;
}

export interface TradeStats {
	totalTrades: number;
	longCount: number;
	shortCount: number;
	winRate: number; // 0–100
	profitFactor: number;
	totalPnl: number; // sum of profit_abs
	maxDrawdown: number; // percentage of peak equity
	avgWinPct: number; // avg profit_ratio of winners * 100
	avgLossPct: number; // avg profit_ratio of losers * 100 (negative)
	avgDurationHours: number;
	expectancy: number; // expected return per trade as percentage
}

export const AVAILABLE_PAIRS = ['BTCUSDT', 'ETHUSDT', 'LTCUSDT', 'SOLUSDT', 'XRPUSDT'] as const;

export const AVAILABLE_STRATEGIES = [
	'MsFilterV6',
	'MsFilterV6NoMs',
	'MsSupportResistanceV1',
] as const;

export const STRATEGY_LABELS: Record<string, string> = {
	MsFilterV6: 'MS Filter V6',
	MsFilterV6NoMs: 'MS Filter V6 (No MS)',
	MsSupportResistanceV1: 'S&R V1',
};
