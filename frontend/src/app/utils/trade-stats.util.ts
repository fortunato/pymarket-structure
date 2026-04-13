import { Trade, TradeStats } from '../models/trade.model';

/**
 * Compute aggregate statistics from a list of trades.
 *
 * Metric definitions follow quant conventions:
 * - Win Rate: profit_ratio > 0 (zero-profit is NOT a win)
 * - Profit Factor: gross wins / abs(gross losses)
 * - Max Drawdown: peak-to-trough on cumulative equity curve
 * - Expectancy: (winRate * avgWin) + ((1 - winRate) * avgLoss)
 */
export function computeStats(trades: Trade[]): TradeStats {
	const empty: TradeStats = {
		totalTrades: 0,
		longCount: 0,
		shortCount: 0,
		winRate: 0,
		profitFactor: 0,
		totalPnl: 0,
		maxDrawdown: 0,
		avgWinPct: 0,
		avgLossPct: 0,
		avgDurationHours: 0,
		expectancy: 0,
	};

	if (trades.length === 0) return empty;

	const winners = trades.filter((t) => t.profit_ratio > 0);
	const losers = trades.filter((t) => t.profit_ratio <= 0);

	const winRate = winners.length / trades.length;
	const grossWin = winners.reduce((s, t) => s + t.profit_abs, 0);
	const grossLoss = Math.abs(losers.reduce((s, t) => s + t.profit_abs, 0));
	const profitFactor = grossLoss > 0 ? grossWin / grossLoss : grossWin > 0 ? Infinity : 0;

	const avgWinPct =
		winners.length > 0
			? (winners.reduce((s, t) => s + t.profit_ratio, 0) / winners.length) * 100
			: 0;
	const avgLossPct =
		losers.length > 0
			? (losers.reduce((s, t) => s + t.profit_ratio, 0) / losers.length) * 100
			: 0;

	// Max drawdown from cumulative equity curve (trades ordered by close_time)
	// Use 1000 USDT as starting capital (matches backtest config)
	const startingCapital = 1000;
	const sorted = [...trades].sort((a, b) => a.close_time - b.close_time);
	let equity = startingCapital;
	let peak = startingCapital;
	let maxDd = 0;
	for (const t of sorted) {
		equity += t.profit_abs;
		if (equity > peak) peak = equity;
		const dd = (peak - equity) / peak;
		if (dd > maxDd) maxDd = dd;
	}

	// Average duration in hours
	const totalDuration = trades.reduce((s, t) => s + (t.close_time - t.open_time), 0);
	const avgDurationHours = totalDuration / trades.length / 3600;

	// Expectancy: (winRate * avgWin) + ((1 - winRate) * avgLoss)
	const expectancy = winRate * avgWinPct + (1 - winRate) * avgLossPct;

	return {
		totalTrades: trades.length,
		longCount: trades.filter((t) => !t.is_short).length,
		shortCount: trades.filter((t) => t.is_short).length,
		winRate: winRate * 100,
		profitFactor,
		totalPnl: trades.reduce((s, t) => s + t.profit_abs, 0),
		maxDrawdown: maxDd * 100,
		avgWinPct,
		avgLossPct,
		avgDurationHours,
		expectancy,
	};
}
