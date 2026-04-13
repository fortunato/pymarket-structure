"""Multi-timeframe wrapper — project HTF market structure onto LTF DataFrames.

Resamples a lower-timeframe DataFrame to a higher timeframe, runs
``attach_market_structure`` on the HTF frame, shifts values forward by one
HTF period to prevent lookahead, and merges back via forward-fill.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

from market_structure.freqtrade import attach_market_structure

if TYPE_CHECKING:
    from market_structure.helper import MarketStructureHelper


def attach_market_structure_mtf(
    df: pd.DataFrame,
    metadata: dict[str, str],
    store: dict[str, MarketStructureHelper],
    *,
    htf: str,
    ltf: str,
    hist_col: str = "tsi_hist",
    columns: tuple[str, ...] | None = None,
    max_waves: int = 200,
    atr_period: int = 14,
    retest_mode: str = "wick",
    auto_histogram: bool = False,
    tsi_r: int = 12,
    tsi_s: int = 8,
    tsi_signal_period: int = 4,
) -> pd.DataFrame:
    """Project HTF market structure onto an LTF DataFrame.

    Args:
        df: LTF OHLCV DataFrame with a datetime index or ``date`` column.
        metadata: Must contain ``"pair"`` key.
        store: Caller-owned helper store (keyed by ``"{pair}_{htf}"``).
        htf: Higher timeframe as a pandas offset alias (e.g. ``"4h"``).
        ltf: Lower timeframe as a pandas offset alias (e.g. ``"1h"``).
        hist_col: Histogram column name on the LTF frame.
        columns: Which columns to project (short names).
        max_waves: Maximum confirmed waves.
        atr_period: ATR lookback period.
        retest_mode: Zone retest detection mode.
        auto_histogram: Compute TSI automatically.
        tsi_r: TSI long smoothing period.
        tsi_s: TSI short smoothing period.
        tsi_signal_period: TSI signal line period.

    Returns:
        The LTF DataFrame with ``ms_{htf}_*`` columns added.

    Note:
        HTF values are shifted forward by one HTF period, so the forming
        (incomplete) HTF candle is never visible to LTF bars.  In live
        mode this means strategies always see structure from the most
        recently *closed* HTF candle, not the one currently forming.

    Raises:
        ValueError: If HTF is not an even multiple of LTF.
    """
    htf_td = pd.Timedelta(htf)
    ltf_td = pd.Timedelta(ltf)

    if htf_td <= ltf_td:
        msg = f"HTF ({htf}) must be greater than LTF ({ltf})"
        raise ValueError(msg)
    if htf_td % ltf_td != pd.Timedelta(0):  # type: ignore[operator]
        msg = f"HTF ({htf}) must be an even multiple of LTF ({ltf})"
        raise ValueError(msg)

    # Ensure datetime index for resampling.
    has_date_col = "date" in df.columns
    if has_date_col:
        date_series = pd.to_datetime(df["date"])
    elif isinstance(df.index, pd.DatetimeIndex):
        date_series = df.index.to_series()
    else:
        msg = "DataFrame must have a 'date' column or DatetimeIndex for MTF resampling"
        raise ValueError(msg)

    # Resample LTF → HTF.
    df_for_resample = df.copy()
    df_for_resample.index = pd.DatetimeIndex(date_series)

    htf_df = (
        df_for_resample.resample(htf)
        .agg(
            {
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
                **(
                    {hist_col: "last"}
                    if hist_col in df_for_resample.columns and not auto_histogram
                    else {}
                ),
            }
        )
        .dropna(subset=["open"])  # type: ignore[call-overload]
    )

    # Add open_time for hydration.
    htf_df["open_time"] = htf_df.index.astype("int64") // 10**6

    # Preserve the DatetimeIndex — attach_market_structure resets it.
    htf_datetime_index = htf_df.index.copy()

    # Run structure analysis on HTF.
    htf_store_key = f"{metadata['pair']}_{htf}"
    htf_store: dict[str, MarketStructureHelper] = {}
    if htf_store_key in store:
        htf_store[metadata["pair"]] = store[htf_store_key]

    htf_result, htf_helper = attach_market_structure(
        htf_df,
        metadata,
        htf_store,
        hist_col=hist_col,
        columns=columns,
        max_waves=max_waves,
        atr_period=atr_period,
        retest_mode=retest_mode,
        auto_histogram=auto_histogram,
        tsi_r=tsi_r,
        tsi_s=tsi_s,
        tsi_signal_period=tsi_signal_period,
    )
    store[htf_store_key] = htf_helper

    # Shift HTF columns forward by one HTF period to prevent lookahead.
    ms_cols = [c for c in htf_result.columns if c.startswith("ms_")]
    htf_shifted = htf_result[ms_cols].copy()
    htf_shifted.index = htf_datetime_index[: len(htf_shifted)] + htf_td

    # Merge back to LTF via forward-fill.  Reuse the DatetimeIndex built
    # earlier for resampling instead of copying the full DataFrame again.
    ltf_index = pd.DatetimeIndex(date_series)

    for col in ms_cols:
        htf_col_name = col.replace("ms_", f"ms_{htf}_", 1)
        merged = htf_shifted[col].reindex(ltf_index, method="ffill")  # type: ignore[union-attr]
        df[htf_col_name] = merged.to_numpy()

    return df
