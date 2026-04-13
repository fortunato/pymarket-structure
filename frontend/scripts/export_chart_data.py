"""Export enriched market structure data + backtest trades as JSON.

Processes all OHLCV pairs in refs/freqtrade/ohlcv/, computes TSI,
runs attach_market_structure(), and writes enriched data to
frontend/src/assets/data/.  Also transforms backtest trade files
into frontend-friendly JSON with epoch-second timestamps.

Run from the pymarket-structure directory:
    uv run python frontend/scripts/export_chart_data.py
"""

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from market_structure.freqtrade import attach_market_structure

OHLCV_DIR = Path(__file__).resolve().parents[2] / "refs" / "freqtrade" / "ohlcv"
TRADES_DIR = Path(__file__).resolve().parents[2] / "refs" / "freqtrade" / "backtest_results"
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "src" / "assets" / "data"
HISTOGRAM_KEY = "tsi_histogram"


def _compute_tsi(
    close: pd.Series,  # type: ignore[type-arg]
    slow: int = 25,
    fast: int = 13,
    signal_len: int = 7,
) -> pd.DataFrame:
    """Compute TSI, signal line, and histogram from close prices."""
    diff = close.diff()
    smoothed = diff.ewm(span=slow, adjust=False).mean().ewm(span=fast, adjust=False).mean()
    abs_smoothed = (
        diff.abs().ewm(span=slow, adjust=False).mean().ewm(span=fast, adjust=False).mean()
    )
    tsi = 100 * smoothed / abs_smoothed
    signal = tsi.ewm(span=signal_len, adjust=False).mean()
    histogram = tsi - signal
    return pd.DataFrame({"tsi": tsi, "tsi_signal": signal, HISTOGRAM_KEY: histogram})


def _load_ohlcv(path: Path) -> pd.DataFrame:
    """Load an OHLCV JSON file and return a DataFrame ready for attach_market_structure."""
    with path.open() as f:
        raw = json.load(f)

    df = pd.DataFrame(raw)
    df["open_time"] = (
        pd.to_datetime(df["openTime"]).dt.tz_localize(None).astype("datetime64[ms]").astype("int64")
    )

    tsi_df = _compute_tsi(df["close"].astype(float))
    # Fill NaN warmup period with 0 so the histogram doesn't break wave detection
    tsi_df = tsi_df.fillna(0.0)
    df = pd.concat([df, tsi_df], axis=1)

    cols = [
        "open_time",
        "open",
        "high",
        "low",
        "close",
        "volume",
        HISTOGRAM_KEY,
        "tsi",
        "tsi_signal",
    ]
    return pd.DataFrame(df[cols])


def _serialize(df: pd.DataFrame) -> list[dict[str, object]]:
    """Convert enriched DataFrame to JSON-serializable dicts.

    - open_time (epoch ms) → time (epoch seconds) for Lightweight Charts
    - pd.NA / np.nan → None (becomes JSON null)
    """
    records: list[dict[str, object]] = []
    for _, row in df.iterrows():
        rec: dict[str, object] = {}
        rec["time"] = int(row["open_time"]) // 1000
        for col in ["open", "high", "low", "close", "volume", HISTOGRAM_KEY, "tsi", "tsi_signal"]:
            rec[col] = float(row[col])
        for col in df.columns:
            if not col.startswith("ms_"):
                continue
            val = row[col]
            if pd.isna(val):
                rec[col] = None
            elif col.endswith("_anchor_time"):
                rec[col] = int(float(val)) // 1000
            elif isinstance(val, (np.bool_, bool)):
                rec[col] = bool(val)
            elif isinstance(val, (np.integer, int)):
                rec[col] = int(val)
            elif isinstance(val, (np.floating, float)):
                rec[col] = float(val)
            else:
                rec[col] = val
        records.append(rec)
    return records


def _iso_to_epoch_seconds(iso_str: str) -> int:
    """Convert ISO 8601 date string (with or without timezone) to Unix epoch seconds."""
    # Handle both "2025-05-13 04:00:00+00:00" and "2025-05-13 04:00:00" formats
    iso_str = iso_str.strip()
    for fmt in ("%Y-%m-%d %H:%M:%S%z", "%Y-%m-%d %H:%M:%S"):
        try:
            dt = datetime.strptime(iso_str, fmt)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=UTC)
            return int(dt.timestamp())
        except ValueError:
            continue
    raise ValueError(f"Cannot parse date: {iso_str}")


def _pair_from_filename(filename: str) -> str:
    """Extract pair name from OHLCV filename, e.g. 'BTCUSDT-4h.json' → 'BTCUSDT'."""
    return filename.split("-")[0]


def _export_enriched_ohlcv() -> None:
    """Process all OHLCV files and write enriched JSON."""
    ohlcv_files = sorted(OHLCV_DIR.glob("*-4h.json"))
    print(f"Found {len(ohlcv_files)} OHLCV files")

    for ohlcv_path in ohlcv_files:
        pair = _pair_from_filename(ohlcv_path.name)
        print(f"\n--- {pair} ---")

        df = _load_ohlcv(ohlcv_path)
        print(f"  Loaded {len(df)} bars")

        store: dict[str, object] = {}
        df, _ = attach_market_structure(df, {"pair": pair}, store, hist_col=HISTOGRAM_KEY)
        print(f"  Enriched: {len(df.columns)} columns")

        records = _serialize(df)
        out_path = OUTPUT_DIR / f"{pair}-4h.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w") as f:
            json.dump(records, f)
        print(f"  Written {len(records)} records to {out_path.name}")


def _export_trades() -> None:
    """Transform backtest trade files into frontend-friendly JSON."""
    trades_out_dir = OUTPUT_DIR / "trades"
    trades_out_dir.mkdir(parents=True, exist_ok=True)

    trade_files = sorted(TRADES_DIR.glob("*-trades.json"))
    print(f"\nFound {len(trade_files)} trade files")

    for trade_path in trade_files:
        with trade_path.open() as f:
            raw = json.load(f)

        strategy = raw["strategy"]
        # Normalize pair: "BTC/USDT:USDT" → "BTCUSDT"
        pair = raw["pair"].split("/")[0] + raw["pair"].split("/")[1].split(":")[0]

        trades = [
            {
                "open_time": _iso_to_epoch_seconds(t["open_date"]),
                "close_time": _iso_to_epoch_seconds(t["close_date"]),
                "is_short": t["is_short"],
                "open_rate": t["open_rate"],
                "close_rate": t["close_rate"],
                "profit_ratio": t["profit_ratio"],
                "profit_abs": t["profit_abs"],
                "exit_reason": t["exit_reason"],
                "enter_tag": t["enter_tag"],
                "stake_amount": t["stake_amount"],
            }
            for t in raw["trades"]
        ]

        out = {"strategy": strategy, "pair": pair, "trades": trades}
        out_name = f"{strategy}-{pair}-4h.json"
        out_path = trades_out_dir / out_name
        with out_path.open("w") as f:
            json.dump(out, f)
        print(f"  {out_name}: {len(trades)} trades")


def main() -> None:
    _export_enriched_ohlcv()
    _export_trades()
    print("\nDone!")


if __name__ == "__main__":
    main()
