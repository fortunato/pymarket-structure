"""Export enriched market structure data as JSON for the Angular chart viewer.

Reads the test fixture, runs attach_market_structure(), and writes the
enriched DataFrame to frontend/src/assets/data/LTCUSDT-4h.json.

Run from the pymarket-structure directory:
    uv run python frontend/scripts/export_chart_data.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from market_structure.freqtrade import attach_market_structure

FIXTURE_PATH = (
    Path(__file__).resolve().parents[1] / ".." / "tests" / "fixtures" / "ms-LTCUSDT-4h.json"
)
OUTPUT_PATH = Path(__file__).resolve().parents[1] / "src" / "assets" / "data" / "LTCUSDT-4h.json"
HISTOGRAM_KEY = "tsi_histogram"


def _to_dataframe(rows: list[dict[str, object]]) -> pd.DataFrame:
    """Mirror the test helper — same conversion as test_freqtrade.py:38-44."""
    df = pd.DataFrame(rows)
    df["open_time"] = (
        pd.to_datetime(df["openTime"]).dt.tz_localize(None).astype("datetime64[ms]").astype("int64")
    )
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
                # Convert epoch ms → epoch seconds to match LWC time format
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


def main() -> None:
    fixture_path = FIXTURE_PATH.resolve()
    print(f"Reading fixture: {fixture_path}")
    with fixture_path.open() as f:
        raw = json.load(f)

    df = _to_dataframe(raw)
    print(f"DataFrame: {len(df)} bars, columns: {list(df.columns)}")

    store: dict[str, object] = {}
    df, _ = attach_market_structure(df, {"pair": "LTCUSDT"}, store, hist_col=HISTOGRAM_KEY)
    print(f"Enriched: {len(df.columns)} columns total")

    records = _serialize(df)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w") as f:
        json.dump(records, f, indent=2)
    print(f"Written {len(records)} records to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
