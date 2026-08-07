"""Converts a downloaded Kaggle Nifty50/NSE minute-data CSV (or one CSV per
symbol) into the standard per-symbol format the CsvBroker reads.

Usage:
    1. Download a dataset e.g. one of:
         https://www.kaggle.com/datasets/aaditya555/nifty-50-minute-data
         https://www.kaggle.com/datasets/debashis74017/stock-market-data-nifty-50-stocks-1-min-data
       (free Kaggle account required to download, no API needed - just use
       the website's Download button.)
    2. Unzip and place the raw CSV(s) in data/kaggle_raw/
    3. python3 data/import_kaggle_data.py

Handles two common layouts, auto-detected per file:
    - One big CSV with a symbol/ticker column covering many stocks.
    - One CSV per symbol, named after the ticker (e.g. RELIANCE.csv,
      reliance_minute.csv - the symbol is inferred from the filename).

Column names are matched case-insensitively against common aliases (e.g.
"timestamp" or "datetime" for the date column) since Kaggle datasets rarely
use identical schemas. Output always lands at cache/kaggle/<SYMBOL>.csv with
columns date,open,high,low,close,volume, sorted and deduplicated - ready for
common/brokers/csv_broker.py to read.
"""
import re
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent.parent  # kat/
sys.path.insert(0, str(ROOT / "src"))

RAW_DIR = ROOT / "data" / "kaggle_raw"
OUT_DIR = ROOT / "cache" / "kaggle"

COLUMN_ALIASES = {
    "date": ["date", "datetime", "timestamp", "time"],
    "open": ["open", "o"],
    "high": ["high", "h"],
    "low": ["low", "l"],
    "close": ["close", "c", "last", "ltp"],
    "volume": ["volume", "vol", "v", "qty"],
    "symbol": ["symbol", "ticker", "tradingsymbol", "scrip", "stock", "name"],
}


def find_column(columns, aliases):
    lower_map = {c.lower().strip(): c for c in columns}
    for alias in aliases:
        if alias in lower_map:
            return lower_map[alias]
    return None


def infer_symbol_from_filename(path: Path) -> str:
    stem = path.stem.upper()
    stem = re.sub(r"[_\-]?(MINUTE|MIN|DATA|1MIN|5MIN|INTRADAY)[_\-]?", "", stem)
    return stem.strip("_- ") or path.stem.upper()


def load_config():
    with open(ROOT / "config" / "settings.yaml") as f:
        return yaml.safe_load(f)


def process_file(path: Path) -> dict:
    """Returns {symbol: row_count} for whatever this file contributed."""
    try:
        df = pd.read_csv(path)
    except Exception as exc:
        print(f"  SKIPPING {path.name}: could not read as CSV ({exc})")
        return {}

    resolved = {}
    for key, aliases in COLUMN_ALIASES.items():
        col = find_column(df.columns, aliases)
        if col:
            resolved[key] = col

    required = ["date", "open", "high", "low", "close", "volume"]
    missing = [r for r in required if r not in resolved]
    if missing:
        print(f"  SKIPPING {path.name}: couldn't find columns for {missing} "
              f"(file has: {list(df.columns)})")
        return {}

    df = df.rename(columns={v: k for k, v in resolved.items()})
    df["date"] = pd.to_datetime(df["date"])

    if "symbol" in resolved:
        groups = list(df.groupby("symbol"))
    else:
        groups = [(infer_symbol_from_filename(path), df)]

    counts = {}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for symbol, group in groups:
        symbol = str(symbol).strip().upper()
        out = group[["date", "open", "high", "low", "close", "volume"]].copy()
        out = out.sort_values("date").drop_duplicates(subset="date")

        out_path = OUT_DIR / f"{symbol}.csv"
        if out_path.exists():
            existing = pd.read_csv(out_path, parse_dates=["date"])
            out = pd.concat([existing, out]).drop_duplicates(subset="date").sort_values("date")
        out.to_csv(out_path, index=False)
        counts[symbol] = len(out)
    return counts


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    files = sorted(RAW_DIR.glob("*.csv"))
    if not files:
        print(f"No CSV files found in {RAW_DIR}.")
        print("Download a Kaggle Nifty50 minute-data dataset, unzip it, and place")
        print("the CSV(s) there, then re-run this script.")
        sys.exit(1)

    all_counts = {}
    for path in files:
        print(f"Processing {path.name}...")
        counts = process_file(path)
        for symbol, n in counts.items():
            print(f"  {symbol}: {n} candles")
        all_counts.update(counts)

    if not all_counts:
        print("\nNo usable data imported - check the column-detection warnings above.")
        sys.exit(1)

    cfg = load_config()
    universe = set(cfg["universe"]["symbols"])
    imported = set(all_counts.keys())
    missing = universe - imported

    print(f"\nImported {len(imported)} symbol(s) into {OUT_DIR}/")
    if missing:
        print(f"Not covered by this import (still in config/settings.yaml universe): "
              f"{sorted(missing)}")
        print("Either source their data too, or remove them from universe.symbols.")

    print("\nSet broker.name: csv in config/settings.yaml, then run:")
    print("    python3 auto_run.py backtest")


if __name__ == "__main__":
    main()
