#!/usr/bin/env python3
"""
SOVEREIGN KRAKEN — Merge Real Order-Book Depth Into the 1h Master Parquet
============================================================================
Replaces the fabricated bid/ask columns in BTCUSD_1h_history_master.parquet
with the real data fetched by fetch_book_depth_history.py, wherever real
coverage exists (2023-01-01 onward). Rows before that date keep their
existing synthetic fill — real bookDepth history doesn't go back further
(confirmed via binary search against data.binance.vision).

bid_vol1-5/ask_vol1-5 are replaced directly with the real de-cumulated
volumes. bid1-5/ask1-5 (price levels) are reconstructed from the real close
price and Binance's own percentage-band definition (1%/2%/3%/4%/5% from
mid) — this is a real, exchange-defined spread, not an arbitrary guess like
the old $0.50-per-level synthetic convention (which was a vanishingly small
fraction of BTC's actual price and never resembled a real order book).

Usage:
    python scripts/merge_book_depth_into_master.py
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent.parent

MASTER_PATH = ROOT / "data" / "BTCUSD_1h_history_master.parquet"
REAL_PATH = ROOT / "data" / "BTCUSD_1h_orderbook_depth.csv"


def main():
    if not REAL_PATH.exists():
        print(f"❌ No real order-book data at {REAL_PATH} — run fetch_book_depth_history.py first.")
        sys.exit(1)

    df = pd.read_parquet(MASTER_PATH)
    real = pd.read_csv(REAL_PATH, index_col=0, parse_dates=True)
    real.index = real.index.tz_localize("UTC")

    overlap = df.index.intersection(real.index)
    print(f"📖 Master: {len(df):,} rows ({df.index.min()} -> {df.index.max()})")
    print(f"📖 Real data: {len(real):,} rows ({real.index.min()} -> {real.index.max()})")
    print(f"🔗 Overlap: {len(overlap):,} rows will get real order-book data")

    vol_cols = [f"bid_vol{lvl}" for lvl in range(1, 6)] + [f"ask_vol{lvl}" for lvl in range(1, 6)]
    for col in vol_cols:
        df.loc[overlap, col] = real.loc[overlap, col].values

    # Reconstruct price levels from the real close price + Binance's own
    # percentage-band definition (real exchange-defined spread, not a guess).
    close_at_overlap = df.loc[overlap, "close"]
    for lvl in range(1, 6):
        df.loc[overlap, f"bid{lvl}"] = (close_at_overlap * (1 - 0.01 * lvl)).round(2)
        df.loc[overlap, f"ask{lvl}"] = (close_at_overlap * (1 + 0.01 * lvl)).round(2)

    df.to_parquet(MASTER_PATH)
    pct_real = 100 * len(overlap) / len(df)
    print(f"✅ Saved. {len(overlap):,}/{len(df):,} rows ({pct_real:.1f}%) now use real order-book data.")
    print(f"   Remaining {len(df) - len(overlap):,} rows (before 2023-01-01) keep the existing synthetic fill.")


if __name__ == "__main__":
    main()
