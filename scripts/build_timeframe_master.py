#!/usr/bin/env python3
"""
SOVEREIGN KRAKEN — Timeframe Master Builder
============================================
Fetches native OHLCV candles at a given timeframe and augments them with
order-book-proxy columns via fetch_live_kat_data's existing fetch pipeline,
producing a master parquet with the same schema as the other timeframes.

Usage:
    python scripts/build_timeframe_master.py --timeframe 4h --candles 8000
"""
import sys
import argparse
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from exchange.fetch_data import fetch_live_kat_data


def build(symbol, timeframe, n_candles):
    print(f"📡 Building {symbol} {timeframe} master ({n_candles:,} candles requested)...")
    df = fetch_live_kat_data(symbol=symbol, n_candles=n_candles, timeframe=timeframe)

    # Delta's REST candles are OHLCV-only — approximate the quote/taker split
    # neutrally (50/50), matching the convention used for the existing masters.
    df["quote_volume"] = df["volume"] * df["close"]
    df["taker_buy_volume"] = df["volume"] * 0.5
    df["taker_buy_quote_volume"] = df["quote_volume"] * 0.5

    out_path = ROOT / "data" / f"{symbol}_{timeframe}_history_master.parquet"
    df.to_parquet(out_path)
    print(f"✅ Saved {len(df):,} candles ({df.index.min()} -> {df.index.max()}) to {out_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--symbol", default="BTCUSD")
    p.add_argument("--timeframe", required=True)
    p.add_argument("--candles", type=int, required=True)
    args = p.parse_args()
    build(args.symbol, args.timeframe, args.candles)
