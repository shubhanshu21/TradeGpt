#!/usr/bin/env python3
"""
SOVEREIGN KRAKEN — Real Order-Book Depth Backfill
====================================================
Fetches Binance's free public bookDepth archive (data.binance.vision) — real
market depth at 5 percentage bands per side (-5%..-1% bid, +1%..+5% ask),
sampled every ~30 seconds — and resamples it to 1h, replacing the fabricated
bid_vol1-5/ask_vol1-5 columns in the master parquet with genuine data.

bookDepth's "depth" field is CUMULATIVE from the mid price out to that band
(band -1 = liquidity within 1% of mid, band -5 = liquidity within 5%), not a
literal per-level size — this script de-cumulates each band into an
incremental volume before mapping it onto our bidN/askN (N=1..5) convention,
where N=1 is nearest the touch. Confirmed via binary search that real data
starts 2023-01-01 (404 before that date); our 1h master starts 2022-12-07,
so the ~25-day gap at the very start keeps its existing synthetic fill.

Resumable: re-running only fetches days missing from the output file, so a
partial/interrupted run picks up where it left off rather than restarting.

Usage:
    python scripts/fetch_book_depth_history.py
"""
import sys
import io
import time
import zipfile
from pathlib import Path
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).parent.parent

SYMBOL = "BTCUSDT"
START_DATE = "2023-01-01"  # confirmed earliest available via binary search
BASE_URL = f"https://data.binance.vision/data/futures/um/daily/bookDepth/{SYMBOL}"
OUT_PATH = ROOT / "data" / "BTCUSD_1h_orderbook_depth.csv"

BID_BANDS = [-1, -2, -3, -4, -5]
ASK_BANDS = [1, 2, 3, 4, 5]


def fetch_day(date_str: str):
    """Download and parse one day's raw bookDepth CSV. Returns None on 404
    (e.g. today's not-yet-published data) or any transient failure."""
    url = f"{BASE_URL}/{SYMBOL}-bookDepth-{date_str}.zip"
    try:
        r = requests.get(url, timeout=60)
        if r.status_code != 200:
            return None
        z = zipfile.ZipFile(io.BytesIO(r.content))
        name = z.namelist()[0]
        return pd.read_csv(z.open(name))
    except Exception as e:
        print(f"   ⚠️ {date_str} failed: {e}")
        return None


def decumulate_to_hourly(df: pd.DataFrame) -> pd.DataFrame:
    """Pivot percentage bands into per-timestamp incremental bid/ask volumes
    (undoing the cumulative-from-mid convention), then resample to 1h."""
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    pivot = df.pivot_table(index="timestamp", columns="percentage", values="depth", aggfunc="last")
    for c in BID_BANDS + ASK_BANDS:
        if c not in pivot.columns:
            pivot[c] = np.nan
    pivot = pivot.ffill()

    out = pd.DataFrame(index=pivot.index)
    prev = 0.0
    for lvl, c in enumerate(BID_BANDS, start=1):
        out[f"bid_vol{lvl}"] = (pivot[c] - prev).clip(lower=0)
        prev = pivot[c]
    prev = 0.0
    for lvl, c in enumerate(ASK_BANDS, start=1):
        out[f"ask_vol{lvl}"] = (pivot[c] - prev).clip(lower=0)
        prev = pivot[c]

    return out.resample("1h").last().ffill()


def already_covered_dates() -> set:
    if not OUT_PATH.exists():
        return set()
    existing = pd.read_csv(OUT_PATH, usecols=[0], parse_dates=[0])
    return set(existing.iloc[:, 0].dt.strftime("%Y-%m-%d"))


def main():
    start = datetime.strptime(START_DATE, "%Y-%m-%d")
    end = datetime.now(timezone.utc).replace(tzinfo=None)
    total_days = (end - start).days + 1

    covered = already_covered_dates()
    print(f"📡 Real bookDepth backfill for {SYMBOL}: {total_days} days total, "
          f"{len(covered)} already done.")

    fetched = 0
    curr = start
    while curr <= end:
        date_str = curr.strftime("%Y-%m-%d")
        curr += timedelta(days=1)
        if date_str in covered:
            continue

        raw = fetch_day(date_str)
        if raw is None or len(raw) == 0:
            continue

        hourly = decumulate_to_hourly(raw)
        write_header = not OUT_PATH.exists()
        hourly.to_csv(OUT_PATH, mode="a", header=write_header)
        fetched += 1
        if fetched % 30 == 0:
            print(f"   ... {fetched} new days fetched (latest: {date_str})")
        time.sleep(0.05)  # polite to the free public archive

    if OUT_PATH.exists():
        final = pd.read_csv(OUT_PATH, index_col=0, parse_dates=True)
        final = final[~final.index.duplicated(keep="last")].sort_index()
        final.to_csv(OUT_PATH)
        print(f"✅ Done. {len(final):,} real hourly order-book rows saved to {OUT_PATH}")
        print(f"   Range: {final.index.min()} -> {final.index.max()}")
    else:
        print("❌ No data was fetched.")


if __name__ == "__main__":
    main()
