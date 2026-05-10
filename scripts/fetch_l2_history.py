import pandas as pd
import requests
import zipfile
import io
import os
import datetime
from pathlib import Path

# --- CONFIG ---
SYMBOL = "BTCUSDT"
# First available day on Binance Vision Daily folder
START_DATE = "2023-05-16" 
END_DATE   = "2026-05-08" 
BASE_URL   = "https://data.binance.vision/data/futures/um/daily/bookTicker"
ROOT = Path(__file__).parent.parent
OUTPUT_FILE = ROOT / "data/L2_3Year_Master.csv"

# Optimizing memory usage and avoiding DtypeWarnings
COLUMN_NAMES = ['event_time', 'trans_time', 'symbol', 'bid1', 'bid_vol1', 'ask1', 'ask_vol1']
DTYPES = {
    'event_time': 'int64',
    'trans_time': 'int64',
    'symbol': 'category',
    'bid1': 'float32',
    'bid_vol1': 'float32',
    'ask1': 'float32',
    'ask_vol1': 'float32'
}

import concurrent.futures

def process_day(day_info):
    day, date_str = day_info
    url = f"{BASE_URL}/{SYMBOL}/{SYMBOL}-bookTicker-{date_str}.zip"
    try:
        r = requests.get(url, timeout=30)
        if r.status_code == 200:
            with zipfile.ZipFile(io.BytesIO(r.content)) as z:
                csv_name = z.namelist()[0]
                with z.open(csv_name) as f:
                    df = pd.read_csv(f, header=None, low_memory=False)
                    if not str(df.iloc[0, 0]).isdigit():
                        df = df.iloc[1:].reset_index(drop=True)
                    
                    cols = ['update_id', 'bid1', 'bid_vol1', 'ask1', 'ask_vol1', 'trans_time', 'event_time']
                    df.columns = cols[:len(df.columns)]
                    df['event_time'] = pd.to_numeric(df['event_time'], errors='coerce')
                    df = df.dropna(subset=['event_time'])
                    df['dt'] = pd.to_datetime(df['event_time'], unit='ms')
                    
                    snap = df.set_index('dt').resample('15min').last().dropna()
                    new_data = snap[['bid1', 'bid_vol1', 'ask1', 'ask_vol1']]
                    return date_str, new_data
        return date_str, None
    except Exception as e:
        return date_str, f"Error: {e}"

def fetch_history():
    start_dt = datetime.datetime.strptime(START_DATE, "%Y-%m-%d")
    end_dt   = datetime.datetime.strptime(END_DATE, "%Y-%m-%d")

    # Resume logic: Check what's already in the CSV
    existing_dates = set()
    if OUTPUT_FILE.exists():
        try:
            tmp = pd.read_csv(OUTPUT_FILE, usecols=[0], parse_dates=[0])
            # Only count a day as finished if it has enough snapshots
            counts = tmp.iloc[:, 0].dt.date.value_counts()
            existing_dates = set(counts[counts >= 90].index)
            del tmp
            log(f"   Found {len(existing_dates)} full days in master.")
        except Exception as e: 
            log(f"⚠️ Resume check failed: {e}")

    print(f"🚀 Starting Parallel 3-Year L2 Build...")
    days_to_fetch = []
    delta = end_dt - start_dt
    for i in range(delta.days + 1):
        day = start_dt + datetime.timedelta(days=i)
        if day.date() not in existing_dates:
            days_to_fetch.append((day, day.strftime("%Y-%m-%d")))

    if not days_to_fetch:
        print("✅ Data is already up to date.")
        return

    log(f"   Target: {len(days_to_fetch)} days remaining.")
    
    # Process in batches of 2 to stay safe with 24GB RAM (each worker uses ~4GB)
    with concurrent.futures.ProcessPoolExecutor(max_workers=2) as executor:
        futures = {executor.submit(process_day, d): d for d in days_to_fetch}
        for future in concurrent.futures.as_completed(futures):
            date_str, result = future.result()
            if isinstance(result, pd.DataFrame):
                write_header = not OUTPUT_FILE.exists()
                result.to_csv(OUTPUT_FILE, mode='a', header=write_header)
                print(f"   ✅ {date_str} added. ({len(result)} snapshots)")
            elif result is None:
                print(f"   ⚠️ {date_str} not available.")
            else:
                print(f"   ❌ {date_str} failed: {result}")

    print(f"\n✨ DONE! 3-year history saved to {OUTPUT_FILE}")

def log(msg):
    print(msg, flush=True)

if __name__ == "__main__":
    fetch_history()
