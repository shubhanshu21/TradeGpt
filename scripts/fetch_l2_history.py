import pandas as pd
import requests
import zipfile
import io
import os
import datetime
import gc
import sys
from pathlib import Path

# --- CONFIG ---
SYMBOL = "BTCUSDT"
START_DATE = "2023-05-16" 
END_DATE   = "2026-05-08" 
BASE_URL_DAILY   = "https://data.binance.vision/data/futures/um/daily/bookTicker"
BASE_URL_MONTHLY = "https://data.binance.vision/data/futures/um/monthly/bookTicker"
ROOT = Path(__file__).parent.parent
OUTPUT_FILE = ROOT / "data/L2_3Year_Master.csv"
LOG_FILE = ROOT / "logs/history_fetch.log"

# Setup logging to file
LOG_F = open(LOG_FILE, "a", buffering=1)
sys.stdout = LOG_F
sys.stderr = LOG_F

def log(msg):
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)

def process_zip_content(content, label="archive"):
    try:
        with zipfile.ZipFile(io.BytesIO(content)) as z:
            all_data = []
            for csv_name in z.namelist():
                if not csv_name.endswith('.csv'): continue
                # Use chunked reading for large CSVs
                chunks = []
                for chunk in pd.read_csv(z.open(csv_name), header=None, low_memory=False, chunksize=100000):
                    if not str(chunk.iloc[0, 0]).isdigit():
                        chunk = chunk.iloc[1:].reset_index(drop=True)
                    
                    cols = ['update_id', 'bid1', 'bid_vol1', 'ask1', 'ask_vol1', 'trans_time', 'event_time']
                    chunk.columns = cols[:len(chunk.columns)]
                    chunk['event_time'] = pd.to_numeric(chunk['event_time'], errors='coerce')
                    chunk = chunk.dropna(subset=['event_time'])
                    chunk['dt'] = pd.to_datetime(chunk['event_time'], unit='ms')
                    
                    # Resample chunk
                    snap = chunk.set_index('dt').resample('15min').last().dropna()
                    chunks.append(snap[['bid1', 'bid_vol1', 'ask1', 'ask_vol1']])
                    del chunk
                
                if chunks:
                    all_data.append(pd.concat(chunks).sort_index())
                    del chunks
            
            if not all_data: return None
            
            combined = pd.concat(all_data).sort_index()
            # Final resample for the whole zip content
            final = combined.resample('15min').last().dropna()
            del combined
            del all_data
            gc.collect()
            return final
    except Exception as e:
        log(f"   ❌ Error processing {label}: {e}")
        return None

def fetch_history():
    log("="*60)
    log("🚀 Starting LEAN Historical OB Build")
    log("="*60)

    start_dt = datetime.datetime.strptime(START_DATE, "%Y-%m-%d")
    end_dt   = datetime.datetime.strptime(END_DATE, "%Y-%m-%d")

    # 1. Check existing data
    existing_dates = set()
    if OUTPUT_FILE.exists():
        try:
            tmp = pd.read_csv(OUTPUT_FILE, usecols=[0], parse_dates=[0])
            counts = tmp.iloc[:, 0].dt.date.value_counts()
            existing_dates = set(counts[counts >= 90].index)
            del tmp
            log(f"   Found {len(existing_dates)} full days in master.")
        except Exception as e: 
            log(f"⚠️ Resume check failed: {e}")

    # 2. Identify missing months and days
    months_to_fetch = []
    days_to_fetch = []
    
    curr = start_dt
    while curr <= end_dt:
        month_start = curr.replace(day=1)
        next_month = (month_start + datetime.timedelta(days=32)).replace(day=1)
        month_end = next_month - datetime.timedelta(days=1)
        
        if month_end >= datetime.datetime.now():
            d = curr
            while d < next_month and d <= end_dt:
                if d.date() not in existing_dates:
                    days_to_fetch.append(d.strftime("%Y-%m-%d"))
                d += datetime.timedelta(days=1)
            curr = next_month
            continue

        days_in_month = (month_end - month_start).days + 1
        days_we_have = sum(1 for d in existing_dates if d.month == curr.month and d.year == curr.year)
        
        if days_we_have < days_in_month:
            months_to_fetch.append(curr.strftime("%Y-%m"))
        
        curr = next_month

    if not months_to_fetch and not days_to_fetch:
        log("✅ Data is already up to date.")
        return

    log(f"   Targeting {len(months_to_fetch)} months and {len(days_to_fetch)} days.")

    # 3. Fetch Months (SEQUENTIAL for RAM stability)
    for m_str in months_to_fetch:
        log(f"📡 Fetching Month {m_str}...")
        url = f"{BASE_URL_MONTHLY}/{SYMBOL}/{SYMBOL}-bookTicker-{m_str}.zip"
        try:
            r = requests.get(url, timeout=120)
            if r.status_code == 200:
                result = process_zip_content(r.content, label=m_str)
                if result is not None:
                    write_header = not OUTPUT_FILE.exists()
                    result.to_csv(OUTPUT_FILE, mode='a', header=write_header)
                    log(f"      ✅ Month {m_str} added. ({len(result)} snapshots)")
                del result
            else:
                log(f"      ⚠️ Month {m_str} 404 archive. Falling back to daily.")
                m_dt = datetime.datetime.strptime(m_str, "%Y-%m")
                next_m = (m_dt + datetime.timedelta(days=32)).replace(day=1)
                d = m_dt
                while d < next_m and d <= end_dt:
                    if d.date() not in existing_dates:
                        days_to_fetch.append(d.strftime("%Y-%m-%d"))
                    d += datetime.timedelta(days=1)
            gc.collect()
        except Exception as e:
            log(f"      ❌ Month {m_str} fetch failed: {e}")

    # 4. Fetch Days (SEQUENTIAL for RAM stability)
    days_to_fetch = sorted(list(set(days_to_fetch)))
    for d_str in days_to_fetch:
        log(f"📡 Fetching Day {d_str}...")
        url = f"{BASE_URL_DAILY}/{SYMBOL}/{SYMBOL}-bookTicker-{d_str}.zip"
        try:
            r = requests.get(url, timeout=60)
            if r.status_code == 200:
                result = process_zip_content(r.content, label=d_str)
                if result is not None:
                    write_header = not OUTPUT_FILE.exists()
                    result.to_csv(OUTPUT_FILE, mode='a', header=write_header)
                    log(f"      ✅ Day {d_str} added.")
                del result
            else:
                log(f"      ⚠️ Day {d_str} not available.")
            gc.collect()
        except Exception as e:
            log(f"      ❌ Day {d_str} fetch failed: {e}")

    # 5. Final Cleanup: Deduplicate and Sort
    if OUTPUT_FILE.exists():
        log("🧹 Final cleanup: Deduplicating and sorting master CSV...")
        df_master = pd.read_csv(OUTPUT_FILE)
        df_master.iloc[:, 0] = pd.to_datetime(df_master.iloc[:, 0])
        df_master = df_master.drop_duplicates(subset=[df_master.columns[0]])
        df_master = df_master.sort_values(df_master.columns[0])
        df_master.to_csv(OUTPUT_FILE, index=False)
        log(f"✨ MISSION COMPLETE! Master OB history ready: {OUTPUT_FILE}")
        log(f"   Range: {df_master.iloc[0,0]} to {df_master.iloc[-1,0]}")
        log(f"   Total Snapshots: {len(df_master):,}")

if __name__ == "__main__":
    fetch_history()
