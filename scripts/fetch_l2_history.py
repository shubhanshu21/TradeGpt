import pandas as pd
import requests
import zipfile
import io
import os
import datetime
import gc
import sys
import traceback
from pathlib import Path

# --- CONFIG ---
SYMBOL = "BTCUSDT"
START_DATE = "2023-05-16"
END_DATE   = "2026-07-10"
BASE_URL_DAILY   = "https://data.binance.vision/data/futures/um/daily/bookTicker"
BASE_URL_MONTHLY_OB = "https://data.binance.vision/data/futures/um/monthly/bookTicker"
BASE_URL_MONTHLY_TRADES = "https://data.binance.vision/data/futures/um/monthly/aggTrades"
ROOT = Path(__file__).parent.parent
OUTPUT_FILE = ROOT / "data/L2_3Year_Master.csv"
LOG_FILE = ROOT / "logs/history_fetch.log"
TEMP_ZIP = Path("/dev/shm/binance_temp.zip")

# Setup logging
LOG_F = open(LOG_FILE, "a", buffering=1)
sys.stdout = LOG_F
sys.stderr = LOG_F

def log(msg):
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)

def process_zip_stream(url, label, mode="OB"):
    try:
        log(f"📡 Fetching {label} ({mode}) via /dev/shm...")
        with requests.get(url, stream=True, timeout=120) as r:
            if r.status_code == 404: return None
            r.raise_for_status()
            with open(TEMP_ZIP, 'wb') as f:
                for chunk in r.iter_content(chunk_size=1024*1024):
                    f.write(chunk)
        
        with zipfile.ZipFile(TEMP_ZIP) as z:
            csv_names = [n for n in z.namelist() if n.endswith('.csv')]
            if not csv_names: return None
            csv_name = csv_names[0]
            
            all_resampled = []
            # Read first few lines to detect header
            with z.open(csv_name) as f:
                head = f.read(1000).decode()
                has_header = not head.strip()[0].isdigit()

            for chunk in pd.read_csv(z.open(csv_name), header=0 if has_header else None, low_memory=False, chunksize=1000000):
                if mode == "OB":
                    # Expected: update_id, bid1, bid_vol1, ask1, ask_vol1, trans_time, event_time
                    # We need indices 1, 2, 3, 4 and the last one (event_time)
                    ts_idx = -1 # Usually last
                    chunk = chunk.iloc[:, [1, 2, 3, 4, ts_idx]]
                    chunk.columns = ['bid1', 'bid_vol1', 'ask1', 'ask_vol1', 'ts']
                else:
                    # aggTrades: agg_id, price, qty, first_id, last_id, ts, is_buyer_maker
                    chunk = chunk.iloc[:, [1, 2, 5, 6]]
                    chunk.columns = ['price', 'qty', 'ts', 'is_sell']
                    chunk['price'] = pd.to_numeric(chunk['price'], errors='coerce')
                    chunk['qty'] = pd.to_numeric(chunk['qty'], errors='coerce')
                    chunk['is_sell'] = chunk['is_sell'].astype(str).str.lower() == 'true'
                    
                    chunk['bid1'] = chunk['price']
                    chunk['ask1'] = chunk['price']
                    chunk['bid_vol1'] = chunk['qty'].where(chunk['is_sell'], 0)
                    chunk['ask_vol1'] = chunk['qty'].where(~chunk['is_sell'], 0)
                    chunk = chunk[['bid1', 'bid_vol1', 'ask1', 'ask_vol1', 'ts']]

                chunk['ts'] = pd.to_numeric(chunk['ts'], errors='coerce')
                chunk = chunk.dropna(subset=['ts'])
                chunk['dt'] = pd.to_datetime(chunk['ts'], unit='ms')
                
                # Immediate resample
                res = chunk.set_index('dt').resample('15min').agg({
                    'bid1': 'last', 'ask1': 'last',
                    'bid_vol1': 'sum', 'ask_vol1': 'sum'
                }).dropna()
                all_resampled.append(res)
                del chunk
                gc.collect()
            
            if not all_resampled: return None
            combined = pd.concat(all_resampled).sort_index()
            final = combined.resample('15min').agg({
                'bid1': 'last', 'ask1': 'last',
                'bid_vol1': 'sum', 'ask_vol1': 'sum'
            }).dropna()
            return final
            
    except Exception as e:
        log(f"   ❌ Failed {label} ({mode}): {e}")
        log(traceback.format_exc())
        return None
    finally:
        if TEMP_ZIP.exists(): TEMP_ZIP.unlink()

def fetch_history():
    log("="*60)
    log("🚀 HYBRID MISSION START (V2 - Robust Headers)")
    log("="*60)

    start_dt = datetime.datetime.strptime(START_DATE, "%Y-%m-%d")
    end_dt   = datetime.datetime.strptime(END_DATE, "%Y-%m-%d")

    existing_months = set()
    if OUTPUT_FILE.exists():
        try:
            df_check = pd.read_csv(OUTPUT_FILE, usecols=[0], parse_dates=[0])
            counts = df_check.iloc[:, 0].dt.to_period('M').value_counts()
            existing_months = set(counts[counts >= 2000].index.astype(str))
            log(f"   Found {len(existing_months)} months in master.")
        except: pass

    curr = start_dt
    while curr <= end_dt:
        m_str = curr.strftime("%Y-%m")
        if m_str in existing_months:
            curr = (curr.replace(day=1) + datetime.timedelta(days=32)).replace(day=1)
            continue

        res = process_zip_stream(f"{BASE_URL_MONTHLY_OB}/{SYMBOL}/{SYMBOL}-bookTicker-{m_str}.zip", m_str, "OB")
        if res is None:
            log(f"   ⚠️ Falling back to Trades for {m_str}...")
            res = process_zip_stream(f"{BASE_URL_MONTHLY_TRADES}/{SYMBOL}/{SYMBOL}-aggTrades-{m_str}.zip", m_str, "Trades")
        
        if res is not None:
            write_header = not OUTPUT_FILE.exists()
            res.to_csv(OUTPUT_FILE, mode='a', header=write_header)
            log(f"   ✅ {m_str} saved.")
        else:
            log(f"   ❌ {m_str} totally unavailable.")
            
        curr = (curr.replace(day=1) + datetime.timedelta(days=32)).replace(day=1)
        gc.collect()

    if OUTPUT_FILE.exists():
        log("🧹 Final Polish...")
        df = pd.read_csv(OUTPUT_FILE)
        df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0])
        df = df.drop_duplicates(subset=[df.columns[0]]).sort_values(df.columns[0])
        df.to_csv(OUTPUT_FILE, index=False)
        log(f"✨ READY: {len(df)} snapshots.")

if __name__ == "__main__":
    fetch_history()
