import time
import json
import os
from pathlib import Path
import sys

# Add src to path
ROOT = Path(__file__).parent.parent
sys.path.append(str(ROOT / "src"))

from exchange.delta_client import DeltaClient

def collect():
    client = DeltaClient(testnet=False)
    data_dir = ROOT / "data/orderbook_history"
    data_dir.mkdir(parents=True, exist_ok=True)
    
    symbol = "BTCUSD"
    print(f"🚀 Starting L2 Collector for {symbol}...")
    
    while True:
        try:
            ob = client.get_orderbook(symbol)
            if ob:
                # Align TS to the nearest 15-minute mark for perfect matching with candles
                ts = int(time.time())
                ts_aligned = (ts // 900) * 900
                
                filename = data_dir / f"ob_{ts_aligned}.json"
                with open(filename, "w") as f:
                    json.dump(ob, f)
                
                # --- AUTO-PURGE: Keep only last 48 hours ---
                cutoff = ts - (48 * 3600)
                for old_file in data_dir.glob("ob_*.json"):
                    try:
                        file_ts = int(old_file.stem.split("_")[1])
                        if file_ts < cutoff:
                            old_file.unlink()
                    except: pass
            
            # Sleep until the next 15-minute mark + 10s buffer
            ts_now = time.time()
            sleep_time = 900 - (ts_now % 900) + 10
            time.sleep(sleep_time)
        except Exception as e:
            print(f"Error collecting OB: {e}")
            time.sleep(10)

if __name__ == "__main__":
    collect()
