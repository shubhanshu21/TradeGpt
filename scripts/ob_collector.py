import time
import json
import csv
import os
from pathlib import Path
import sys

# Add src to path
ROOT = Path(__file__).parent.parent
sys.path.append(str(ROOT / "src"))

from exchange.delta_client import DeltaClient

# Permanent, compact top-5-level log — never purged. This is what actually
# feeds the model's obi_l5 feature; the raw full-depth JSON snapshots below
# are kept only briefly for debugging (they're 200KB+ each with thousands of
# levels we don't use, so keeping them forever would be pure disk waste).
L5_LOG_HEADER = ["timestamp"]
for lvl in range(1, 6):
    L5_LOG_HEADER += [f"bid{lvl}", f"bid_vol{lvl}", f"ask{lvl}", f"ask_vol{lvl}"]


def extract_top5(ob):
    """Distill a raw (thousands-of-levels) order book snapshot into the
    top-5 bid/ask price+size pairs, matching the historical parquet's
    bid1-5/ask1-5/bid_vol1-5/ask_vol1-5 column convention exactly."""
    buys = sorted(ob.get("buy", []), key=lambda r: -float(r["price"]))[:5]
    sells = sorted(ob.get("sell", []), key=lambda r: float(r["price"]))[:5]
    row = {}
    for lvl in range(1, 6):
        b = buys[lvl - 1] if lvl <= len(buys) else None
        s = sells[lvl - 1] if lvl <= len(sells) else None
        row[f"bid{lvl}"] = float(b["price"]) if b else None
        row[f"bid_vol{lvl}"] = float(b["size"]) if b else None
        row[f"ask{lvl}"] = float(s["price"]) if s else None
        row[f"ask_vol{lvl}"] = float(s["size"]) if s else None
    return row


def collect():
    client = DeltaClient(testnet=False)
    data_dir = ROOT / "data/orderbook_history"
    data_dir.mkdir(parents=True, exist_ok=True)
    l5_log_path = ROOT / "data/orderbook_l5_history.csv"

    symbol = "BTCUSD"
    from datetime import datetime
    print(f"[{datetime.now().isoformat()}] 🚀 Starting L2 Collector for {symbol}...")

    retry_delay = 10
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

                # --- PERMANENT LOG: append distilled top-5 levels, never purged ---
                try:
                    row = extract_top5(ob)
                    write_header = not l5_log_path.exists()
                    with open(l5_log_path, "a", newline="") as lf:
                        writer = csv.DictWriter(lf, fieldnames=L5_LOG_HEADER)
                        if write_header:
                            writer.writeheader()
                        writer.writerow({"timestamp": ts_aligned, **row})
                except Exception as log_err:
                    print(f"[{datetime.now().isoformat()}] ⚠️ L5 log append failed: {log_err}")

                # --- AUTO-PURGE (raw full-depth JSON only): keep last 48 hours ---
                cutoff = ts - (48 * 3600)
                for old_file in data_dir.glob("ob_*.json"):
                    try:
                        file_ts = int(old_file.stem.split("_")[1])
                        if file_ts < cutoff:
                            old_file.unlink()
                    except: pass
            
            # Reset retry delay on successful run
            retry_delay = 10
            
            # Sleep until the next 15-minute mark + 10s buffer
            ts_now = time.time()
            sleep_time = 900 - (ts_now % 900) + 10
            time.sleep(sleep_time)
        except Exception as e:
            from datetime import datetime
            print(f"[{datetime.now().isoformat()}] Error collecting OB: {e}")
            print(f"[{datetime.now().isoformat()}] Retrying in {retry_delay} seconds...")
            time.sleep(retry_delay)
            retry_delay = min(retry_delay * 2, 900)

if __name__ == "__main__":
    collect()
