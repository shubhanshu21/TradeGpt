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
                ts = int(time.time())
                filename = data_dir / f"ob_{ts}.json"
                with open(filename, "w") as f:
                    json.dump(ob, f)
            
            # Wait for 15 minutes (900 seconds) to match training timeframe
            time.sleep(900)
        except Exception as e:
            print(f"Error collecting OB: {e}")
            time.sleep(10)

if __name__ == "__main__":
    collect()
