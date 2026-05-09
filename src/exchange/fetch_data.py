import pandas as pd
import numpy as np
import time
import json
from pathlib import Path
from exchange.delta_client import DeltaClient

def fetch_live_kat_data(symbol: str = "BTCUSDT", n_candles: int = 1000, timeframe: str = "1m") -> pd.DataFrame:
    """
    Fetches real candles from Delta and augments with OB data.
    For the latest candle, it tries to fetch the actual live OB.
    For historical candles, it simulates OB to maintain feature consistency.
    """
    # Use Live Production Server for market data (even during paper trading)
    client = DeltaClient(testnet=False)
    
    print(f"   Fetching {n_candles} candles for {symbol} ({timeframe})...")
    df = client.get_candles(symbol, resolution=timeframe, limit=n_candles)
    
    if df.empty:
        raise ValueError(f"No data returned for {symbol}")

    # ── Augment with Order Book ──────────────────────────────────────────────
    # We need 10 cols for bid1-5/ask1-5 and 10 cols for volumes
    
    # 1. Search for REAL historical files first
    ob_dir = Path(__file__).parent.parent.parent / "data/orderbook_history"
    
    for i, row in df.iterrows():
        ts_sec = int(i.timestamp())
        # Check for a file within 60s of this candle's start
        found_file = None
        if ob_dir.exists():
            # Look for ob_{ts}.json where ts is close to ts_sec
            # For speed, we just check the exact 15m mark or nearby
            potential_file = ob_dir / f"ob_{ts_sec}.json"
            if potential_file.exists():
                found_file = potential_file

        if found_file:
            try:
                with open(found_file, "r") as f:
                    ob_data = json.load(f)
                buy  = ob_data.get("buy", [])
                sell = ob_data.get("sell", [])
                for lvl in range(1, 6):
                    if lvl <= len(buy):
                        df.loc[i, f"bid{lvl}"] = float(buy[lvl-1]["price"])
                        df.loc[i, f"bid_vol{lvl}"] = float(buy[lvl-1]["size"])
                    if lvl <= len(sell):
                        df.loc[i, f"ask{lvl}"] = float(sell[lvl-1]["price"])
                        df.loc[i, f"ask_vol{lvl}"] = float(sell[lvl-1]["size"])
                continue # Move to next candle
            except: pass

        # 2. Fallback to Neutral Zeros (NOT simulation)
        for lvl in range(1, 6):
            df.loc[i, f"bid{lvl}"] = 0.0
            df.loc[i, f"ask{lvl}"] = 0.0
            df.loc[i, f"bid_vol{lvl}"] = 0.0
            df.loc[i, f"ask_vol{lvl}"] = 0.0

    # For the VERY LATEST candle, try to inject the LIVE OB from API
    try:
        live_ob = client.get_orderbook(symbol)
        if live_ob:
            buy  = live_ob.get("buy", [])
            sell = live_ob.get("sell", [])
            for i_lvl in range(min(5, len(buy))):
                df.loc[df.index[-1], f"bid{i_lvl+1}"] = float(buy[i_lvl]["price"])
                df.loc[df.index[-1], f"bid_vol{i_lvl+1}"] = float(buy[i_lvl]["size"])
            for i_lvl in range(min(5, len(sell))):
                df.loc[df.index[-1], f"ask{i_lvl+1}"] = float(sell[i_lvl]["price"])
                df.loc[df.index[-1], f"ask_vol{i_lvl+1}"] = float(sell[i_lvl]["size"])
            print("   ✓ Injected live L2 order book for the latest candle.")
    except Exception as e:
        print(f"   ! Could not fetch live OB ({e}). Using simulated OB for latest candle.")

    return df
