import pandas as pd
import numpy as np
from delta_client import DeltaClient
import time

def fetch_live_kat_data(symbol: str = "BTCUSDT", n_candles: int = 1000) -> pd.DataFrame:
    """
    Fetches real candles from Delta and augments with OB data.
    For the latest candle, it tries to fetch the actual live OB.
    For historical candles, it simulates OB to maintain feature consistency.
    """
    # Use Live Production Server for market data (even during paper trading)
    client = DeltaClient(testnet=False)
    
    print(f"   Fetching {n_candles} candles for {symbol}...")
    df = client.get_candles(symbol, resolution="1m", limit=n_candles)
    
    if df.empty:
        raise ValueError(f"No data returned for {symbol}")

    # ── Augment with Order Book ──────────────────────────────────────────────
    # We need 10 cols for bid1-5/ask1-5 and 10 cols for volumes
    
    # Simulate OB for all candles first (historical consistency)
    spread_base = 0.5
    rng = np.random.default_rng(42)
    
    for lvl in range(1, 6):
        # Simulation logic similar to generate_data.py
        df[f"bid{lvl}"] = (df["close"] - (spread_base * lvl)).round(2)
        df[f"ask{lvl}"] = (df["close"] + (spread_base * lvl)).round(2)
        df[f"bid_vol{lvl}"] = rng.exponential(2.0, len(df)).round(4)
        df[f"ask_vol{lvl}"] = rng.exponential(2.0, len(df)).round(4)

    # For the VERY LATEST candle, try to inject the REAL live OB if possible
    try:
        live_ob = client.get_orderbook(symbol)
        if live_ob:
            buy  = live_ob.get("buy", [])
            sell = live_ob.get("sell", [])
            for i in range(min(5, len(buy))):
                df.loc[df.index[-1], f"bid{i+1}"] = float(buy[i]["price"])
                df.loc[df.index[-1], f"bid_vol{i+1}"] = float(buy[i]["size"])
            for i in range(min(5, len(sell))):
                df.loc[df.index[-1], f"ask{i+1}"] = float(sell[i]["price"])
                df.loc[df.index[-1], f"ask_vol{i+1}"] = float(sell[i]["size"])
            print("   ✓ Injected live L2 order book for the latest candle.")
    except Exception as e:
        print(f"   ! Could not fetch live OB ({e}). Using simulated OB for latest candle.")

    return df
