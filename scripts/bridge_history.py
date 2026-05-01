import pandas as pd
import numpy as np
import requests
import time
from pathlib import Path
import os
import sys

# Add src to path
ROOT = Path(__file__).parent.parent
sys.path.append(str(ROOT / "src"))

from exchange.delta_client import DeltaClient

# Manual Log Setup
LOG_F = open(ROOT / "logs/bridge.log", "w", buffering=1)
sys.stdout = LOG_F
sys.stderr = LOG_F

def log(msg):
    print(msg, flush=True)

def fetch_binance_history(symbol="BTCUSDT", limit=400000, end_ts=None):
    """Fetch history from Binance Public API."""
    url = "https://api.binance.com/api/v3/klines"
    all_candles = []
    
    if end_ts is None:
        end_ts = int(time.time() * 1000)
    else:
        end_ts = int(end_ts * 1000)

    log(f"🚀 Bridging from Binance: {symbol}...")
    
    while len(all_candles) < limit:
        params = {
            "symbol": symbol,
            "interval": "15m",
            "limit": 1000,
            "endTime": end_ts
        }
        resp = requests.get(url, params=params)
        if resp.status_code != 200:
            log(f"   ⚠️ Binance error: {resp.status_code}")
            break
        
        data = resp.json()
        if not data:
            break
            
        # Binance returns: [OpenTime, Open, High, Low, Close, Vol, CloseTime, QuoteVol, Count, TakerBuyBase, TakerBuyQuote, Ignore]
        batch = []
        for c in data:
            batch.append({
                "timestamp": pd.to_datetime(c[0], unit='ms', utc=True),
                "open": float(c[1]),
                "high": float(c[2]),
                "low": float(c[3]),
                "close": float(c[4]),
                "volume": float(c[5]),
                "quote_volume": float(c[7]),
                "taker_buy_volume": float(c[9]),
                "taker_buy_quote_volume": float(c[10])
            })
            
        all_candles = batch + all_candles
        end_ts = data[0][0] - 1
        log(f"   Fetched {len(all_candles):,} Binance candles... (Back to {all_candles[0]['timestamp']})")
        
        if len(data) < 1000:
            break
            
    return pd.DataFrame(all_candles)

def bridge():
    DATA_DIR = ROOT / "data"
    DATA_DIR.mkdir(exist_ok=True)
    CACHE_P = DATA_DIR / "BTCUSD_15m_history_400000.parquet"
    
    # 1. Load Delta History (the most recent "Gold" data)
    delta_df = pd.DataFrame()
    if CACHE_P.exists():
        log("📖 Loading existing cache...")
        delta_df = pd.read_parquet(CACHE_P)
        oldest_delta = delta_df.timestamp.min().timestamp()
    else:
        log("📡 No cache found. Fetching Delta history first...")
        # We'll just run a small fetch to get the start point
        from exchange.fetch_data import fetch_live_kat_data
        delta_df = fetch_live_kat_data(symbol="BTCUSD", n_candles=80000, timeframe="15m")
        oldest_delta = delta_df.index.min().timestamp() if isinstance(delta_df.index, pd.DatetimeIndex) else delta_df.timestamp.min().timestamp()

    # 2. Fetch Binance to fill the gap (Targeting 400k candles ~ 11 years)
    needed = 400000 - len(delta_df)
    if needed > 0:
        binance_df = fetch_binance_history(limit=needed, end_ts=oldest_delta)
        
        # 3. Simulate Order Book for Binance (to match Delta feature set)
        log("🧠 Simulating Order Book for Binance history...")
        rng = np.random.default_rng(42)
        spread_base = 0.5
        for lvl in range(1, 6):
            binance_df[f"bid{lvl}"] = (binance_df["close"] - (spread_base * lvl)).round(2)
            binance_df[f"ask{lvl}"] = (binance_df["close"] + (spread_base * lvl)).round(2)
            binance_df[f"bid_vol{lvl}"] = rng.exponential(2.0, len(binance_df)).round(4)
            binance_df[f"ask_vol{lvl}"] = rng.exponential(2.0, len(binance_df)).round(4)
            
        # 4. Merge
        if 'timestamp' in delta_df.columns:
            delta_df = delta_df.set_index('timestamp')
        if 'timestamp' in binance_df.columns:
            binance_df = binance_df.set_index('timestamp')
            
        combined = pd.concat([binance_df, delta_df]).sort_index()
        combined = combined[~combined.index.duplicated(keep='last')]
        combined = combined.tail(400000)
        
        log(f"✅ Bridge Complete! Total Candles: {len(combined):,}")
        log(f"📅 Range: {combined.index.min()} to {combined.index.max()}")
        
        combined.to_parquet(CACHE_P)
        log(f"💾 Saved to {CACHE_P}")
    else:
        log("✅ Delta history is already sufficient.")

if __name__ == "__main__":
    bridge()
