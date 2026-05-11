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
    # We need 20 cols total: 10 for prices (bid1-5/ask1-5) and 10 for volumes
    
    # 1. Check for Master CSV first (Highly efficient for years of data)
    master_l2_path = Path(__file__).parent.parent.parent / "data/L2_3Year_Master.csv"
    master_l2 = None
    if master_l2_path.exists():
        try:
            master_l2 = pd.read_csv(master_l2_path, index_col=0)
            master_l2.index = pd.to_datetime(master_l2.index, utc=True)
            # Deduplicate to prevent .loc[] from returning a DataFrame
            master_l2 = master_l2[~master_l2.index.duplicated(keep='last')]
            print(f"   📖 Loaded {len(master_l2):,} real snapshots from {master_l2_path.name}")
        except: pass

    ob_dir = Path(__file__).parent.parent.parent / "data/orderbook_history"
    
    for i, row in df.iterrows():
        ts_val = row['timestamp'] if 'timestamp' in df.columns else i
        if not isinstance(ts_val, pd.Timestamp):
            try:
                ts_val = pd.to_datetime(ts_val, unit='s', utc=True)
            except:
                ts_val = pd.to_datetime(ts_val, utc=True)
        
        ts_utc = ts_val.tz_localize('UTC') if ts_val.tzinfo is None else ts_val.tz_convert('UTC')
        
        # Priority 1: Master CSV (Binance History)
        if master_l2 is not None and ts_utc in master_l2.index:
            m_row = master_l2.loc[ts_utc]
            df.loc[i, "bid1"] = float(m_row["bid1"])
            df.loc[i, "bid_vol1"] = float(m_row["bid_vol1"])
            df.loc[i, "ask1"] = float(m_row["ask1"])
            df.loc[i, "ask_vol1"] = float(m_row["ask_vol1"])
            # Levels 2-5 are simulated relative to Level 1
            for lvl in range(2, 6):
                df.loc[i, f"bid{lvl}"] = float(m_row["bid1"]) - (0.5 * (lvl-1))
                df.loc[i, f"ask{lvl}"] = float(m_row["ask1"]) + (0.5 * (lvl-1))
                df.loc[i, f"bid_vol{lvl}"] = 1.0 # Static proxy for deep levels
                df.loc[i, f"ask_vol{lvl}"] = 1.0
            continue

        # Priority 2: Individual JSON files (Delta Live Collection)
        ts_sec = int(ts_utc.timestamp())
        found_file = None
        if ob_dir.exists():
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
                continue 
            except: pass

        # Priority 3: Fallback to Neutral (Not zeros, but close-relative)
        curr_p = row['close']
        for lvl in range(1, 6):
            df.loc[i, f"bid{lvl}"] = curr_p - (0.5 * lvl)
            df.loc[i, f"ask{lvl}"] = curr_p + (0.5 * lvl)
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

    if 'timestamp' in df.columns:
        df = df.set_index('timestamp')
    return df
