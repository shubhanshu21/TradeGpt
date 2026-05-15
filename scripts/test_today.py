import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
import keras
from pathlib import Path
from datetime import datetime

# Setup paths
ROOT = Path(__file__).parent.parent
sys.path.append(str(ROOT / "src"))

from core.hydra import build_kraken, init_kraken_hardware, CertaintyMetric, SovereignAccuracy, SovereignLoss
from data.preprocess import compute_indicators, build_feature_cols
from exchange.fetch_data import fetch_live_kat_data

def run_today_audit():
    print("\n" + "="*60)
    print("🚀 IRON ORACLE V11: TODAY'S DATA AUDIT")
    print("="*60)
    
    # 1. Initialize and Build Model
    model_path = ROOT / "models" / "hydra_best.keras"
    if not model_path.exists():
        print(f"❌ Error: Model weights not found at {model_path}")
        return
    
    n_features = len(build_feature_cols())
    print(f"🏗️ Building Hydra V11.0 architecture ({n_features} features)...")
    try:
        # Build the model manually first to avoid Lambda deserialization issues
        model = build_kraken(context_window=120, n_features=n_features)
        print("📂 Loading weights into architecture...")
        model.load_weights(model_path)
        print("✅ Model ready for inference.")
    except Exception as e:
        print(f"❌ Error building/loading model: {e}")
        # Fallback to direct load if build fails
        print("🔄 Attempting direct load with safe_mode=False...")
        try:
            keras.config.enable_unsafe_deserialization()
            model = keras.models.load_model(model_path, custom_objects={
                "SovereignLoss": SovereignLoss,
                "CertaintyMetric": CertaintyMetric,
                "SovereignAccuracy": SovereignAccuracy
            })
            print("✅ Direct load successful.")
        except Exception as e2:
            print(f"❌ Critical Error: {e2}")
            return

    # 2. Fetch live data for today
    symbol = "BTCUSD"
    timeframe = "15m"
    n_candles = 300 
    print(f"📡 Fetching live {symbol} market context...")
    
    try:
        df_raw = fetch_live_kat_data(symbol, n_candles, timeframe)
        df = compute_indicators(df_raw)
        features = build_feature_cols()
        data = df[features].values.astype('float32')
        raw_prices = df['close'].values
        timestamps = df.index
    except Exception as e:
        print(f"❌ Error fetching data: {e}")
        return

    print(f"📊 Data synced. Current local time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📊 Last candle in dataset: {timestamps[-1]}")
    
    # 3. Run inference on the LATEST window
    ctx = 120
    if len(data) < ctx:
        print(f"❌ Error: Not enough data ({len(data)} < {ctx})")
        return
    
    latest_x = data[-ctx:]
    local_mean = latest_x.mean(axis=0)
    local_std  = latest_x.std(axis=0)
    local_std = np.maximum(local_std, 1e-3) # Stability floor from preprocess.py
    
    x_scaled = np.clip((latest_x - local_mean) / local_std, -5.0, 5.0)
    X = np.expand_dims(x_scaled, axis=0)
    
    outputs = model(X, training=False)
    forecast = outputs[0].numpy()[0]
    certainty = np.mean(outputs[1].numpy()[0])
    reasoning = int(np.argmax(outputs[2].numpy()[0]))
    
    bias_map = {0: "SOVEREIGN_LONG 🏹", 1: "SOVEREIGN_SHORT 📉", 2: "FEE_TRAP ⚠️", 3: "NOISE 😴"}
    current_bias = bias_map.get(reasoning, "UNKNOWN")
    
    print("\n" + "─"*30)
    print(f"🧠 LIVE MARKET VERDICT")
    print(f"Current Price:  ${raw_prices[-1]:,.2f}")
    print(f"Model Bias:     {current_bias}")
    print(f"Certainty:      {certainty:.3f}")
    print("─"*30)

    # 4. Detailed Performance Check on Today's candles (May 15)
    # Today's data starts at 00:00 UTC on May 15
    today_start = pd.Timestamp(datetime.now().date(), tz='UTC')
    today_mask = timestamps >= today_start
    today_indices = np.where(today_mask)[0]
    
    if len(today_indices) == 0:
        print("\n⚠️ No candles recorded yet for today (May 15). Showing last 20 candles instead.")
        today_indices = range(len(data)-20, len(data))

    print(f"\n📜 PERFORMANCE AUDIT (Today: {today_start.strftime('%Y-%m-%d')})")
    print(f"{'Time (UTC)':<18} | {'Side':<6} | {'Entry':<10} | {'Exit':<10} | {'Result':<8}")
    print("-" * 65)
    
    f = 15 # 15-step (3h 45m) forward horizon
    total_net = 0
    trades = 0
    wins = 0
    fee_rate = 0.0012
    
    # We test each candle for today that has enough forward context
    for i in today_indices:
        if i + f >= len(data): continue
        if i < ctx: continue
        
        # Prepare window
        win_x = data[i-ctx:i]
        l_mean = win_x.mean(axis=0)
        l_std  = np.maximum(win_x.std(axis=0), 1e-3)
        xs = np.clip((win_x - l_mean) / l_std, -5.0, 5.0)
        inp = np.expand_dims(xs, axis=0)
        
        out = model(inp, training=False)
        reas = int(np.argmax(out[2].numpy()[0]))
        cert = np.mean(out[1].numpy()[0])
        
        # Lower certainty threshold for audit to see more trades, 
        # but Sovereign moves only
        if reas in [0, 1] and cert > 110.0:
            side = "LONG" if reas == 0 else "SHORT"
            entry = raw_prices[i-1]
            exit_p = raw_prices[i+f-1]
            
            raw_ret = (exit_p - entry) / entry if side == "LONG" else (entry - exit_p) / entry
            net_ret = raw_ret - fee_rate
            
            total_net += net_ret
            trades += 1
            if net_ret > 0: wins += 1
            
            ts_str = timestamps[i-1].strftime("%H:%M")
            print(f"{ts_str:<18} | {side:<6} | {entry:<10.1f} | {exit_p:<10.1f} | {net_ret*100:>+7.2f}%")

    if trades > 0:
        print("-" * 65)
        print(f"SUMMARY: {trades} Trades | {wins} Wins | Win Rate: {(wins/trades*100):.1f}%")
        print(f"TOTAL NET ROI FOR TODAY: {total_net*100:>+7.2f}%")
    else:
        print("\n😴 Model was in deep scanning mode today. No high-conviction trades fired.")
    print("="*60 + "\n")

if __name__ == "__main__":
    run_today_audit()
