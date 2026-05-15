import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
import keras
from pathlib import Path
from datetime import datetime, timedelta

# Setup paths
ROOT = Path(__file__).parent.parent
sys.path.append(str(ROOT / "src"))

from core.hydra import build_kraken, CertaintyMetric, SovereignAccuracy, SovereignLoss
from data.preprocess import compute_indicators, build_feature_cols
from exchange.fetch_data import fetch_live_kat_data

def run_certainty_distribution():
    print("\n" + "="*60)
    print("🚀 IRON ORACLE V11: MULTI-CERTAINTY TIMELINE AUDIT")
    print("="*60)
    
    model_path = ROOT / "models" / "hydra_best.keras"
    n_features = len(build_feature_cols())
    
    try:
        model = build_kraken(context_window=120, n_features=n_features)
        model.load_weights(model_path)
        print("✅ Model weights loaded.")
    except Exception as e:
        print(f"❌ Error: {e}")
        return

    # Fetch live data
    symbol = "BTCUSD"
    n_candles = 300 
    df_raw = fetch_live_kat_data(symbol, n_candles, "15m")
    df = compute_indicators(df_raw)
    features = build_feature_cols()
    data = df[features].values.astype('float32')
    timestamps = df.index

    # Today's range
    today_start = pd.Timestamp(datetime.now().date(), tz='UTC')
    today_indices = np.where(timestamps >= today_start)[0]
    
    if len(today_indices) == 0:
        print("⚠️ No data for today yet. Checking last 50 candles.")
        today_indices = range(len(data)-50, len(data))

    print(f"\n📊 CERTAINTY DISTRIBUTION (Today: {today_start.strftime('%Y-%m-%d')})")
    print(f"{'Time (UTC)':<12} | {'Price':<10} | {'Certainty':<10} | {'Bias':<18} | {'Status'}")
    print("-" * 75)
    
    ctx = 120
    for i in today_indices:
        if i < ctx: continue
        
        # Prepare window
        win_x = data[i-ctx:i]
        l_mean = win_x.mean(axis=0)
        l_std  = np.maximum(win_x.std(axis=0), 1e-3)
        xs = np.clip((win_x - l_mean) / l_std, -5.0, 5.0)
        inp = np.expand_dims(xs, axis=0)
        
        out = model(inp, training=False)
        # Scaled certainty (Mean * 120)
        cert_score = np.mean(out[1].numpy()[0]) * 120.0
        reas_idx = int(np.argmax(out[2].numpy()[0]))
        
        bias_map = {0: "LONG 🏹", 1: "SHORT 📉", 2: "FEE_TRAP ⚠️", 3: "NOISE 😴"}
        bias_str = bias_map.get(reas_idx, "UNKNOWN")
        
        status = "🟢 TRADE" if (reas_idx in [0, 1] and cert_score >= 115.0) else "⚪ SCAN"
        # Color the noise and fee traps
        if reas_idx == 2: status = "🟡 TRAP"
        if reas_idx == 3: status = "🔵 NOISE"

        ts_str = timestamps[i-1].strftime("%H:%M")
        price_str = f"${df['close'].iloc[i-1]:,.0f}"
        
        print(f"{ts_str:<12} | {price_str:<10} | {cert_score:<10.2f} | {bias_str:<18} | {status}")

    print("="*60 + "\n")

if __name__ == "__main__":
    run_certainty_distribution()
