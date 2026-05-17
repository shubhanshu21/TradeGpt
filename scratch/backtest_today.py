import os
import sys
import json
import pandas as pd
import numpy as np
import tensorflow as tf
from pathlib import Path
from datetime import datetime, timedelta

# Add root and src to path
ROOT = Path("/var/www/html/ML/kat")
sys.path.append(str(ROOT))
sys.path.append(str(ROOT / "src"))

from data.preprocess import compute_indicators, build_feature_cols
from exchange.fetch_data import fetch_live_kat_data
from core.hydra import build_kraken

MODEL_PATH = ROOT / "models" / "hydra_best.keras"

def run_backtest_sweep():
    print(f"🏛️ SOVEREIGN SWEEP: Testing Multiple Certainty Tiers ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')})")
    
    # 1. Load Model
    if not MODEL_PATH.exists():
        print(f"❌ Error: Model not found at {MODEL_PATH}")
        return
    
    features = build_feature_cols()
    print(f"🧠 Building Neural Architecture ({len(features)} features)...")
    model = build_kraken(n_features=len(features), context_window=120)
    
    print("📦 Loading Weights...")
    model.load_weights(str(MODEL_PATH))
    
    # 2. Fetch Live Data
    df_raw = fetch_live_kat_data('BTCUSD', 500, '15m')
    df = compute_indicators(df_raw)
    
    data = df[features].values.astype('float32')
    raw_prices = df['close'].values
    timestamps = df.index
    
    ctx = 120; f = 15
    test_range = range(len(df) - ctx - f - 100, len(df) - ctx - f)
    
    # 3. Inference
    X = []
    for i in test_range:
        window = data[i:i+ctx]
        window_scaled = (window - window.mean(0)) / (window.std(0) + 1e-8)
        X.append(window_scaled)
    
    X = np.array(X)
    outputs = model.predict(X, verbose=0)
    
    preds = outputs[0][:, -1, 0] 
    certs = np.mean(outputs[1], axis=1) 
    
    print(f"📡 Model Certainty Distribution: {certs.min():.4f} to {certs.max():.4f}")
    
    # 4. Sweep Simulation
    thresholds = [0.95, 0.96, 0.965, 0.97, 0.975]
    fee_rate = 0.0012
    
    print("\n" + "="*70)
    print(f"{'Threshold':<12} | {'Trades':<8} | {'Win Rate':<10} | {'Net PnL':<10} | {'Final Wallet'}")
    print("-" * 70)
    
    best_threshold = 0
    max_pnl = -999
    
    for thr in thresholds:
        wallet = 200.0
        n_trades = 0
        wins = 0
        
        for idx, i in enumerate(test_range):
            cert = certs[idx]
            if cert >= thr:
                curr_price = raw_prices[i + ctx - 1]
                future_price = raw_prices[i + ctx + f - 1]
                side = "LONG" if preds[idx] > 0 else "SHORT"
                
                pnl_pct = (future_price - curr_price) / curr_price if side == "LONG" else (curr_price - future_price) / curr_price
                net_pnl_pct = pnl_pct - (fee_rate * 2) 
                
                pnl_usd = wallet * net_pnl_pct
                wallet += pnl_usd
                n_trades += 1
                if pnl_usd > 0: wins += 1
        
        wr = (wins / n_trades * 100) if n_trades > 0 else 0
        total_pnl = wallet - 200.0
        print(f"{thr:<12.3f} | {n_trades:<8} | {wr:<9.1f}% | ${total_pnl:<9.2f} | ${wallet:.2f}")
        
        if total_pnl > max_pnl and n_trades > 0:
            max_pnl = total_pnl
            best_threshold = thr

    print("="*70)
    if best_threshold:
        print(f"✅ Recommended Threshold for Today: {best_threshold:.3f}")
    else:
        print("⚠️ No profitable threshold found for this 100-window sample.")
    print("\n[NOTE] Epoch 13 is still in the 'Discovery' phase. High loss today is expected due to the vertical bull run.")

if __name__ == "__main__":
    run_backtest_sweep()
