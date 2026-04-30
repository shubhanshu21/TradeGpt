import os, sys, json
import numpy as np
import pandas as pd
import tensorflow as tf
import keras
from pathlib import Path

# Setup paths
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from core.hydra import build_kraken
from data.preprocess import compute_indicators, build_feature_cols
from exchange.fetch_data import fetch_live_kat_data

def run_diagnostic_bench():
    print("⚓ TRIGGERING DIAGNOSTIC STRIKE FEED...")
    
    # 1. Load weights
    model_p = ROOT / "models" / "hydra_checkpoint_E004.keras"
    model = build_kraken(n_features=42, context_window=120)
    model.load_weights(str(model_p))
    
    # 2. Fetch live slice
    df_raw = fetch_live_kat_data('BTCUSD', 500, '15m')
    df = compute_indicators(df_raw)
    features = build_feature_cols()
    data = df[features].values.astype('float32')
    raw_prices = df['close'].values
    
    ctx = 120; f = 15
    indices = range(len(df) - ctx - f - 100, len(df) - ctx - f)
    X = np.array([(data[i:i+ctx] - data[i:i+ctx].mean(0)) / (data[i:i+ctx].std(0) + 1e-8) for i in indices])
    usd_diffs = np.array([raw_prices[i + ctx + f - 1] - raw_prices[i + ctx - 1] for i in indices])
    
    # 3. Inference
    outputs = model(X, training=False)
    traj = outputs[0].numpy()[:, -1, 0]
    certs = np.mean(outputs[1].numpy(), axis=1)
    
    c_pct = (certs - certs.min()) / (certs.max() - certs.min() + 1e-9) * 100
    
    # 4. Save trades
    recent_trades = []
    mask80 = c_pct >= 80
    if mask80.any():
        t_indices = np.array(indices)[mask80]
        t_traj = traj[mask80]
        t_usd = usd_diffs[mask80]
        
        for i in range(max(0, len(t_indices)-10), len(t_indices)):
            idx = t_indices[i]
            entry_p = raw_prices[idx + ctx - 1]
            price_move_pct = (t_usd[i] / entry_p)
            side = "LONG" if t_traj[i] > 0 else "SHORT"
            
            # Net profit (0.12% fees)
            raw_ret = price_move_pct if side == "LONG" else -price_move_pct
            net_ret = raw_ret - 0.0012
            
            recent_trades.append({
                "timestamp": pd.to_datetime(df.index[idx + ctx - 1]).strftime("%H:%M"),
                "side": side,
                "entry": float(entry_p),
                "net_pct": float(net_ret * 100)
            })
            
    with open(ROOT / "logs" / "recent_sim_trades.json", "w") as f:
        json.dump(recent_trades[::-1], f, indent=4)
        
    print(f"✅ Success: {len(recent_trades)} strikes captured in feed.")

if __name__ == "__main__":
    run_diagnostic_bench()
