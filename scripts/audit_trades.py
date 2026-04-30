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

def generate_trade_report():
    print("⚓ SOVEREIGN TRADE AUDIT — Memory-Safe Mode")
    print("="*50)
    
    # 1. Load the "Best" Model
    model_p = ROOT / "models" / "hydra_best.keras"
    if not model_p.exists():
        print("❌ No model found.")
        return
        
    model = build_kraken(n_features=42, context_window=120)
    model.load_weights(str(model_p))
    
    # 2. Fetch Backtest Data (Last 500 candles)
    df_raw = fetch_live_kat_data('BTCUSD', 500, '15m')
    df = compute_indicators(df_raw)
    features = build_feature_cols()
    data = df[features].values.astype('float32')
    raw_prices = df['close'].values
    timestamps = df.index.values
    
    # 3. Process in Chunks (Memory Safe)
    ctx = 120; f = 15; threshold = 80
    indices = list(range(len(df) - ctx - f - 350, len(df) - ctx - f))
    
    trades_list = []
    print(f"{'Timestamp':<20} | {'Type':<6} | {'1x Net %':<10} | {'10x Net %':<10}")
    print("-" * 65)
    
    pos_size = 2000.0; fee_rate = 0.0012
    
    # Process 32 windows at a time to save RAM
    for start_idx in range(0, len(indices), 32):
        chunk_indices = indices[start_idx : start_idx + 32]
        X_chunk = np.array([(data[i:i+ctx] - data[i:i+ctx].mean(0)) / (data[i:i+ctx].std(0) + 1e-8) for i in chunk_indices])
        
        outputs = model(X_chunk, training=False)
        traj = outputs[0].numpy()[:, -1, 0] 
        certs = np.mean(outputs[1].numpy(), axis=1)
        
        # Local normalization for the chunk
        c_min, c_max = certs.min(), certs.max()
        c_pct = (certs - c_min) / (c_max - c_min + 1e-9) * 100
        
        for i, idx in enumerate(chunk_indices):
            if c_pct[i] >= threshold:
                side = "LONG" if traj[i] > 0 else "SHORT"
                entry_p = raw_prices[idx + ctx - 1]
                exit_p  = raw_prices[idx + ctx + f - 1]
                
                price_diff = (exit_p - entry_p) if side == "LONG" else (entry_p - exit_p)
                raw_ret_1x = price_diff / entry_p
                net_ret_1x = raw_ret_1x - fee_rate
                net_ret_10x = (raw_ret_1x * 10) - (fee_rate * 10)
                roi_usd = (pos_size * raw_ret_1x) - (pos_size * fee_rate)
                
                ts = pd.to_datetime(timestamps[idx + ctx - 1]).strftime('%Y-%m-%d %H:%M')
                print(f"{ts:<20} | {side:<6} | {net_ret_1x*100:>+8.2f}% | {net_ret_10x*100:>+8.2f}%")
                
                trades_list.append({
                    "timestamp": ts, "type": side, "entry_p": entry_p, "exit_p": exit_p,
                    "net_1x": net_ret_1x*100, "net_10x": net_ret_10x*100, "roi_usd": roi_usd
                })

    # Export to CSV
    csv_path = ROOT / "logs" / "sovereign_trades_audit.csv"
    pd.DataFrame(trades_list).to_csv(csv_path, index=False)
    print(f"\n✅ SUCCESS: {len(trades_list)} trades saved to {csv_path}")

if __name__ == "__main__":
    generate_trade_report()
