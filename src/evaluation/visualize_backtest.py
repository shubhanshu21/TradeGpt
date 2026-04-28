import os, time, sys
import numpy as np
import tensorflow as tf
import keras
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Paths
ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

from core.hydra import build_kraken
from data.preprocess import compute_indicators, build_feature_cols
from exchange.fetch_data import fetch_live_kat_data

def visualize_performance(model_path: str, timeframe: str = "15m"):
    print(f"🔬 INITIALIZING SINGULARITY VISUALIZER (V10.3) | Source: {model_path}")
    
    ctx = 120
    forecast = 15
    n_feat = 42
    
    # 1. Build and Load
    print(f"🏗️  Re-building Iron Oracle V11.0 (Phase 5)...")
    model = build_kraken(n_features=n_feat, context_window=ctx, forecast_steps=forecast)
    model.load_weights(model_path)
    
    # 2. Fetch Data
    print(f"📡 Fetching 1,000 candles for visualization...")
    from exchange.fetch_data import fetch_live_kat_data
    df = fetch_live_kat_data(symbol="BTCUSD", n_candles=1000, timeframe=timeframe)
    if df is None: return
    
    # 3. Preprocess with DLS (Dynamic Local Scaling)
    df_feat = compute_indicators(df)
    features = build_feature_cols()
    data = df_feat[features].values.astype("float32")
    
    # We must scale windows exactly like training (Abyss-Streamer V4.7) 
    def prepare_dls_window(idx):
        x_raw = data[idx : idx + ctx]
        l_mean = x_raw.mean(axis=0); l_std = x_raw.std(axis=0) + 1e-8
        return (x_raw - l_mean) / l_std
    
    # 4. Generate Windows and Predict
    num_windows = 150
    Xs = np.array([prepare_dls_window(i) for i in range(len(data) - ctx)])
    Xs = Xs[-num_windows:]
    
    print(f"🔬 Generating {len(Xs)} predictions...")
    out = model.predict(Xs, verbose=0, batch_size=16)
    preds = out[0] # (N, 16, 3)
    
    # 5. Extract Targets (Terminal 15-step)
    y_true_usd = df_feat["close"].values[-(num_windows):]
    
    # Prediction logic: entry + (scaled_return * entry_std)
    y_pred_usd = []
    for i in range(num_windows):
        idx = len(data) - num_windows + i - ctx
        x_raw = data[idx : idx + ctx]
        l_mean = x_raw.mean(axis=0); l_std = x_raw.std(axis=0) + 1e-8
        
        entry_p = df_feat['close'].iloc[idx + ctx - 1]
        pred_scaled_ret = preds[i, -1, 0] # terminal return
        
        # Denormalize: scaled = (raw - mean) / std -> raw = scaled * std + mean
        # But our target is 'close' return.
        y_pred_usd.append(entry_p + (pred_scaled_ret * l_std[features.index('close')]))
    
    y_pred_usd = np.array(y_pred_usd)
    
    print(f"📊 Crafting HIGH-RESOLUTION ZOOM (Last 150 Cycles)...")
    plt.figure(figsize=(16, 8))
    plt.style.use('dark_background')
    
    plt.plot(y_true_usd, color='#00FFFF', linewidth=3.0, label='ACTUAL BTC (TRUTH)')
    plt.plot(y_pred_usd, color='#FFD700', linewidth=2.0, linestyle='--', label='IRON ORACLE FORECAST')
    
    # Calculate stats
    avg_err = np.mean(np.abs(y_true_usd - y_pred_usd))
    dir_acc = np.mean(np.sign(np.diff(y_true_usd)) == np.sign(np.diff(y_pred_usd))) * 100
    
    plt.title(f'IRON ORACLE V11.0: ${avg_err:.2f} Avg Error | {dir_acc:.1f}% Trend Accuracy', fontsize=16, color='white')
    plt.xlabel('Last 150 Candles (15m Resolution)', color='#888888')
    plt.ylabel('BTC Price (USD)', color='#888888')
    plt.grid(True, linestyle=':', alpha=0.2)
    plt.legend(loc='upper right')
    
    plot_path = ROOT / "backtest_honesty.png"
    plt.savefig(plot_path, dpi=180)
    plt.close()
    
    print(f"✅ VISUAL COMPLETED: {plot_path}")
    print(f"📊 FINAL STATS (Visual): MAE ${avg_err:.2f} | DIR {dir_acc:.1f}%")

if __name__ == "__main__":
    MODEL_P = str(ROOT / "models/hydra_best.keras")
    visualize_performance(MODEL_P)
