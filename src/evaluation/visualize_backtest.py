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
from data.preprocess import compute_indicators, build_feature_cols, KATScaler

def visualize_performance(model_path: str, timeframe: str = "1m"):
    print(f"🔬 INITIALIZING SINGULARITY VISUALIZER (V10.3) | Source: {model_path}")
    
    ctx = 120
    forecast = 15
    features = build_feature_cols()
    n_feat = len(features)
    
    # 1. Build and Load
    print(f"🏗️  Re-building Kraken V10.3...")
    model = build_kraken(n_features=n_feat, context_window=ctx, forecast_steps=forecast)
    model.load_weights(model_path)
    
    scaler_p = ROOT / "models/scaler_base.pkl"
    scaler = KATScaler.load(scaler_p)
    
    # 2. Fetch Data
    print(f"📡 Fetching 1,000 candles for visualization...")
    from fetch_data import fetch_live_kat_data
    df = fetch_live_kat_data(symbol="BTCUSD", n_candles=1000, timeframe=timeframe)
    if df is None: return
    
    # 3. Preprocess
    df_feat = compute_indicators(df)
    data = df_feat[features].values.astype("float32")
    scaled_data = scaler.transform_X(data)
    
    # 4. Generate Windows and Predict
    from numpy.lib.stride_tricks import sliding_window_view
    Xs = sliding_window_view(scaled_data, window_shape=(ctx, n_feat)).squeeze()
    
    print(f"🔬 Generating {len(Xs)} predictions...")
    out = model.predict(Xs, verbose=0, batch_size=64)
    preds = out[0] # (N, 16, 3)
    
    # 5. Invert and Finalize
    # Actual Price at T+5min
    y_true_usd = df_feat["close"].values[ctx+5 : ctx+5+120]
    
    # Prediction for T+5min
    pred_returns = preds[:120, 5, 0] # Returns
    y_pred_usd = scaler.inverse_y(pred_returns.reshape(-1, 1)).flatten()
    
    print(f"📊 Crafting HIGH-RESOLUTION ZOOM (Last 120 Minutes)...")
    plt.figure(figsize=(16, 8))
    plt.style.use('dark_background')
    
    plt.plot(y_true_usd, color='#00FFFF', linewidth=3.0, label='ACTUAL BTC (TRUTH)')
    plt.plot(y_pred_usd, color='#FF00FF', linewidth=2.0, linestyle='--', label='HYDRA PREDICTION')
    
    # Calculate stats
    avg_err = np.mean(np.abs(y_true_usd - y_pred_usd))
    dir_acc = np.mean(np.sign(np.diff(y_true_usd)) == np.sign(np.diff(y_pred_usd))) * 100
    
    plt.title(f'HYDRA V10.3 SINGULARITY: ${avg_err:.2f} Avg Error | {dir_acc:.1f}% Trend Accuracy', fontsize=16, color='white')
    plt.xlabel('Last 120 Minutes', color='#888888')
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
