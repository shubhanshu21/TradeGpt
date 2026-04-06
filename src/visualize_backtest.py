"""
HYDRA SOVEREIGN VISUALIZER (V2.0) 📊⚖️
======================================
Focus: Zero-Margin High-Resolution Performance Zoom.
Outputs to: backtest_honesty.png
"""

import os, time, sys
import numpy as np
import tensorflow as tf
import keras
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Paths
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

# Architecture Imports
from architectures.hydra import Hydra, HydraBlock, GatedMoE, MLAAttention, AttnRes, VSN, RMSNorm, DualScaleFusion, TemporalGating, SovereignLoss
from preprocess import add_derived_features, build_feature_cols, KATScaler
from fetch_data import fetch_live_kat_data

def prepare_hydra_targets(X: np.ndarray, mtp_steps: int = 5) -> tuple:
    if X is None or len(X.shape) < 3: return X, None
    B, L, F = X.shape
    T = L - mtp_steps
    X_in = X[:, :T, :] 
    y_blocks = []
    for s in range(0, mtp_steps):
        y_blocks.append(X[:, T+s:T+s+1, 3:4]) 
    return X_in, np.concatenate(y_blocks, axis=-1)

def visualize_performance(model_path: str, timeframe: str = "1m"):
    print(f"🔬 INITIALIZING ZOOM VISUALIZER (V2.0) | Source: {model_path}")
    custom_objs = {
        "Hydra": Hydra, "HydraBlock": HydraBlock, "GatedMoE": GatedMoE,
        "MLAAttention": MLAAttention, "AttnRes": AttnRes, "VSN": VSN,
        "RMSNorm": RMSNorm, "DualScaleFusion": DualScaleFusion, 
        "TemporalGating": TemporalGating, "SovereignLoss": SovereignLoss
    }
    model = keras.models.load_model(model_path, custom_objects=custom_objs, compile=False)
    
    # Fetch 600 points (Focusing on a 4-hour window)
    df = fetch_live_kat_data(symbol="BTCUSD", n_candles=600, timeframe=timeframe)
    if df is None: return
    
    df_feat = add_derived_features(df)
    features = build_feature_cols()
    data = df_feat[features].values.astype("float32")
    
    scaler_p = ROOT / "models/scaler_base.pkl"
    scaler = KATScaler.load(scaler_p)
    scaled_data = scaler.transform_X(data)
    
    from numpy.lib.stride_tricks import sliding_window_view
    Xs = sliding_window_view(scaled_data, window_shape=(360, len(features))).squeeze()
    X_in, y_true = prepare_hydra_targets(Xs, mtp_steps=5)
    
    y_pred = model.predict(X_in.astype("float32"), verbose=0, batch_size=32)
    
    # 5. Invert and Finalize
    y_true_usd = scaler.inverse_y(y_true[:, 0, 0])
    y_pred_usd = scaler.inverse_y(y_pred[:, 0, 0])
    
    # Filter to the LATEST 120 points only (2 hours)
    y_true_usd = y_true_usd[-120:]
    y_pred_usd = y_pred_usd[-120:]
    
    print(f"📊 Crafting HIGH-RESOLUTION ZOOM ({len(y_true_usd)} points)...")
    plt.figure(figsize=(16, 8))
    plt.style.use('dark_background')
    
    plt.plot(y_true_usd, color='#00FFFF', linewidth=3.0, label='ACTUAL BTC (TRUTH)')
    plt.plot(y_pred_usd, color='#FF00FF', linewidth=2.0, linestyle='--', label='HYDRA PREDICTION')
    
    # Calculate stats
    avg_err = np.mean(np.abs(y_true_usd - y_pred_usd))
    dir_acc = np.mean(np.sign(np.diff(y_true_usd)) == np.sign(np.diff(y_pred_usd))) * 100
    
    plt.title(f'HYDRA ZOOM: ${avg_err:.2f} Avg Error | {dir_acc:.1f}% Trend Accuracy', fontsize=16, color='white')
    plt.xlabel('Last 120 Minutes', color='#888888')
    plt.ylabel('BTC Price (USD)', color='#888888')
    plt.grid(True, linestyle=':', alpha=0.2)
    plt.legend(loc='upper right')
    
    # Force Y-axis to be tight around the price
    p_min = min(np.min(y_true_usd), np.min(y_pred_usd)) - 5
    p_max = max(np.max(y_true_usd), np.max(y_pred_usd)) + 5
    plt.ylim(p_min, p_max)
    
    plot_path = ROOT / "backtest_honesty.png"
    plt.savefig(plot_path, dpi=180)
    plt.close()
    
    print(f"✅ ZOOM VISUAL COMPLETED: {plot_path}")
    print(f"📊 FINAL STATS (Visual): MAE ${avg_err:.2f} | DIR {dir_acc:.1f}%")

if __name__ == "__main__":
    MODEL_P = str(ROOT / "models/hydra_best.keras")
    visualize_performance(MODEL_P)
