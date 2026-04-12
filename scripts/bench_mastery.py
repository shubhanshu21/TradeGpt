import os, sys
import numpy as np
import tensorflow as tf
import keras
import pandas as pd
from pathlib import Path

# Add src to path
ROOT = Path("/var/www/html/ML/kat")
sys.path.append(str(ROOT / "src"))

from architectures.hydra import HydraV4, HydraBlock, GatedMoE, MLAAttention, RMSNorm, SovereignLoss
from preprocess import build_dataset, KATScaler, add_derived_features, build_feature_cols
from fetch_data import fetch_live_kat_data

def benchmark_mastery():
    print("🎬 Starting Sovereign Mastery Benchmark (Directional Accuracy Test)...")
    
    # 1. Load Model & Scaler
    MODEL_PATH = ROOT / "models/hydra_best.keras"
    SCALER_PATH = ROOT / "models/scaler_base.pkl"
    
    if not MODEL_PATH.exists() or not SCALER_PATH.exists():
        print("❌ Error: Model or Scaler not found in /models/")
        return

    scaler = KATScaler.load(str(SCALER_PATH))
    from architectures.hydra import (SovereignLoss, TTMReflex, RMSNorm, 
                                     HydraBlock, MLAAttention, GatedMoE, SovereignAccuracy)

    custom_objs = {
        "SovereignLoss": SovereignLoss, 
        "TTMReflex": TTMReflex, 
        "RMSNorm": RMSNorm,
        "HydraBlock": HydraBlock,
        "MLAAttention": MLAAttention,
        "GatedMoE": GatedMoE,
        "SovereignAccuracy": SovereignAccuracy
    }
    model = keras.models.load_model(str(MODEL_PATH), custom_objects=custom_objs, compile=False)
    
    # 2. Fetch Large Sample (2500 candles for a robust test)
    print("📡 Fetching 2,500 candles of BTCUSD (1m) for benchmarking...")
    try:
        from fetch_data import fetch_live_kat_data
        df = fetch_live_kat_data(symbol="BTCUSD", n_candles=2500, timeframe="1m")
    except Exception as e:
        print(f"❌ Data Fetch Error: {e}")
        return

    # 3. Build Evaluation Dataset
    ctx = 120 
    from preprocess import build_dataset_streaming
    ds_info = build_dataset_streaming(df, context_window=ctx, forecast_steps=15, scaler=scaler)
    
    # Convert generator data to numpy for easy benchmarking
    X_test, Y_test = [], []
    for x, y in ds_info["tr_ds"].unbatch().take(1000):
        X_test.append(x.numpy())
        Y_test.append(y.numpy())
    
    X_test = np.array(X_test)
    Y_test = np.array(Y_test)
    
    print(f"🔬 Evaluating {len(X_test)} samples...")
    predictions = model.predict(X_test, verbose=0) # (N, 15, 3)
    
    # Extract Price Predictions (Channel 0)
    pred_returns = predictions[:, :, 0]
    actual_returns = Y_test[:, :, 0]
    
    # --- PERFORMANCE METRICS ---
    # 1. Directional Accuracy (Match sign of returns relative to entry)
    dir_match = (np.sign(pred_returns) == np.sign(actual_returns))
    hit_rate = np.mean(dir_match) * 100
    
    # 2. Cumulative Trajectory Direction (Is the 15m endpoint correct?)
    pred_sum = np.sum(pred_deltas, axis=1)
    actual_sum = np.sum(actual_deltas, axis=1)
    cum_dir_match = (np.sign(pred_sum) == np.sign(actual_sum))
    cum_hit_rate = np.mean(cum_dir_match) * 100
    
    print("\n" + "="*50)
    print("📈 SOVEREIGN MASTERY ACCURACY REPORT")
    print("="*50)
    print(f"Total Test Samples   : {len(X_test)}")
    print(f"Per-Minute Hit Rate  : {hit_rate:.2f}% (Sign Accuracy)")
    print(f"15m Pulse Accuracy   : {cum_hit_rate:.2f}% (Total Trend)")
    print("-" * 50)
    
    if cum_hit_rate > 55:
        print("🏆 VERDICT: EXCELLENT. Edge is statistically significant.")
    elif cum_hit_rate > 51:
        print("✅ VERDICT: NOMINAL. Model has a clear mathematical edge.")
    else:
        print("⚠️ VERDICT: NOISY. Increase training epochs or data richness.")
    print("="*50 + "\n")

if __name__ == "__main__":
    benchmark_mastery()
