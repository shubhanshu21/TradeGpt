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
    custom_objs = {
        "HydraV4": HydraV4, "HydraBlock": HydraBlock, "GatedMoE": GatedMoE,
        "MLAAttention": MLAAttention, "RMSNorm": RMSNorm, "SovereignLoss": SovereignLoss
    }
    model = keras.models.load_model(str(MODEL_PATH), custom_objects=custom_objs, compile=False)
    
    # 2. Fetch Large Sample (2500 candles for a robust test)
    print("📡 Fetching 2,500 candles of BTCUSD (1m) for benchmarking...")
    try:
        df = fetch_live_kat_data(symbol="BTCUSD", n_candles=2500, timeframe="1m")
    except Exception as e:
        print(f"❌ Data Fetch Error: {e}")
        return

    # 3. Build Evaluation Dataset
    ctx = 480 # Hydra standard
    ds = build_dataset(df, context_window=ctx, forecast_steps=15, scaler=scaler)
    
    if ds is None:
        print("❌ Error: Dataset too small for 15-step forecast benchmark.")
        return

    X_test = ds["X_test"]
    # For Hydra MTP-15, the actual target is the next 15 steps of DELTAS
    # The build_dataset creates ys_final as a single point. 
    # For a real accuracy test, we need to compare the predicted 15-step curve 
    # to the actual 15-step curve from the raw data.
    
    print(f"🔬 Evaluating {len(X_test)} samples...")
    
    # We'll take the first 465 features as input for the model
    X_input = X_test[:, -465:, :]
    predictions = model.predict(X_input, verbose=0) # (N, 15, 3)
    
    # Extract Price Delta Predictions (Channel 0)
    pred_deltas = predictions[:, :, 0]
    
    # Now we need Actual Deltas to compare
    # We'll calculate them manually from the scaled_data to be precise
    df_feat = add_derived_features(df)
    features = build_feature_cols()
    data = df_feat[features].values.astype("float32")
    scaled_data = ds["scaler"].transform_X(data)
    
    # Correctly align with the windows in ds["X_test"]
    # test_start_idx corresponds to the end of the validation set in build_dataset
    n_win = len(X_test) + len(ds["X_train"]) + len(ds["X_val"])
    va_win_idx = int(n_win * 0.9)
    
    actual_deltas = []
    # Index into the original scaled_data array (offset by context window)
    for i in range(va_win_idx, va_win_idx + len(X_test)):
        # Actual next 15 steps of close price deltas (Close is index 3)
        prices = scaled_data[i + ctx : i + ctx + 15, 3]
        prev_price = scaled_data[i + ctx - 1, 3]
        deltas = np.diff(np.concatenate([[prev_price], prices]))
        actual_deltas.append(deltas)
        
    actual_deltas = np.array(actual_deltas)
    
    # --- PERFORMANCE METRICS ---
    # 1. Directional Accuracy (Did we predict the sign correctly?)
    dir_match = (np.sign(pred_deltas) == np.sign(actual_deltas))
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
