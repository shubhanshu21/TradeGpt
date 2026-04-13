import os, sys
import numpy as np
import tensorflow as tf
import keras
import pandas as pd
from pathlib import Path

# Add src to path
ROOT = Path("/var/www/html/ML/kat")
sys.path.append(str(ROOT / "src"))

from core.hydra import build_kraken, SovereignLoss, CertaintyMetric, SovereignAccuracy
from data.preprocess import build_dataset_streaming, build_feature_cols, KATScaler
from exchange.fetch_data import fetch_live_kat_data

def benchmark_mastery():
    print("🎬 Starting Sovereign Mastery Benchmark (Directional Accuracy Test)...")
    
    # 1. Load Model & Scaler
    MODEL_PATH = ROOT / "models/hydra_best.keras"
    SCALER_PATH = ROOT / "models/scaler_base.pkl"
    
    if not MODEL_PATH.exists() or not SCALER_PATH.exists():
        print("❌ Error: Model or Scaler not found in /models/")
        return

    scaler = KATScaler.load(str(SCALER_PATH))
    
    # ── 1. Re-Build Kraken Archive (Functional Architecture) ─────────────────
    ctx = 120 
    forecast = 15
    n_feat = 27 # V10.3 Singularity Standard
    
    print(f"🏗️  Re-building Kraken V10.3 (Functional Architecture)...")
    model = build_kraken(n_features=n_feat, context_window=ctx, forecast_steps=forecast)
    
    print(f"🧠 Loading Weights from: {MODEL_PATH.name}")
    model.load_weights(str(MODEL_PATH))
    
    # 2. Load Local History (API Bypassed for speed/stability)
    HISTORY_P = ROOT / "data/BTCUSD_1m_history_120000.parquet"
    print(f"📡 Loading 5,000 candles from Local History: {HISTORY_P.name}")
    try:
        df = pd.read_parquet(str(HISTORY_P)).tail(5000)
    except Exception as e:
        print(f"❌ Data Load Error: {e}")
        return

    # 3. Build Evaluation Dataset
    ctx = 120 
    ds_info = build_dataset_streaming(df, context_window=ctx, forecast_steps=15, scaler=scaler)
    
    # Extract data from generator
    X_test, Y_test = [], []
    for x, y_all in ds_info["tr_ds"].unbatch().take(1000):
        y_price, _ = y_all
        X_test.append(x.numpy())
        Y_test.append(y_price.numpy())
    
    X_test = np.array(X_test)
    Y_test = np.array(Y_test)
    
    print(f"🔬 Evaluating {len(X_test)} samples...")
    # Dual output: prediction [0], certainty [1]
    preds_all = model.predict(X_test, verbose=0)
    predictions = preds_all[0] # (N, 16, 3)
    
    # Entry-Anchored Returns calculation
    # index 0 is the entry candle. index 1-15 are the future windows.
    p_entry = Y_test[:, 0, 0:1] # (N, 1)
    
    actual_returns = Y_test[:, 1:, 0] - p_entry 
    pred_returns   = predictions[:, 1:, 0] - p_entry
    
    # --- PERFORMANCE METRICS ---
    # 1. Directional Accuracy (Sign Match)
    dir_match = (np.sign(pred_returns) == np.sign(actual_returns))
    hit_rate = np.mean(dir_match) * 100
    
    # 2. Terminal Pulse Accuracy (Was it up/down at the end of 15m?)
    actual_terminal = actual_returns[:, -1]
    pred_terminal   = pred_returns[:, -1]
    cum_dir_match = (np.sign(pred_terminal) == np.sign(actual_terminal))
    cum_hit_rate = np.mean(cum_dir_match) * 100
    
    print("\n" + "="*50)
    print("📈 SOVEREIGN MASTERY ACCURACY REPORT")
    print("="*50)
    print(f"Total Test Samples    : {len(X_test)}")
    print(f"Per-Minute Hit Rate   : {hit_rate:.2f}% (Average Signal)")
    print(f"15m Terminal Accuracy : {cum_hit_rate:.2f}% (Trend Capture)")
    print("-" * 50)
    
    if cum_hit_rate > 55:
        print("🏆 VERDICT: EXCELLENT. High-Alpha identified.")
    elif cum_hit_rate > 51.5:
        print("✅ VERDICT: NOMINAL. Model has established a trading edge.")
    else:
        print("⚠️ VERDICT: NOISY. Needs more epochs or Predator features.")
    print("="*50 + "\n")

if __name__ == "__main__":
    benchmark_mastery()
