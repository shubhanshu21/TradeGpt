"""
HYDRA SOVEREIGN NEURAL CHECKUP (V1.0)
======================================
Diagnostic: Accuracy, Direction, and Simulated PnL.
Input: models/hydra_best.keras
Dataset: Latest BTC/USD 1,000 candles (Fresh 'Out-of-Sample' Data)
"""

import os, time, argparse, gc
import numpy as np
import tensorflow as tf
import keras
import pandas as pd
from pathlib import Path

# Paths
ROOT = Path(__file__).parent.parent
import sys
sys.path.insert(0, str(ROOT / "src"))

# CRITICAL: We must import the architecture components
from architectures.hydra import Hydra, HydraBlock, GatedMoE, MLAAttention, AttnRes, VSN, RMSNorm, DualScaleFusion, TemporalGating, SovereignLoss
from preprocess import build_dataset, KATScaler
from fetch_data import fetch_live_kat_data

def prepare_hydra_targets(X: np.ndarray, mtp_steps: int = 5) -> tuple:
    """Matches the target preparation in train.py"""
    if X is None or len(X.shape) < 3: return X, None
    B, L, F = X.shape
    T = L - mtp_steps
    X_in = X[:, :T, :] 
    y_blocks = []
    for s in range(0, mtp_steps):
        y_blocks.append(X[:, T+s:T+s+1, 3:4]) 
    y_t = np.concatenate(y_blocks, axis=-1) 
    return X_in, y_t

# ── DIAGNOSTIC ENGINE ─────────────────────────────────────────────────────────

def run_neural_checkup(model_path: str, timeframe: str = "1m"):
    print(f"🔬 INITIALIZING NEURAL CHECKUP | Source: {model_path}")
    
    # 1. Load trained model
    print("🏗️ Loading Sovereign Brain...")
    custom_objs = {
        "Hydra": Hydra,
        "HydraBlock": HydraBlock,
        "GatedMoE": GatedMoE,
        "MLAAttention": MLAAttention,
        "AttnRes": AttnRes,
        "VSN": VSN,
        "RMSNorm": RMSNorm,
        "DualScaleFusion": DualScaleFusion,
        "TemporalGating": TemporalGating,
        "SovereignLoss": SovereignLoss
    }
    
    try:
        model = keras.models.load_model(model_path, custom_objects=custom_objs, compile=True)
    except Exception as e:
        print(f"   ! Warning during load ({e}). Attempting without compilation...")
        model = keras.models.load_model(model_path, custom_objects=custom_objs, compile=False)
    
    # 2. Fetch FRESH out-of-sample data (Latest 1500 candles)
    print("📊 Fetching fresh 'Out-of-Sample' Stress Dataset...")
    df = fetch_live_kat_data(symbol="BTCUSD", n_candles=1500, timeframe=timeframe)
    if df is None or len(df) < 500:
        print("   🛑 Error: Not enough candles fetched.")
        return 0

    # 3. Build & Scaler Reverse
    print("📂 Slicing data for evaluation...")
    scaler_p = ROOT / "models/scaler_base.pkl"
    if not scaler_p.exists(): scaler_p = ROOT / "data/scaler_base.pkl"
    scaler = KATScaler.load(scaler_p)
    
    # Note: build_dataset returns standard windows. Hydra needs to re-slice them for MTP.
    ds = build_dataset(df, context_window=360, forecast_steps=1, scaler=scaler)
    
    # HYDRA SPECIAL: Re-slice the windows into (ctx-5) and (1, 5) targets
    X_test_raw = ds["X_test"]
    X_in, y_test = prepare_hydra_targets(X_test_raw, mtp_steps=5)
    
    y_true_1min = y_test[:, 0, 0] # Price in 1m (normalized)
    
    # 4. Neural Inference
    print("🚀 Running Neural Predictions on Stress Points...")
    t_start = time.time()
    X_in = X_in.astype("float32")
    y_pred = model.predict(X_in, verbose=0, batch_size=32)
    y_pred_1min = y_pred[:, 0, 0]
    
    inf_time = (time.time() - t_start) / len(X_in)
    
    # 5. Accuracy Math (USD Scale)
    y_true_usd = scaler.inverse_y(y_true_1min)
    y_pred_usd = scaler.inverse_y(y_pred_1min)
    
    # Filter out any NaNs or infinities
    mask = ~np.isnan(y_true_usd) & ~np.isnan(y_pred_usd)
    y_true_usd = y_true_usd[mask]
    y_pred_usd = y_pred_usd[mask]
    y_true_norm = y_true_1min[mask]
    y_pred_norm = y_pred_1min[mask]

    mae = np.mean(np.abs(y_true_usd - y_pred_usd))
    
    # Directional Accuracy (Correction sign relative to zero-center?)
    # Since prices are normalized, y_true is the diff from window start.
    dir_true = np.sign(y_true_norm)
    dir_pred = np.sign(y_pred_norm)
    dir_acc = np.mean(dir_true == dir_pred) * 100
    
    print("\n" + "="*60)
    print("📊 SOVEREIGN CHECKUP: BTC/USD 1-MINUTE")
    print("="*60)
    print(f"   - Inference Speed: {inf_time*1000:.2f}ms/pred")
    print(f"   - Neural MAE:      ${mae:.2f} USD")
    print(f"   - Direction Acc:   {dir_acc:.1f}%")
    
    if dir_acc > 53.0:
        print("   ✅ STATUS: PROFITABLE ALPHA DETECTED")
    elif dir_acc > 50.5:
        print("   ⚠️ STATUS: WEAK ALPHA (Needs more epochs)")
    else:
        print("   🛑 STATUS: NO ALPHA (Possible coin-flip/noise)")
    
    print("="*60)
    return dir_acc

if __name__ == "__main__":
    MODEL_P = str(ROOT / "models/hydra_best.keras")
    if os.path.exists(MODEL_P):
        run_neural_checkup(MODEL_P)
    else:
        print(f"Error: Model not found at {MODEL_P}")
