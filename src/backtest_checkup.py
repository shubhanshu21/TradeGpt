"""
HYDRA SOVEREIGN NEURAL CHECKUP (V1.2) - DELTA EDITION 
=====================================================
Diagnostic: Real-Time Directional Delta Accuracy (+/- Change).
This measures the 'UP or DOWN' guess from the CURRENT price.
"""

import os, time, sys
import numpy as np
import tensorflow as tf
import keras
import pandas as pd
from pathlib import Path

# Paths
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

# Architecture Imports
from architectures.hydra import Hydra, HydraBlock, GatedMoE, MLAAttention, AttnRes, VSN, RMSNorm, DualScaleFusion, TemporalGating, SovereignLoss
from preprocess import build_dataset, KATScaler
from fetch_data import fetch_live_kat_data

def prepare_hydra_targets(X: np.ndarray, mtp_steps: int = 5) -> tuple:
    if X is None or len(X.shape) < 3: return X, None
    B, L, F = X.shape
    T = L - mtp_steps
    X_in = X[:, :T, :] 
    y_blocks = []
    for s in range(0, mtp_steps) : y_blocks.append(X[:, T+s:T+s+1, 3:4]) 
    return X_in, np.concatenate(y_blocks, axis=-1)

# ── DIAGNOSTIC ENGINE ─────────────────────────────────────────────────────────

def run_neural_checkup(model_path: str, timeframe: str = "1m"):
    print(f"🔬 INITIALIZING SOVEREIGN DELTA-CHECKUP | Source: {model_path}")
    
    # 1. Load Brain
    custom_objs = {
        "Hydra": Hydra, "HydraBlock": HydraBlock, "GatedMoE": GatedMoE,
        "MLAAttention": MLAAttention, "AttnRes": AttnRes, "VSN": VSN,
        "RMSNorm": RMSNorm, "DualScaleFusion": DualScaleFusion, 
        "TemporalGating": TemporalGating, "SovereignLoss": SovereignLoss
    }
    model = keras.models.load_model(model_path, custom_objects=custom_objs, compile=False)
    
    # 2. Fetch Latest 1500 Candles
    df = fetch_live_kat_data(symbol="BTCUSD", n_candles=1500, timeframe=timeframe)
    if df is None or len(df) < 500: return 0

    # 3. Scaler & Prep
    scaler_p = ROOT / "models/scaler_base.pkl"
    scaler = KATScaler.load(scaler_p)
    ds = build_dataset(df, context_window=360, forecast_steps=1, scaler=scaler)
    
    X_test_raw = ds["X_test"]
    X_in, y_true_all = prepare_hydra_targets(X_test_raw, mtp_steps=5)
    
    # Capture the price exactly at the Moment of Prediction (The very last input step)
    # Shape of X_in is (B, 355, 23). The price is index 3.
    last_input_prices = X_in[:, -1, 3] # (B,)
    
    # 4. Neural Predict
    print(f"🚀 Predicting {len(X_in)} market intervals...")
    y_pred_all = model.predict(X_in.astype("float32"), verbose=0, batch_size=32)
    
    # Focus on the next 1-minute target
    y_true_1min = y_true_all[:, 0, 0]
    y_pred_1min = y_pred_all[:, 0, 0]
    
    # ── THE REAL ALPHA CALCULATION ──────────────────────────────────────────
    # We want to know if y_pred is GREATER or LESS than its own starting point
    delta_true = y_true_1min - last_input_prices
    delta_pred = y_pred_1min - last_input_prices
    
    # Directional Delta Accuracy
    dir_acc = np.mean(np.sign(delta_true) == np.sign(delta_pred)) * 100
    
    # USD MAE
    t_usd = scaler.inverse_y(y_true_1min)
    p_usd = scaler.inverse_y(y_pred_1min)
    mae_usd = np.mean(np.abs(t_usd - p_usd))
    
    print("\n" + "="*60)
    print("📊 SOVEREIGN DELTA-CHECKUP (HONEST ALPHA TEST)")
    print("="*60)
    print(f"   - Target:          Predict +1m Close vs Current")
    print(f"   - Neural MAE:      ${mae_usd:,.2f} USD")
    print(f"   - Delta Accuracy:  {dir_acc:.1f}%")
    
    if dir_acc > 53.0:
        print(f"   ✅ STATUS: PROFITABLE ALPHA ({dir_acc:.1f}%)")
    elif dir_acc > 50.5:
        print(f"   ⚠️ STATUS: WEAK ALPHA (Low conviction)")
    else:
        print(f"   🛑 STATUS: NO ALPHA (Model is lagging)")
    print("="*60)
    return dir_acc

if __name__ == "__main__":
    MODEL_P = str(ROOT / "models/hydra_best.keras")
    if os.path.exists(MODEL_P):
        run_neural_checkup(MODEL_P)
    else:
        print(f"Error: Model not found at {MODEL_P}")
