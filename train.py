"""
SOVEREIGN KRAKEN TRAINING ORCHESTRATOR (V4.0) ⚓🚀⚡
===================================================
- Model: HYDRA Sovereign Kraken (Adaptive Hardware)
- Target: MTP-15 (15-Minute Prophecy)
- Runtime: Optimized for 4-Core CPU or NVIDIA A40 (48GB)
"""

import os, time, argparse, gc
import numpy as np
import tensorflow as tf
import keras
import pandas as pd
from pathlib import Path
from datetime import datetime

# Paths
ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
CKPT_DIR = ROOT / "models"

import sys
sys.path.insert(0, str(ROOT / "src"))
from architectures.hydra import HydraV4, build_kraken, SovereignLoss, IS_GPU, init_kraken_hardware
from preprocess import build_dataset, KATScaler
from fetch_data import fetch_live_kat_data

def prepare_kraken_targets(X: np.ndarray, mtp_steps: int = 15) -> tuple:
    """Prepares MTP-15 targets (Predicting the next 15 mins)"""
    if X is None or len(X.shape) < 3: return X, None
    B, L, F = X.shape
    T = L - mtp_steps # The rest is the future
    X_in = X[:, :T, :] # Context
    # Targets: Future prices (B, 15)
    y_blocks = []
    for s in range(0, mtp_steps) : y_blocks.append(X[:, T+s:T+s+1, 3:4]) 
    y_t = np.concatenate(y_blocks, axis=-1).squeeze(axis=1) # (B, 15)
    return X_in, y_t

def train_kraken(args):
    # Arm the Hardware (Enable Mixed-Precision on GPU)
    init_kraken_hardware()
    
    print("\n" + "="*60)
    print(f"  { '🚀 GPU MODE' if IS_GPU else '🐌 CPU MODE' } — SOVEREIGN KRAKEN V4.0 (32 Experts + MTP-15)")
    print("="*60)

    # 1. Hardware-Aware Configuration
    BATCH_S = 128 if IS_GPU else 32
    EPOCHS  = args.epochs
    CANDLES = 120000 if IS_GPU else 30000 # 30k is safer for stable CPU paging
    CTX_WIN = 1440 if IS_GPU else 360     # 24-hour vs 6-hour context
    
    # 2. Fetch Data (with Scaling Guard)
    CACHE_P = DATA_DIR / f"{args.symbol}_1m_history_60000.parquet"
    if CACHE_P.exists():
        print(f"📖 CACHE DETECTED: Loading {CANDLES:,} candles from local Sovereign Deep-History...")
        df = pd.read_parquet(CACHE_P).tail(CANDLES)
    else:
        print(f"📡 No cache found. Gathering {CANDLES:,} candles of {args.symbol} Alpha Data...")
        df = fetch_live_kat_data(symbol=args.symbol, n_candles=CANDLES, timeframe=args.timeframe)
    if df is None or len(df) == 0: 
        print("🛑 Error: No data available for training.")
        return

    # Build Hardened Dataset (No Lookahead Scaling)
    ds_raw = build_dataset(df, context_window=CTX_WIN, forecast_steps=1, 
                          scaler_save_path=str(CKPT_DIR / "scaler_base.pkl"))
    
    # 3. Create Kraken Targets
    X_tr_in, y_tr = prepare_kraken_targets(ds_raw["X_train"], mtp_steps=15)
    X_va_in, y_va = prepare_kraken_targets(ds_raw["X_val"],   mtp_steps=15)
    
    X_tr_in = X_tr_in.astype("float32")
    y_tr    = y_tr.astype("float32")
    
    # 4. Neural Construction
    model = build_kraken(n_features=ds_raw["n_features"])
    # Force Parameter Population with actual shape
    dummy_x = np.zeros((1, CTX_WIN, ds_raw["n_features"])).astype("float32")
    _ = model(dummy_x) # This triggers the weights to materialize
    model.summary()

    # 5. Callbacks (Sovereign Safety)
    callbacks = [
        keras.callbacks.ModelCheckpoint(str(CKPT_DIR / "hydra_best.keras"), 
                                      monitor="val_mae", save_best_only=True),
        keras.callbacks.EarlyStopping(monitor="val_mae", patience=15, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_mae", factor=0.5, patience=5)
    ]

    # 6. Ignite Training
    print(f"\n🚀 IGNITION: Mission starts 300-Epoch Mastery on { 'A40' if IS_GPU else 'CPU' }...")
    history = model.fit(
        X_tr_in, y_tr,
        validation_data=(X_va_in, y_va),
        epochs=EPOCHS,
        batch_size=BATCH_S,
        callbacks=callbacks,
        verbose=1
    )
    
    # Save Final Alpha-Brain
    model.save(str(CKPT_DIR / "hydra_final.keras"))
    print("\n✓ MISSION COMPLETE: Sovereign Alpha-Brain Hardened & Saved.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="BTCUSD")
    parser.add_argument("--timeframe", default="1m")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--model", default="hydra")
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--candles", type=int, default=60000)
    args = parser.parse_args()
    
    CKPT_DIR.mkdir(exist_ok=True)
    train_kraken(args)