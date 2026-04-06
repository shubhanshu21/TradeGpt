#!/usr/bin/env python3
"""
SOVEREIGN ALPHA PILOT (V3.7.2) ⚓🚀
==========================================
- Timeframe: 1 minute (Institutional Scaling)
- Brain: HYDRA Sovereign MoE Transformer
- Targets: MTP-5 (Multi-Target Future Curve)
- Compliance: Delta Exchange (Market-Order + Brackets)
"""

import os, sys, time
import numpy as np
import tensorflow as tf
import keras
import pandas as pd
from datetime import datetime
from pathlib import Path

# Paths
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))

# Architecture Imports (Required for Keras Serialization)
from architectures.hydra import Hydra, HydraBlock, GatedMoE, MLAAttention, AttnRes, VSN, RMSNorm, DualScaleFusion, TemporalGating, SovereignLoss
from delta_client import DeltaClient
from fetch_data    import fetch_live_kat_data
from preprocess    import KATScaler, build_feature_cols, add_derived_features

# ── CONFIG ───────────────────────────────────────────────────────────────────
SYMBOL     = "BTCUSD"
SIZE       = 1           # Delta Units
THRESHOLD  = 0.08        # 0.08% Neural Conviction (Fee-Aware)
TIMEFRAME  = "1m"
MODEL_FILE = "hydra_best.keras" # Uses the 'Best' model from training
CONTEXT_L  = 360
INPUT_N    = 355 # MTP Slicing (360 -> 355 Input + 5 Target)

# ── PILOT HUD ────────────────────────────────────────────────────────────────
C_RESET = "\033[0m"
C_BOLD  = "\033[1m"
C_GREEN = "\033[32m"
C_RED   = "\033[31m"
C_CYAN  = "\033[36m"
C_YELLOW = "\033[33m"

def log_event(msg, color=C_RESET):
    stamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{C_CYAN}{stamp}{C_RESET}] {color}{msg}{C_RESET}")

def run_pilot():
    print("="*60)
    print(f"  SOVEREIGN ALPHA PILOT (V3.7.2) — TESTNET PILOT [ {SYMBOL} ]")
    print("="*60)

    # 1. Initialize
    client = DeltaClient(testnet=True)
    scaler_p = ROOT / "models/scaler_base.pkl"
    model_p  = ROOT / f"models/{MODEL_FILE}"

    if not scaler_p.exists():
        log_event("Error: Scaler missing. Train the model first.", C_RED)
        return
    scaler = KATScaler.load(str(scaler_p))

    # Load Model with custom objects
    if not model_p.exists():
        log_event(f"Error: No model at {model_p}", C_RED)
        return
    
    log_event(f"🏗️ Syncing Neural Brain: {MODEL_FILE}...")
    custom_objs = {
        "Hydra": Hydra, "HydraBlock": HydraBlock, "GatedMoE": GatedMoE,
        "MLAAttention": MLAAttention, "AttnRes": AttnRes, "VSN": VSN,
        "RMSNorm": RMSNorm, "DualScaleFusion": DualScaleFusion, 
        "TemporalGating": TemporalGating, "SovereignLoss": SovereignLoss
    }
    model = keras.models.load_model(str(model_p), custom_objects=custom_objs, compile=False)
    log_event("✓ Brain Sync Complete. Entering Live Loop.", C_GREEN)

    while True:
        try:
            # 2. Market Data Pull
            log_event(f"📡 Polling {SYMBOL} Market Stream...")
            df = fetch_live_kat_data(symbol=SYMBOL, n_candles=CONTEXT_L + 10, timeframe=TIMEFRAME)
            
            # Prepare Features for Neural Input
            df_feat = add_derived_features(df)
            features = build_feature_cols()
            data = df_feat[features].values.astype("float32")
            
            # Transform and Slice the latest window
            scaled = scaler.transform_X(data)
            # Take the latest 355 points for Hydra Input
            X_in = scaled[-INPUT_N:].reshape(1, INPUT_N, len(features))
            X_in = X_in.astype("float32")
            
            # 3. Neural Pulse (Predicting the next 5 mins)
            pred = model.predict(X_in, verbose=0) # Shape (1, 1, 5)
            curve = pred[0, 0] # 5 points
            mean_move = np.mean(curve) # Average direction of next 5 mins
            
            p_str = " | ".join([f"{x:+.4f}" for x in curve])
            log_event(f"🔮 NEURAL CURVE: [ {p_str} ] | {C_BOLD}AVG: {mean_move:+.4f}{C_RESET}")

            # 4. Strategy Bridge
            # Check Position
            pos = client.get_positions(SYMBOL)
            is_in_long  = any(p['size'] > 0 for p in pos)
            is_in_short = any(p['size'] < 0 for p in pos)

            # Signal Logic
            if mean_move > THRESHOLD:
                if not is_in_long:
                    log_event(f"📈 SIGNAL: BULLISH CONVICTION DETECTED. Executing LONG...", C_GREEN)
                    if is_in_short: client.close_position(SYMBOL)
                    client.place_market_order(SYMBOL, "buy", SIZE)
                    # Note: Add SL/TP logic here in DeltaClient if desired
            
            elif mean_move < -THRESHOLD:
                if not is_in_short:
                    log_event(f"📉 SIGNAL: BEARISH CONVICTION DETECTED. Executing SHORT...", C_RED)
                    if is_in_long: client.close_position(SYMBOL)
                    client.place_market_order(SYMBOL, "sell", SIZE)
            
            else:
                log_event("⏸️ NEURAL STATUS: STABLE / SIDEWAYS. Holding Cash.", C_YELLOW)
                # If we are in profit or trend flipped, close
                # (Simple strategy logic: exit if sentiment neutralizes)
                if is_in_long or is_in_short:
                    log_event("🔄 Trend neutralised. Exiting current position.", C_YELLOW)
                    client.close_position(SYMBOL)

            # 5. Cycle Management
            time.sleep(60) # High-Frequency Heartbeat

        except Exception as e:
            log_event(f"⚠️ Exception in Pilot Loop: {e}", C_RED)
            time.sleep(10)

if __name__ == "__main__":
    run_pilot()
