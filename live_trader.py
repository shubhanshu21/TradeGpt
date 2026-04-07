"""
SOVEREIGN ALPHA PILOT (V4.2) ⚓🚀
==========================================
- Timeframe: 1 minute (Institutional Scaling)
- Brain: HYDRA Sovereign V4.2 (RoPE + MLA + Aux-MTP)
- Targets: MTP-15 (Predicting Price, Volatility, Volume)
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

# Architecture Imports (Sync with Hydra V4.2)
from architectures.hydra import HydraV4, HydraBlock, GatedMoE, MLAAttention, RMSNorm, SovereignLoss
from delta_client import DeltaClient
from fetch_data    import fetch_live_kat_data
from preprocess    import KATScaler, build_feature_cols, add_derived_features

# ── CONFIG ───────────────────────────────────────────────────────────────────
SYMBOL     = "BTCUSD"
SIZE       = 1           
THRESHOLD  = 0.08        # Base Conviction (Fee-Aware)
TIMEFRAME  = "1m"
MODEL_FILE = "hydra_best.keras" 
CONTEXT_L  = 360         # Matches V4.2 TurboQuant CPU Training context
INPUT_N    = 345         # 360 - 15 target steps

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
    print(f"  SOVEREIGN ALPHA PILOT (V4.2) — TESTNET PILOT [ {SYMBOL} ]")
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
    
    log_event(f"🏗️ Syncing Neural Brain V4.2: {MODEL_FILE}...")
    custom_objs = {
        "HydraV4": HydraV4, "HydraBlock": HydraBlock, "GatedMoE": GatedMoE,
        "MLAAttention": MLAAttention, "RMSNorm": RMSNorm, "SovereignLoss": SovereignLoss,
        "TTMReflex": TTMReflex
    }
    # Compile=False as we only need inference
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
            X_in = scaled[-INPUT_N:].reshape(1, INPUT_N, len(features))
            X_in = X_in.astype("float32")
            
            # 3. Neural Pulse (V4.2 Multi-Target Inference)
            pred = model.predict(X_in, verbose=0) # Shape (1, 15, 3)
            pred = pred[0] # Take first batch (15, 3)
            
            p_curve = pred[:, 0] # 15 Price steps
            v_curve = pred[:, 1] # 15 Volatility steps
            q_curve = pred[:, 2] # 15 Volume steps
            
            mean_move = np.mean(p_curve)
            mean_vol  = np.mean(v_curve)
            
            p_str = " | ".join([f"{x:+.3f}" for x in p_curve[:5]]) # Show first 5 mins
            log_event(f"🔮 PRICE CURVE: [ {p_str}... ] | AVG: {mean_move:+.4f}")
            log_event(f"📊 VOLATILITY: {mean_vol:.4f} | VOLUME FLOW: {np.mean(q_curve):.4f}")

            # 4. Strategy Bridge (V4.2 Regime-Aware Execution)
            pos = client.get_positions()
            # Filter for our symbol
            my_pos = [p for p in pos if client._resolve_product_id(SYMBOL) == int(p['product_id'])]
            
            is_in_long  = any(float(p['size']) > 0 for p in my_pos)
            is_in_short = any(float(p['size']) < 0 for p in my_pos)

            # Volatility Adaptive Threshold (High Vol = Need Higher Conviction)
            # Normalizing mean_vol (assumed around 0-1 range after scaling)
            dynamic_thresh = THRESHOLD * (1.0 + max(0, mean_vol))

            # Signal Logic
            if mean_move > dynamic_thresh:
                if not is_in_long:
                    log_event(f"📈 SIGNAL: BULLISH CONVICTION ({mean_move:.4f} > {dynamic_thresh:.4f}). Executing LONG...", C_GREEN)
                    client.place_order(SYMBOL, SIZE, "buy", sl_pct=1.0, tp_pct=2.5)
            
            elif mean_move < -dynamic_thresh:
                if not is_in_short:
                    log_event(f"📉 SIGNAL: BEARISH CONVICTION ({mean_move:.4f} < {-dynamic_thresh:.4f}). Executing SHORT...", C_RED)
                    client.place_order(SYMBOL, SIZE, "sell", sl_pct=1.0, tp_pct=2.5)
            
            else:
                log_event(f"⏸️ NEURAL STATUS: STABLE / WEAK ({mean_move:.4f} vs {dynamic_thresh:.4f}).", C_YELLOW)

            # 5. Cycle Management
            time.sleep(60) 

        except Exception as e:
            log_event(f"⚠️ Exception in Pilot Loop: {e}", C_RED)
            time.sleep(10)

if __name__ == "__main__":
    run_pilot()
