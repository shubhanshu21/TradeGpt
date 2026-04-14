"""
SOVEREIGN ALPHA PILOT (V10.6 — PREDATOR) ⚓🚀
==============================================
- Timeframe: 1 minute
- Brain: HYDRA V10.6 (SwiGLU + 256-Expert MoE + TurboQuant)
- Targets: MTP-15 (Price, Volatility, Volume Flow)
- Exchange: Delta Exchange (Market-Order + SL/TP Brackets)
"""

import os, sys, time
import numpy as np
import keras
import pandas as pd
from datetime import datetime
from pathlib import Path

# ROOT must point to the project root (/kat/), not the file's own directory
ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

from core.hydra import (HydraBlock, GatedMoE, LightningAttention,
                        RMSNorm, TurboQuant, SwiGLU,
                        SovereignLoss, CertaintyMetric, SovereignAccuracy)
from exchange.delta_client import DeltaClient
from exchange.fetch_data   import fetch_live_kat_data
from data.preprocess       import KATScaler, build_feature_cols, compute_indicators

# ── CONFIG ────────────────────────────────────────────────────────────────────
SYMBOL     = "BTCUSD"
SIZE       = 1              # Contract size
THRESHOLD  = 0.05           # Base conviction (Z-score units on scaled output)
TIMEFRAME  = "1m"
MODEL_FILE = "hydra_best.keras"
CTX_WIN    = 120            # Must match training context window
SLEEP_S    = 60             # Poll every 1 minute

# ── HUD ───────────────────────────────────────────────────────────────────────
C_RESET  = "\033[0m"
C_BOLD   = "\033[1m"
C_GREEN  = "\033[32m"
C_RED    = "\033[31m"
C_CYAN   = "\033[36m"
C_YELLOW = "\033[33m"

def log(msg, color=C_RESET):
    stamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{C_CYAN}{stamp}{C_RESET}] {color}{msg}{C_RESET}", flush=True)

def load_model_and_scaler():
    """Load the trained brain and fitted scaler. Exits if either is missing."""
    scaler_p = ROOT / "models" / "scaler_base.pkl"
    model_p  = ROOT / "models" / MODEL_FILE

    if not scaler_p.exists():
        log("❌ Scaler missing — run training first.", C_RED); sys.exit(1)
    if not model_p.exists():
        log(f"❌ Model not found: {MODEL_FILE}", C_RED); sys.exit(1)

    scaler = KATScaler.load(str(scaler_p))

    log(f"🏗️  Loading brain: {MODEL_FILE}...")
    custom_objs = {
        "HydraBlock":        HydraBlock,
        "GatedMoE":          GatedMoE,
        "LightningAttention": LightningAttention,
        "RMSNorm":           RMSNorm,
        "TurboQuant":        TurboQuant,
        "SwiGLU":            SwiGLU,
        "SovereignLoss":     SovereignLoss,
        "CertaintyMetric":   CertaintyMetric,
        "SovereignAccuracy": SovereignAccuracy,
    }
    model = keras.models.load_model(
        str(model_p), custom_objects=custom_objs, compile=False, safe_mode=False
    )
    log("✅ Brain sync complete.", C_GREEN)
    return model, scaler

def get_neural_signal(model, scaler):
    """
    Fetch latest market data, engineer features, run inference.
    Returns (mean_price_move, mean_volatility, full_pred_array).
    """
    features = build_feature_cols()
    n_feats  = len(features)

    # Fetch CTX_WIN + small buffer for indicator warm-up
    df = fetch_live_kat_data(symbol=SYMBOL, n_candles=CTX_WIN + 150, timeframe=TIMEFRAME)

    # V5.0 feature engineering (compute_indicators replaces add_derived_features)
    df_feat = compute_indicators(df)
    data    = df_feat[features].values.astype("float32")

    # Scale and take the last CTX_WIN rows
    scaled = scaler.transform_X(data)
    X_in   = scaled[-CTX_WIN:].reshape(1, CTX_WIN, n_feats).astype("float32")

    # V10.6 model returns [prediction, consensus, reasoning] — index [0] for price head
    outputs  = model(X_in, training=False)
    pred     = outputs[0].numpy()[0]    # (forecast+1, 3) — drop entry-anchor (index 0)
    pred     = pred[1:]                 # (15, 3) — only future steps
    p_curve  = pred[:, 0]   # 15 price steps
    v_curve  = pred[:, 1]   # 15 volatility steps
    q_curve  = pred[:, 2]   # 15 volume flow steps

    return np.mean(p_curve), np.mean(v_curve), np.mean(q_curve), pred

def run_pilot():
    print("="*60)
    print(f"  ⚓ SOVEREIGN ALPHA PILOT V5.0 — [ {SYMBOL} ] LIVE")
    print("="*60)

    client        = DeltaClient(testnet=True)
    model, scaler = load_model_and_scaler()

    while True:
        try:
            log(f"📡 Polling {SYMBOL} market stream...")

            # ── Inference ────────────────────────────────────────────────────
            mean_price, mean_vol, mean_q, pred = get_neural_signal(model, scaler)

            p_str = " | ".join([f"{x:+.3f}" for x in pred[:5, 0]])
            log(f"🔮 PRICE CURVE : [ {p_str}... ]  AVG: {mean_price:+.4f}")
            log(f"📊 VOLATILITY  : {mean_vol:.4f}  |  VOLUME FLOW: {mean_q:.4f}")

            # ── Dynamic threshold (scale with volatility) ─────────────────────
            dynamic_thresh = THRESHOLD * (1.0 + max(0.0, mean_vol))

            # ── Position check ────────────────────────────────────────────────
            pos         = client.get_positions()
            product_id  = client._resolve_product_id(SYMBOL)
            my_pos      = [p for p in pos if int(p["product_id"]) == product_id]
            is_in_long  = any(float(p["size"]) > 0 for p in my_pos)
            is_in_short = any(float(p["size"]) < 0 for p in my_pos)
            in_position = is_in_long or is_in_short

            # ── Signal Logic ──────────────────────────────────────────────────
            if mean_price > dynamic_thresh:
                if not is_in_long:
                    log(f"📈 LONG  ({mean_price:+.4f} > +{dynamic_thresh:.4f})", C_GREEN)
                    client.place_order(SYMBOL, SIZE, "buy", sl_pct=1.0, tp_pct=2.5)
                else:
                    log(f"✅ Already LONG — holding.", C_GREEN)

            elif mean_price < -dynamic_thresh:
                if not is_in_short:
                    log(f"📉 SHORT ({mean_price:+.4f} < -{dynamic_thresh:.4f})", C_RED)
                    client.place_order(SYMBOL, SIZE, "sell", sl_pct=1.0, tp_pct=2.5)
                else:
                    log(f"✅ Already SHORT — holding.", C_RED)

            else:
                log(f"⏸️  HOLD — weak signal ({mean_price:+.4f}, thresh ±{dynamic_thresh:.4f})", C_YELLOW)

            # ── Cycle ─────────────────────────────────────────────────────────
            log(f"💤 Next poll in {SLEEP_S}s...")
            time.sleep(SLEEP_S)

        except KeyboardInterrupt:
            log("🛑 Pilot stopped by operator.", C_YELLOW)
            break
        except Exception as e:
            log(f"⚠️  Loop error: {e}", C_RED)
            time.sleep(10)

if __name__ == "__main__":
    run_pilot()
