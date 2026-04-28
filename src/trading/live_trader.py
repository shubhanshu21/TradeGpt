"""
SOVEREIGN ALPHA PILOT (V11.0 — IRON ORACLE) ⚓🏛️
==================================================
- Timeframe: 15 minutes (Phase 5 — Maximum SNR)
- Brain: IRON ORACLE V11.0 (256-Expert MoE + MLA + RoPE)
- Targets: MTP-15 (Price, Volatility, Volume Flow)
- Exchange: Delta Exchange (Market-Order + SL/TP Brackets)
- Filter: Certainty threshold (80%+ = Sovereign Edge)
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
from data.preprocess       import build_feature_cols, compute_indicators

# ── CONFIG ────────────────────────────────────────────────────────────────────
SYMBOL         = "BTCUSD"
SIZE           = 1              # Contract size
THRESHOLD      = 0.05           # Base conviction (Z-score units on scaled output)
TIMEFRAME      = "15m"          # FIX #2: Must match training timeframe (15m)
MODEL_FILE     = "hydra_best.keras"
CTX_WIN        = 120            # Must match training context window (30 hours)
SLEEP_S        = 900            # Poll every 15 minutes (one 15m candle)
CERT_THRESHOLD = 0.80           # FIX #3: Default certainty filter (80% = Sovereign Edge)

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

def load_model():
    """Load the trained Iron Oracle brain. Exits if model is missing."""
    model_p = ROOT / "models" / MODEL_FILE

    if not model_p.exists():
        log(f"❌ Model not found: {MODEL_FILE}", C_RED); sys.exit(1)

    # FIX #1: Iron Oracle uses DLS (Dynamic Local Scaling) — no global scaler needed.
    # We use build_kraken to reconstruct the architecture and load weights directly.
    from core.hydra import build_kraken
    n_feat = 42  # Phase 5 Sovereign Hive (42 features)
    log(f"🏗️  Re-building Iron Oracle V11.0 ({n_feat} features)...")
    model = build_kraken(n_features=n_feat, context_window=CTX_WIN)
    model.load_weights(str(model_p))
    log("✅ Brain sync complete.", C_GREEN)
    return model

def get_neural_signal(model):
    """
    Fetch latest 15m market data, engineer features, run inference.
    FIX #1: Uses Dynamic Local Scaling (DLS) — matching training pipeline.
    Returns (mean_price_move, certainty_pct, mean_volatility, full_pred_array).
    """
    features = build_feature_cols()
    n_feats  = len(features)  # 42

    # Fetch CTX_WIN + buffer for indicator warm-up
    df = fetch_live_kat_data(symbol=SYMBOL, n_candles=CTX_WIN + 150, timeframe=TIMEFRAME)
    df_feat = compute_indicators(df)
    data    = df_feat[features].values.astype("float32")

    # FIX #1: DLS — scale using the local window stats (same as training)
    x_raw  = data[-CTX_WIN:]
    l_mean = x_raw.mean(axis=0)
    l_std  = x_raw.std(axis=0) + 1e-8
    x_scaled = (x_raw - l_mean) / l_std
    X_in   = x_scaled[np.newaxis].astype("float32")  # (1, 120, 42)

    # Iron Oracle returns [prediction_trajectory, certainty_map]
    outputs      = model(X_in, training=False)
    pred         = outputs[0].numpy()[0]   # (16, 3)
    certainty_2d = outputs[1].numpy()[0]   # (120,) per-step certainty

    pred_future  = pred[1:]                # (15, 3) — future steps only
    p_curve      = pred_future[:, 0]       # price trajectory
    v_curve      = pred_future[:, 1]       # volatility
    q_curve      = pred_future[:, 2]       # volume flow

    # Normalize certainty to 0–100%
    cert_mean    = float(np.mean(certainty_2d))
    # cert_pct is relative — we use the raw value for threshold comparison
    cert_pct     = cert_mean  # compared against CERT_THRESHOLD in run_pilot

    return np.mean(p_curve), cert_pct, np.mean(v_curve), np.mean(q_curve), pred_future

def run_pilot():
    print("="*60)
    print(f"  ⚓ IRON ORACLE V11.0 — [ {SYMBOL} ] LIVE PILOT")
    print(f"  📡 Timeframe : {TIMEFRAME}")
    print(f"  🎯 Certainty : {CERT_THRESHOLD*100:.0f}%+ required to trade")
    print(f"  💰 Position  : {SIZE} contract(s)")
    print("="*60)

    client = DeltaClient(testnet=True)
    model  = load_model()  # FIX #1: DLS — no scaler needed

    while True:
        try:
            log(f"📡 Polling {SYMBOL} [{TIMEFRAME}] market stream...")

            # ── Inference ────────────────────────────────────────────────────
            mean_price, cert_raw, mean_vol, mean_q, pred = get_neural_signal(model)

            # Normalize certainty to 0–1 range for threshold comparison
            # We use a simple sigmoid-style normalization against a known baseline
            cert_norm = min(1.0, max(0.0, (cert_raw - 100.0) / 30.0 + 0.5))

            p_str = " | ".join([f"{x:+.3f}" for x in pred[:5, 0]])
            log(f"🔮 PRICE CURVE  : [ {p_str}... ]  AVG: {mean_price:+.4f}")
            log(f"🧠 CERTAINTY    : {cert_norm*100:.1f}%  (threshold: {CERT_THRESHOLD*100:.0f}%)")
            log(f"📊 VOLATILITY   : {mean_vol:.4f}  |  VOLUME FLOW: {mean_q:.4f}")

            # ── FIX #3: Certainty Gate — only trade high-conviction signals ──
            if cert_norm < CERT_THRESHOLD:
                log(f"🔕 CERTAINTY TOO LOW ({cert_norm*100:.1f}% < {CERT_THRESHOLD*100:.0f}%) — HOLDING CASH", C_YELLOW)
                log(f"💤 Next poll in {SLEEP_S}s...")
                time.sleep(SLEEP_S)
                continue

            # ── Dynamic threshold (scale with volatility) ─────────────────────
            dynamic_thresh = THRESHOLD * (1.0 + max(0.0, mean_vol))

            # ── Position check ────────────────────────────────────────────────
            pos        = client.get_positions()
            product_id = client._resolve_product_id(SYMBOL)
            my_pos     = [p for p in pos if int(p["product_id"]) == product_id]
            is_in_long  = any(float(p["size"]) > 0 for p in my_pos)
            is_in_short = any(float(p["size"]) < 0 for p in my_pos)

            # ── Signal Logic ──────────────────────────────────────────────────
            if mean_price > dynamic_thresh:
                if not is_in_long:
                    log(f"📈 LONG  signal ({mean_price:+.4f} > +{dynamic_thresh:.4f}) @ {cert_norm*100:.1f}% certainty", C_GREEN)
                    client.place_order(SYMBOL, SIZE, "buy", sl_pct=1.0, tp_pct=2.5)
                else:
                    log(f"✅ Already LONG — holding.", C_GREEN)

            elif mean_price < -dynamic_thresh:
                if not is_in_short:
                    log(f"📉 SHORT signal ({mean_price:+.4f} < -{dynamic_thresh:.4f}) @ {cert_norm*100:.1f}% certainty", C_RED)
                    client.place_order(SYMBOL, SIZE, "sell", sl_pct=1.0, tp_pct=2.5)
                else:
                    log(f"✅ Already SHORT — holding.", C_RED)

            else:
                log(f"⏸️  HOLD — weak signal ({mean_price:+.4f}, thresh ±{dynamic_thresh:.4f})", C_YELLOW)

            # ── Cycle ─────────────────────────────────────────────────────────
            log(f"💤 Next poll in {SLEEP_S}s (one 15m candle)...")
            time.sleep(SLEEP_S)

        except KeyboardInterrupt:
            log("🛑 Pilot stopped by operator.", C_YELLOW)
            break
        except Exception as e:
            log(f"⚠️  Loop error: {e}", C_RED)
            time.sleep(30)

if __name__ == "__main__":
    run_pilot()
