"""
SOVEREIGN KRAKEN — Walk-Forward Fine-Tuning Simulator ⚓🔬
=========================================================
Simulates live production by iteratively predicting on out-of-sample data chunks,
and then fine-tuning the model on those chunks as if a trading day had concluded.

Usage:
    python scripts/wfa_backtest.py
"""

import sys, os, shutil, time
os.environ["PYTHONUNBUFFERED"] = "1"
import numpy as np
import pandas as pd
import tensorflow as tf
import keras
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from core.hydra import build_kraken, SovereignLoss, CertaintyMetric, SovereignAccuracy, certainty_loss, init_kraken_hardware
from data.preprocess import build_feature_cols, compute_indicators, apply_dls, build_dataset_streaming
from exchange.fetch_data import fetch_live_kat_data
from config.sovereign_config import FEE_RATE, CONTEXT_WINDOW, FORECAST_STEPS

# ── CONFIG ────────────────────────────────────────────────────────────────────
SYMBOL       = "BTCUSD"
TIMEFRAME    = "15m"
CTX_WIN      = CONTEXT_WINDOW
TEST_CANDLES = 2880       # Approx 30 days of 15m candles
CHUNK_SIZE   = 96         # 1 day chunk for fine-tuning
FT_EPOCHS    = 3          # Epochs per fine-tune step
LR           = 1e-6       # Fine-tuning learning rate
THRESHOLD    = 0.15       # Trade trigger Z-score
FEE_PCT      = FEE_RATE   # 0.0012
FREEZE_BELOW = 6          # Freeze bottom blocks during fine-tuning

init_kraken_hardware()

# ── 1. Setup Sandbox Model ────────────────────────────────────────────────────
MODELS_DIR  = ROOT / "models"
checkpoints = sorted(MODELS_DIR.glob("hydra_checkpoint_E*.keras"), reverse=True)
SRC_MODEL_PATH = checkpoints[0] if checkpoints else MODELS_DIR / "hydra_best.keras"
WFA_MODEL_PATH = MODELS_DIR / "wfa_sandbox.keras"

if not SRC_MODEL_PATH.exists():
    print(f"❌ Base model {SRC_MODEL_PATH.name} not found. Train first."); sys.exit(1)

print(f"\n⚓ WALKFOWARD FINETUNING SIMULATOR V1.0")
print("="*60)
print(f"📦 Cloning {SRC_MODEL_PATH.name} -> {WFA_MODEL_PATH.name}")
shutil.copy2(SRC_MODEL_PATH, WFA_MODEL_PATH)

features = build_feature_cols()
n_feats  = len(features)

print("🏗️  Building Kraken Architecture...")
model = build_kraken(n_features=n_feats, context_window=CTX_WIN, forecast_steps=FORECAST_STEPS)
model.load_weights(str(WFA_MODEL_PATH))

# Freeze blocks for fine-tuning
frozen = 0
for layer in model.layers:
    parts = layer.name.split("_")
    if parts[0] == "hydra" and len(parts) == 2 and parts[1].isdigit():
        idx = int(parts[1])
        if idx < FREEZE_BELOW:
            layer.trainable = False
            frozen += 1

print(f"🔒 Frozen {frozen} core blocks for fine-tuning stability.")

# ── 2. Fetch Unseen Data ──────────────────────────────────────────────────────
fetch_size = TEST_CANDLES + CTX_WIN + FORECAST_STEPS + 50
print(f"📡 Fetching {fetch_size:,} recent candles for walk-forward simulation...")
df_raw = fetch_live_kat_data(symbol=SYMBOL, n_candles=fetch_size, timeframe=TIMEFRAME)

print("🌊 Computing features (DLS)...")
df_feat = compute_indicators(df_raw)
data    = df_feat[features].values.astype("float32")

# ── 3. Iterate Over Chunks ────────────────────────────────────────────────────
n_chunks = TEST_CANDLES // CHUNK_SIZE
start_idx = len(data) - TEST_CANDLES - FORECAST_STEPS
close_col = features.index("close")

results = []

print(f"\n🚀 Initiating Walk-Forward Loop: {n_chunks} days (chunks of {CHUNK_SIZE})...")

for c in range(n_chunks):
    chunk_start = start_idx + c * CHUNK_SIZE
    chunk_end   = chunk_start + CHUNK_SIZE
    
    print(f"\n--- [ DAY {c+1}/{n_chunks} ] ---")
    
    # ── A. Trade (Inference on Unseen Chunk) ──
    chunk_results = []
    for i in range(chunk_start, chunk_end):
        X_in = apply_dls(data[i - CTX_WIN : i])[0].reshape(1, CTX_WIN, n_feats)
        
        out  = model(X_in, training=False)
        pred = out[0].numpy()[0]
        cert = out[1].numpy()[0]
        
        p_anchor = pred[0, 0]
        mean_move = float(np.mean(pred[1:, 0] - p_anchor))
        mean_cert = float(np.mean(cert))
        
        if mean_cert < 0.85: signal = "HOLD"
        elif mean_move > THRESHOLD: signal = "LONG"
        elif mean_move < -THRESHOLD: signal = "SHORT"
        else: signal = "HOLD"
        
        actual_now = float(data[i - 1, close_col])
        actual_mean_future = float(np.mean(data[i : i + 15, close_col]))
        actual_dir = np.sign(actual_mean_future - actual_now)
        
        chunk_results.append({
            "chunk": c,
            "signal": signal,
            "actual_dir": actual_dir
        })
    
    # Calculate chunk stats
    df_c = pd.DataFrame(chunk_results)
    trades_c = df_c[df_c["signal"] != "HOLD"]
    if len(trades_c) > 0:
        trades_c = trades_c.copy()
        trades_c["correct"] = trades_c.apply(lambda r: (1.0 if r["signal"] == "LONG" else -1.0) == r["actual_dir"], axis=1)
        c_win = trades_c["correct"].mean() * 100
        c_pnl = trades_c.apply(lambda r: (1 - FEE_PCT*2) if r["correct"] else (-1 - FEE_PCT*2), axis=1).sum()
    else:
        c_win = 0.0
        c_pnl = 0.0
        
    print(f"📊 Trades: {len(trades_c):<3} | Win: {c_win:>5.1f}% | Net P&L: {c_pnl:+.2f}")
    results.extend(chunk_results)
    
    # ── B. Fine-Tune (Learn from the Chunk) ──
    # We slice df_raw to include enough history to compute the dataset for this chunk
    # To predict the chunk, we needed context window prior.
    # To train on the chunk, we need the chunk + forecast steps.
    slice_start = chunk_start - CTX_WIN - 50 # buffer for indicator computation
    slice_end   = chunk_end + FORECAST_STEPS
    
    df_chunk = df_raw.iloc[slice_start:slice_end]
    
    ds_info = build_dataset_streaming(df_chunk, context_window=CTX_WIN, forecast_steps=FORECAST_STEPS, batch_size=8)
    
    label_counts = np.maximum(ds_info["label_counts"], 1)
    ft_total = label_counts.sum()
    ft_class_weights = {i: ft_total / (4 * label_counts[i]) for i in range(4)}
    ft_weights_tensor = tf.constant([ft_class_weights[i] for i in range(4)], dtype=tf.float32)

    def ft_weighted_reasoning_loss(y_true, y_pred):
        y_true_int = tf.reshape(tf.cast(y_true, tf.int32), [-1])
        y_true_oh  = tf.one_hot(y_true_int, depth=4)
        unweighted = tf.keras.losses.categorical_crossentropy(y_true_oh, y_pred, label_smoothing=0.1)
        return unweighted * tf.gather(ft_weights_tensor, y_true_int)

    model.compile(
        optimizer=keras.optimizers.AdamW(LR, weight_decay=0.01, clipnorm=0.5),
        loss={
            "prediction": SovereignLoss(direction_weight=10.0),
            "certainty": certainty_loss,
            "reasoning": ft_weighted_reasoning_loss
        },
        metrics={"prediction": [SovereignAccuracy()]}
    )
    
    # Fine-tune natively
    if ds_info["steps_tr"] > 0:
        model.fit(ds_info["tr_ds"], epochs=FT_EPOCHS, steps_per_epoch=ds_info["steps_tr"], verbose=0)
        # Update sandbox file
        model.save(str(WFA_MODEL_PATH))

# ── 4. Final Report ───────────────────────────────────────────────────────────
df_r = pd.DataFrame(results)
trades = df_r[df_r["signal"] != "HOLD"].copy()
n_trades = len(trades)

print("\n" + "="*60)
print("📊 WALKFOWARD SIMULATION RESULTS")
print("="*60)

if n_trades > 0:
    trades["pred_dir"] = trades.apply(lambda r: 1.0 if r["signal"] == "LONG" else -1.0, axis=1)
    trades["correct"]  = (trades["pred_dir"] == trades["actual_dir"])
    trades["pnl"]      = trades.apply(lambda r: (1 - FEE_PCT*2) if r["correct"] else (-1 - FEE_PCT*2), axis=1)
    
    win_rate = trades["correct"].mean() * 100
    tot_pnl  = trades["pnl"].sum()
    
    print(f"  Total Days : {n_chunks}")
    print(f"  Total Trades: {n_trades}")
    print(f"  Win Rate   : {win_rate:.1f}%")
    print(f"  Total P&L  : {tot_pnl:+.2f} units")
    print(f"  Return/Trd : {tot_pnl/n_trades*100:+.1f}%")
else:
    print("  No trades executed during the simulation period.")
print("="*60 + "\n")
