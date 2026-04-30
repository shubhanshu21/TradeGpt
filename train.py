"""
SOVEREIGN KRAKEN TRAINING ORCHESTRATOR (V10.6 — Predator) ⚓🚀⚡
===========================================================================
- Model: HYDRA V10.6 (128-wide, 8-block, 256-Expert MoE + SwiGLU + TurboQuant)
- RAM Strategy: TF Generator streaming — NO materialization, no OOM
- Context: 120 candles (2 hours) — proven stable on this hardware
- Batch: 128 — calibrated for 21GB RAM host
"""

import os, argparse, gc, glob, time
from datetime import datetime
import numpy as np
import tensorflow as tf
import keras
import pandas as pd
from pathlib import Path

ROOT     = Path(__file__).parent
DATA_DIR = ROOT / "data"
CKPT_DIR = ROOT / "models"
LOG_DIR  = ROOT / "logs"

import sys
sys.path.insert(0, str(ROOT / "src"))
from core.hydra import build_kraken, IS_GPU, init_kraken_hardware, CertaintyMetric, SovereignAccuracy, SovereignLoss
from data.preprocess import build_dataset_streaming, build_feature_cols, KATScaler
from exchange.fetch_data import fetch_live_kat_data
import glob as _glob


class CheckpointPruner(keras.callbacks.Callback):
    """
    Enhancement #5 — Auto-prune old epoch checkpoints.
    Keeps: best val_loss checkpoint + last N epoch checkpoints.
    Prevents disk fill during 300-epoch runs.
    """
    def __init__(self, ckpt_dir: Path, keep_n: int = 3):
        super().__init__()
        self.ckpt_dir = ckpt_dir
        self.keep_n   = keep_n

    def on_epoch_end(self, epoch, logs=None):
        pattern  = str(self.ckpt_dir / "hydra_checkpoint_E*.keras")
        all_ckpt = sorted(_glob.glob(pattern))
        # Keep only the last keep_n; delete the rest
        to_delete = all_ckpt[: max(0, len(all_ckpt) - self.keep_n)]
        for f in to_delete:
            try:
                os.remove(f)
            except OSError:
                pass

class MissionControl(keras.callbacks.Callback):
    """
    Early-Warning Diagnostic System (V11.0: Real-time console reports).
    """
    def on_train_begin(self, logs=None):
        print("\n" + "="*50)
        print(f"{'Time':<10} | {'Epoch':<5} | {'Val_Acc':<8} | {'Certainty':<10} | {'Status'}")
        print("-" * 50)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        v_acc = logs.get("val_prediction_dir_acc", 0.0)
        cert  = logs.get("val_certainty_certainty", 0.0)
        ts    = datetime.now().strftime("%H:%M:%S")
        
        status = "⚓ LEARNING"
        if v_acc >= 0.54: status = "🏛️ SOVEREIGN"
        elif v_acc >= 0.53: status = "⚡ ALPHA FLOW"
        
        print(f"{ts:<10} | {epoch+1:<5} | {v_acc:<8.4f} | {cert:<10.3f} | {status}")
        
        # ── Automated Sovereign Benchmarking (V11.1) ────────────────────────
        # This runs every epoch to update the dashboard ROI card automatically
        try:
            import json
            from data.preprocess import compute_indicators, build_feature_cols
            from exchange.fetch_data import fetch_live_kat_data
            
            # 1. Fetch live slice for bench (Last 500 candles)
            df_raw = fetch_live_kat_data('BTCUSD', 500, '15m')
            if df_raw is not None:
                df = compute_indicators(df_raw)
                features = build_feature_cols()
                data = df[features].values.astype('float32')
                raw_prices = df['close'].values
                
                ctx = 120; f = 15
                # Slice indices for a quick backtest
                indices = range(len(df) - ctx - f - 100, len(df) - ctx - f)
                X = np.array([(data[i:i+ctx] - data[i:i+ctx].mean(0)) / (data[i:i+ctx].std(0) + 1e-8) for i in indices])
                usd_diffs = np.array([raw_prices[i + ctx + f - 1] - raw_prices[i + ctx - 1] for i in indices])
                
                # Inference using CURRENT weights
                outputs = self.model(X, training=False)
                traj    = outputs[0].numpy()[:, -1, 0] # Price at end of forecast
                certs   = np.mean(outputs[1].numpy(), axis=1) # Avg certainty
                
                # Normalize certainty for bench (80-100% range)
                c_pct = (certs - certs.min()) / (certs.max() - certs.min() + 1e-9) * 100
                
                roi_data = {"tiers": {}, "last_update": ts}
                pos_size = 2000.0; fee_rate = 0.0006
                for th in [80, 85, 90]:
                    mask = c_pct >= th
                    n_t  = int(mask.sum())
                    if n_t > 0:
                        e_p = raw_prices[np.array(indices)[mask] + ctx - 1]
                        gross = float((np.sign(traj[mask]) * (usd_diffs[mask] / e_p) * pos_size).sum())
                        fees = float(n_t * (pos_size * fee_rate))
                        roi_data["tiers"][str(th)] = {"trades": n_t, "net": gross - fees}
                
                # 3. Save detailed recent trades for dashboard feed
                recent_trades = []
                # Look at the last 10 trades that passed the 80% threshold
                mask80 = c_pct >= 80
                if mask80.any():
                    t_indices = np.array(indices)[mask80]
                    t_traj = traj[mask80]
                    t_usd = usd_diffs[mask80]
                    
                    # Take the last 10
                    for i in range(max(0, len(t_indices)-10), len(t_indices)):
                        idx = t_indices[i]
                        entry_p = raw_prices[idx + ctx - 1]
                        price_move_pct = (t_usd[i] / entry_p)
                        side = "LONG" if t_traj[i] > 0 else "SHORT"
                        
                        # Net profit calculation
                        raw_ret = price_move_pct if side == "LONG" else -price_move_pct
                        net_ret = raw_ret - fee_rate
                        
                        recent_trades.append({
                            "timestamp": pd.to_datetime(df.index[idx + ctx - 1]).strftime("%H:%M"),
                            "side": side,
                            "entry": float(entry_p),
                            "net_pct": float(net_ret * 100)
                        })
                
                with open(ROOT / "logs" / "latest_roi.json", "w") as f_json:
                    json.dump(roi_data, f_json, indent=4)
                    
                with open(ROOT / "logs" / "recent_sim_trades.json", "w") as f_trades:
                    json.dump(recent_trades[::-1], f_trades, indent=4) # Newest first
        except Exception as e:
            pass # Bench failed, don't crash the training run


def train_kraken(args):
    # Recreate logs directory if missing
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    
    # ── 0. Hardware Prep ──────────────────────────────────────────────────────
    init_kraken_hardware()

    print("\n" + "="*60)
    print(f"  {'🚀 GPU MODE' if IS_GPU else '🐌 CPU MODE'} — SOVEREIGN KRAKEN V4.7")
    print("="*60)

    BATCH_S  = args.batch          # 64 — calibrated for 15m resolution
    EPOCHS   = args.epochs
    CANDLES  = args.candles        # 120000 15m candles = ~3.4 years
    CTX_WIN  = 120                 # 30-hour context (120 × 15m) — stable macro patterns
    FORECAST = 15                  # Predict next 3.75 hours (15 × 15m)

    # ── 1. Fetch / Cache ──────────────────────────────────────────────────────
    DATA_DIR.mkdir(exist_ok=True)
    CKPT_DIR.mkdir(exist_ok=True)
    LOG_DIR.mkdir(exist_ok=True)
    CACHE_P = DATA_DIR / f"{args.symbol}_{args.timeframe}_history_{CANDLES}.parquet"

    if CACHE_P.exists():
        print(f"📖 CACHE: Loading {CANDLES:,} candles...")
        df = pd.read_parquet(CACHE_P).tail(CANDLES)
    else:
        print(f"📡 Fetching {CANDLES:,} live candles...")
        df = fetch_live_kat_data(symbol=args.symbol, n_candles=CANDLES, timeframe=args.timeframe)
        if df is not None and len(df) > 0:
            df.to_parquet(CACHE_P)

    if df is None or len(df) == 0:
        print("❌ No data. Aborting."); return

    # ── 2. Streaming Dataset (20GB RAM Optimized) ────────────────────────────
    # High Batch + High Shuffle = High RAM utilization and better alpha
    ds_info = build_dataset_streaming(df, context_window=CTX_WIN, forecast_steps=FORECAST,
                                       batch_size=BATCH_S,
                                       scaler_save_path=str(CKPT_DIR / "scaler_base.pkl"))
    tr_ds   = ds_info["tr_ds"]
    va_ds   = ds_info["va_ds"]
    steps_tr = ds_info["steps_tr"]
    steps_va = ds_info["steps_va"]
    n_feat   = ds_info["n_features"]

    print(f"   ✅ {steps_tr} train steps/epoch | {steps_va} val steps")

    # ── 2b. Compute Reasoning Class Weights (Anti-Imbalance) ─────────────────
    # Sideways candles dominate 1m BTC data (~70%). Without class weights,
    # the model learns to predict "Sideways" for everything.
    print("   📊 Computing reasoning class weights (anti-imbalance)...")
    label_counts = np.zeros(4)
    # Mirror the 80% train split used inside build_dataset_streaming
    train_end    = int(len(df) * 0.8)
    sample_limit = min(train_end, 5000)
    raw_data  = df.iloc[:sample_limit]
    ret_col   = (raw_data["close"].pct_change(1).fillna(0)).values  # 15m candle impact
    for r in ret_col:
        if   r >  0.01: label_counts[0] += 1   # Bull
        elif r < -0.01: label_counts[1] += 1   # Bear
        elif abs(r) < 0.002: label_counts[2] += 1  # Sideways
        else: label_counts[3] += 1             # Trend
    label_counts = np.maximum(label_counts, 1)
    total = label_counts.sum()
    class_weights = {i: total / (4 * label_counts[i]) for i in range(4)}
    print(f"   ⚖️  Class weights: Bull={class_weights[0]:.2f} Bear={class_weights[1]:.2f} "
          f"Sideways={class_weights[2]:.2f} Trend={class_weights[3]:.2f}")

    # ── 3. Build Model ────────────────────────────────────────────────────────
    # For 1,152 experts, we use a slower, higher-quality learning profile.
    model = build_kraken(n_features=n_feat, context_window=CTX_WIN, 
                        forecast_steps=FORECAST)
    
    # We allow build_kraken to handle the optimizer initialization

    # ── 4. Load Weights ───────────────────────────────────────────────────────
    CKPT_BEST = CKPT_DIR / "hydra_best.keras"
    saved     = sorted(glob.glob(str(CKPT_DIR / "hydra_checkpoint_E*.keras")))

    if args.resume and saved:
        print(f"📦 Loading weights from {os.path.basename(saved[-1])}")
        model.load_weights(saved[-1])
    elif args.resume and CKPT_BEST.exists():
        print(f"📦 Loading weights from hydra_best.keras")
        model.load_weights(str(CKPT_BEST))

    # ── 5. Callbacks ──────────────────────────────────────────────────────────
    epoch_ckpt_freq = 10 * steps_tr   # save every 10 epochs
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            str(CKPT_BEST), monitor="val_loss",
            save_best_only=True, verbose=1),
        keras.callbacks.ModelCheckpoint(
            str(CKPT_DIR / "hydra_checkpoint_E{epoch:03d}.keras"),
            save_freq="epoch", verbose=0),  # Save every epoch for resume support
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=7,  # 7 epochs * 12.5h = ~3.6 days max wait
            restore_best_weights=True, verbose=1),
        CheckpointPruner(ckpt_dir=CKPT_DIR, keep_n=3),
        MissionControl(),
    ]

    # ── 6. Ignite ─────────────────────────────────────────────────────────────
    print(f"\n🚀 IGNITION: {EPOCHS}-Epoch Mission | Batch {BATCH_S} | CTX {CTX_WIN} candles (2h) | ~14GB RAM Target")
    
    # V10.4: Smart Epoch Detection from Checkpoint Filenames
    current_epoch = 0
    if args.resume and saved:
        try:
            # Parse 'E004' from 'hydra_checkpoint_E004.keras'
            latest_file = os.path.basename(saved[-1])
            import re
            match = re.search(r"E(\d+)", latest_file)
            if match:
                current_epoch = int(match.group(1)) # This is the FINISHED epoch
                print(f"   🎯 RESUMPTION: Detected finished epoch {current_epoch}. Starting Epoch {current_epoch+1}...")
        except Exception as e:
            print(f"   ⚠️ Could not parse epoch from filename: {e}")
            current_epoch = len(saved)
    
    # Standard Keras model.fit resumption

    model.fit(
        tr_ds,
        validation_data=va_ds,
        epochs=EPOCHS,
        initial_epoch=current_epoch,      # FIX: show correct epoch in logs
        steps_per_epoch=steps_tr,
        validation_steps=steps_va,
        callbacks=callbacks,
        verbose=1
    )

    model.save(str(CKPT_DIR / "hydra_final.keras"))
    print("\n✅ MISSION COMPLETE — Sovereign Alpha-Brain saved.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--symbol",    default="BTCUSD")
    p.add_argument("--timeframe", default="15m")   # 15m: maximum SNR for swing trading
    p.add_argument("--epochs",    type=int, default=300)
    p.add_argument("--model",     default="hydra")
    p.add_argument("--batch",     type=int, default=64)
    p.add_argument("--candles",   type=int, default=120000)
    p.add_argument("--resume",    action="store_true")
    args = p.parse_args()
    train_kraken(args)