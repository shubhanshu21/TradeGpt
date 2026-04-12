"""
SOVEREIGN KRAKEN TRAINING ORCHESTRATOR (V4.7 — Stable Abyss Stream) ⚓🚀⚡
===========================================================================
- Model: HYDRA V4.5.9 (384-wide, 12-block, 16-expert MoE)
- RAM Strategy: TF Generator streaming — NO materialization, no OOM
- Context: 60 candles (1 hour) — proven stable on this hardware
- Batch: 256 — saturates gradient activation buffers to ~15GB
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
from architectures.hydra import build_kraken, IS_GPU, init_kraken_hardware
from preprocess          import build_dataset_streaming
from fetch_data          import fetch_live_kat_data
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
    Early-Warning Diagnostic System (V10.2: Expert Insight).
    """
    def __init__(self, log_dir: Path):
        super().__init__()
        self.log_path = log_dir / "diagnostics.log"
        log_dir.mkdir(parents=True, exist_ok=True)

    def on_train_begin(self, logs=None):
        with open(self.log_path, "w") as f:
            f.write("Time,Epoch,Val_Dir_Acc,Certainty,Status\n")

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        # V10.3 Dual Output keys
        v_acc = logs.get("val_prediction_dir_acc", 0.0)
        cert  = logs.get("val_certainty_mean", 0.0)
        ts    = datetime.now().strftime("%H:%M:%S")
        
        status = "KEEPING" if v_acc < 0.53 else "🚀 SOVEREIGN EDGE DETECTED"
        
        with open(self.log_path, "a") as f:
            f.write(f"{ts},{epoch+1},{v_acc:.4f},{cert:.4f},{status}\n")
        
        if v_acc > 0.53:
            print(f"\n[🚀 SOVEREIGN EDGE DETECTED] Win-Rate: {v_acc:.2%} | Certainty: {cert:.2%}")
        
        # 2. Print high-visibility alerts
        if epoch > 10 and v_acc < 0.501:
            print(f"\n[⚠️  MISSION CONTROL ALERT] Stagnation detected at Epoch {epoch+1}.")
            print(f"    Current Win-Rate: {val_acc:.4f} (Under Coin-Flip Threshold)")
        
        if val_acc > 0.53:
            print(f"\n[🚀 SOVEREIGN EDGE DETECTED] Win-Rate: {val_acc:.4f}. Entering Profit Zone!")


def train_kraken(args):
    init_kraken_hardware()
    
    # ── V6.5 Sovereign Auto-Log ──────────────────────────────────────────────
    # Recreate logs directory if missing
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    
    # Auto-redirect output to file if not interactive
    log_file_path = LOG_DIR / "omni_brain_300.log"
    print(f"📡 Internal Logging Engaged: {log_file_path}")
    
    # We use a custom logger to write to both console and file if possible
    class Logger(object):
        def __init__(self, filename):
            self.terminal = sys.stdout
            self.log = open(filename, "a", buffering=1)
        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)
        def flush(self):
            self.terminal.flush()
            self.log.flush()

    sys.stdout = Logger(log_file_path)
    sys.stderr = sys.stdout

    print("\n" + "="*60)
    print(f"  {'🚀 GPU MODE' if IS_GPU else '🐌 CPU MODE'} — SOVEREIGN KRAKEN V4.7")
    print("="*60)

    BATCH_S  = args.batch          # 128 — calibrated for CTX=120 (~14GB)
    EPOCHS   = args.epochs
    CANDLES  = args.candles        # 120000
    CTX_WIN  = 120                 # 2-hour context: max safe on 24GB CPU host
    FORECAST = 15                  # predict next 15 close prices

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

    # ── 3. Build Model (Singularity Tier: 2e-4 Precision) ────────────────────
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
            save_freq=epoch_ckpt_freq, verbose=0),
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=30, restore_best_weights=True),
        # CosineDecay is baked into the optimizer — no ReduceLROnPlateau needed.
        # Pruner auto-deletes old checkpoints, keeping last 3 + best.
        CheckpointPruner(ckpt_dir=CKPT_DIR, keep_n=3),
        MissionControl(log_dir=LOG_DIR),  # Dynamic path
    ]

    # ── 6. Ignite ─────────────────────────────────────────────────────────────
    print(f"\n🚀 IGNITION: {EPOCHS}-Epoch Mission | Batch {BATCH_S} | CTX {CTX_WIN} candles (2h) | ~14GB RAM Target")
    
    current_epoch = 0
    if args.resume and saved:
        try:
            current_epoch = int(os.path.basename(saved[-1]).split("_E")[-1].split(".")[0])
        except: pass

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
    p.add_argument("--timeframe", default="1m")
    p.add_argument("--epochs",    type=int, default=300)
    p.add_argument("--model",     default="hydra")
    p.add_argument("--batch",     type=int, default=128)
    p.add_argument("--candles",   type=int, default=120000)
    p.add_argument("--resume",    action="store_true")
    args = p.parse_args()
    train_kraken(args)