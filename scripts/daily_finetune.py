"""
SOVEREIGN KRAKEN — Daily Fine-Tuner (V5.0) ⚓📅
================================================
Adapts the trained brain to the latest market regime each morning.

Run manually:
    python scripts/daily_finetune.py

Or via cron (midnight UTC daily):
    5 0 * * * /root/miniconda3/bin/python /var/www/html/ML/kat/scripts/daily_finetune.py >> /var/www/html/ML/kat/logs/finetune.log 2>&1
"""

import sys, shutil, time
import numpy as np
import tensorflow as tf
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

import keras
from core.hydra import (build_kraken, HydraBlock, GatedMoE, MLALayer,
                                  RMSNorm, TurboQuant, SwiGLU,
                                  SovereignLoss, CertaintyMetric, SovereignAccuracy,
                                  SovereignReasoningLoss, certainty_loss,
                                  init_kraken_hardware)
from data.preprocess import build_dataset_streaming
from exchange.fetch_data  import fetch_live_kat_data

# Initialize hardware/thread limits immediately
init_kraken_hardware()

# ── DEFAULTS ──────────────────────────────────────────────────────────────────
DAYS         = 2        # Days of recent data to fine-tune on (~2,880 candles)
EPOCHS       = 5        # Fine-tune epochs — keep small (3–10)
LR           = 1e-6     # Very low LR — nudge, don't overwrite base knowledge
CTX_WIN      = 120      # Must match training context window
BATCH        = 8        # Capped for dynamic performance safety
FREEZE_BELOW = 6        # Freeze foundational blocks (0-5), adapt top blocks (6-7)
MODEL_FILE   = "sandbox_active.keras"
SYMBOL       = "BTCUSD"
TIMEFRAME    = "15m"   # Switched to 15m for maximum SNR (Phase 5+)
KEEP_BACKUPS = 7        # Days of backups to retain

# ── Paths ─────────────────────────────────────────────────────────────────────
MODEL_PATH = ROOT / "models" / MODEL_FILE
BACKUP_DIR = ROOT / "models" / "backups"
CANDLES    = DAYS * 24 * 60 + 200   # +200 for indicator warm-up

# ── Logging ───────────────────────────────────────────────────────────────────
def log(msg):
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    print(f"[{ts}] {msg}", flush=True)

# ── Banner ────────────────────────────────────────────────────────────────────
log("=" * 55)
log("⚓ SOVEREIGN DAILY FINE-TUNER V5.0")
log(f"   Data    : last {DAYS} days ({CANDLES:,} candles)")
log(f"   Epochs  : {EPOCHS}  |  LR: {LR}  |  Batch: {BATCH}")
log(f"   Frozen  : bottom {FREEZE_BELOW} blocks  |  CTX: {CTX_WIN}")
log("=" * 55)

# ── 1. Validate model ─────────────────────────────────────────────────────────
if not MODEL_PATH.exists():
    log(f"❌ Model not found: {MODEL_PATH}")
    log("   Run: cp models/hydra_best.keras models/sandbox_active.keras first.")
    sys.exit(1)

# ── 2. Backup yesterday's model ───────────────────────────────────────────────
BACKUP_DIR.mkdir(exist_ok=True)
date_str    = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M")
backup_path = BACKUP_DIR / f"sandbox_active_{date_str}.keras"
shutil.copy2(MODEL_PATH, backup_path)
log(f"💾 Backup: {backup_path.name}")

# Prune old backups — keep last KEEP_BACKUPS
all_backups = sorted(BACKUP_DIR.glob("sandbox_active_*.keras"))
for old in all_backups[:-KEEP_BACKUPS]:
    old.unlink()
    log(f"🗑️  Pruned: {old.name}")

# ── 3. High-level Safeguard Training Pipeline ──────────────────────────────────
try:
    # ── RAM OOM Safeguard: Terminate base training process temporarily ─────────
    import os, subprocess
    try:
        # Find train.py processes
        pids_raw = subprocess.check_output(["pgrep", "-f", "train.py"]).decode().strip().split()
        pids = [int(p) for p in pids_raw if p.isdigit()]
        for pid in pids:
            if pid != os.getpid():
                log(f"🛑 RAM SAFEGUARD: Found running base training process (PID {pid}).")
                log("   Sending SIGTERM for clean shutdown (saves checkpoint)...")
                subprocess.run(["kill", "-15", str(pid)])  # FIX: SIGTERM first (safe)
                time.sleep(5)  # Give it 5s to finish current batch
                # SIGKILL only if still alive
                try:
                    os.kill(pid, 0)  # Check if still running
                    subprocess.run(["kill", "-9", str(pid)])
                    log("   Process did not exit cleanly — SIGKILL sent.")
                except ProcessLookupError:
                    log("   ✅ Process exited cleanly after SIGTERM.")
    except Exception:
        # No train.py process running or pgrep failed, safe to proceed
        pass

    log(f"📡 Fetching {CANDLES:,} fresh candles...")
    try:
        df = fetch_live_kat_data(symbol=SYMBOL, n_candles=CANDLES, timeframe=TIMEFRAME)
        log(f"   ✅ {len(df):,} candles received")
    except Exception as e:
        log(f"❌ Fetch failed: {e}")
        import traceback
        log(traceback.format_exc())
        sys.exit(1)

    if len(df) < CTX_WIN + 50:
        log(f"❌ Insufficient data: {len(df)} candles")
        sys.exit(1)

    # ── 4. Streaming dataset (DLS — no global scaler needed) ──────────────────────
    log("🌊 Building fine-tune stream (DLS — Dynamic Local Scaling)...")
    ds_info  = build_dataset_streaming(df, context_window=CTX_WIN, forecast_steps=15,
                                        batch_size=BATCH)
    tr_ds    = ds_info["tr_ds"]
    va_ds    = ds_info["va_ds"]
    steps_tr = ds_info["steps_tr"]
    steps_va = ds_info["steps_va"]
    log(f"   ✅ {steps_tr} train steps | {steps_va} val steps")

    # Compute class weights from the fine-tune window's actual label distribution
    label_counts = np.maximum(ds_info["label_counts"], 1)
    ft_total = label_counts.sum()
    ft_class_weights = {i: ft_total / (4 * label_counts[i]) for i in range(4)}
    log(f"   ⚖️  Fine-tune class weights: Bull={ft_class_weights[0]:.2f} Bear={ft_class_weights[1]:.2f} "
        f"FeeTrap={ft_class_weights[2]:.2f} Noise={ft_class_weights[3]:.2f}")

    # V10.6 Phase 3: Re-build from scratch to avoid deserialization issues
    log("🏗️  Re-building Phase 3 architecture and loading weights...")
    from data.preprocess import build_feature_cols
    _features = build_feature_cols()
    _n_feat   = len(_features)
    log(f"   Dynamic feature count: {_n_feat}")
    model = build_kraken(n_features=_n_feat, context_window=CTX_WIN, forecast_steps=15)
    try:
        model.load_weights(str(MODEL_PATH))
        log(f"✅ Weights loaded from {MODEL_FILE}")
    except Exception as e:
        log(f"⚠️  Weight load warning (shape mismatch if upgrading phase): {e}")
        log("   Starting from scratch for this fine-tune session.")

    # ── 7. Freeze bottom blocks ───────────────────────────────────────────────────
    # HydraBlocks are named "hydra_0" through "hydra_7" (from build_kraken name=f"hydra_{i}")
    frozen = 0
    for layer in model.layers:
        parts = layer.name.split("_")
        if parts[0] == "hydra" and len(parts) == 2 and parts[1].isdigit():
            idx = int(parts[1])
            if idx < FREEZE_BELOW:
                layer.trainable = False
                frozen += 1

    trainable = sum(w.numpy().size for w in model.trainable_weights)
    total     = sum(w.numpy().size for w in model.weights)
    log(f"🔒 Frozen {frozen} blocks → {trainable:,} / {total:,} params active")

    # ── 8. Recompile at low LR ────────────────────────────────────────────────────
    ft_weights_tensor = tf.constant([ft_class_weights[i] for i in range(4)], dtype=tf.float32)

    def ft_weighted_reasoning_loss(y_true, y_pred):
        y_true_int    = tf.reshape(tf.cast(y_true, tf.int32), [-1])
        y_true_oh     = tf.one_hot(y_true_int, depth=4)
        unweighted    = tf.keras.losses.categorical_crossentropy(y_true_oh, y_pred, label_smoothing=0.1)
        sample_weights = tf.gather(ft_weights_tensor, y_true_int)
        return unweighted * sample_weights

    model.compile(
        optimizer=keras.optimizers.AdamW(LR, weight_decay=0.01, clipnorm=0.5),
        loss={
            "prediction": SovereignLoss(direction_weight=10.0),
            "certainty":  certainty_loss,
            "reasoning":  ft_weighted_reasoning_loss
        },
        metrics={
            "prediction": [SovereignAccuracy(name="dir_acc"), "mae"],
            "certainty":  [CertaintyMetric(name="certainty")]
        }
    )

    # ── 9. Fine-tune ──────────────────────────────────────────────────────────────
    log(f"\n Fine-tuning: {EPOCHS} epochs on latest {DAYS}-day data...")
    history = model.fit(tr_ds, validation_data=va_ds,
                        epochs=EPOCHS, steps_per_epoch=steps_tr,
                        validation_steps=steps_va, verbose=1)

    initial_loss = history.history["loss"][0]
    final_loss   = history.history["loss"][-1]
    val_loss     = history.history["val_loss"][-1]
    delta        = initial_loss - final_loss

    # ── 10. Save or rollback ──────────────────────────────────────────────────────
    log(f"\n📊 Results: loss {initial_loss:.4f} → {final_loss:.4f} (Δ{delta:+.4f}) | val: {val_loss:.4f}")

    if delta > 0:
        model.save(str(MODEL_PATH))
        log(f"✅ Sandbox model updated — improved by {delta:.4f}")
        
        # --- CLEANUP: Clear used L5 snapshots to save space ---
        log("🗑️  Cleaning up processed L5 snapshots...")
        SNAPSHOT_DIR = ROOT / "data/orderbook_history"
        if SNAPSHOT_DIR.exists():
            for f_path in SNAPSHOT_DIR.glob("ob_*.json"):
                f_path.unlink()
            log("   ✅ L5 Snapshots purged.")
    else:
        shutil.copy2(backup_path, MODEL_PATH)
        log(f"⚠️  No improvement — restored yesterday's model")

    log("=" * 55)
    log(f"✅ Fine-tune complete. Brain adapted to {DAYS}-day regime.")
    log("=" * 55)
except Exception as e:
    import traceback
    log(f"❌ CRITICAL FINE-TUNE RUNTIME ERROR: {e}")
    log(traceback.format_exc())
    
    # Safe Rollback Safeguard
    if 'backup_path' in locals() and backup_path.exists() and MODEL_PATH.exists():
        try:
            shutil.copy2(backup_path, MODEL_PATH)
            log(f"🛡️  SAFEGUARD: Rolled back sandbox active model to clean state from backup: {backup_path.name}")
        except Exception as rollback_err:
            log(f"❌ CRITICAL: Safe rollback failed: {rollback_err}")
    sys.exit(1)
finally:
    # 🚀 Self-Healing Watchdog: Restart the base training orchestrator train.py in background
    try:
        # Check if already running to avoid duplicates
        import subprocess, os
        pids_raw = subprocess.check_output(["pgrep", "-f", "train.py"]).decode().strip().split()
        is_running = any(int(p) != os.getpid() for p in pids_raw if p.isdigit())
    except Exception:
        is_running = False

    if not is_running:
        log("🚀 SELF-HEALING WATCHDOG: Restarting the base training orchestrator train.py in background...")
        try:
            subprocess.Popen(
                "nohup /root/miniconda3/bin/python -u /var/www/html/ML/kat/train.py "
                "--model hydra --epochs 300 --batch 32 --candles 120000 "
                "--timeframe 15m --symbol BTCUSD --resume "
                "> /var/www/html/ML/kat/logs/hydra_train_$(date +%Y%m%d_%H%M%S).log 2>&1 &",
                shell=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            log("   ✅ Base training process restarted cleanly.")
        except Exception as relaunch_err:
            log(f"❌ SELF-HEALING: Failed to restart base training: {relaunch_err}")
    else:
        log("ℹ️  Base training process is already running.")
