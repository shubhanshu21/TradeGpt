"""
SOVEREIGN KRAKEN — Daily Fine-Tuner (V5.0) ⚓📅
================================================
Adapts the trained brain to the latest market regime each morning.

Run manually:
    python scripts/daily_finetune.py

Or via cron (midnight UTC daily):
    5 0 * * * /root/miniconda3/bin/python /var/www/html/ML/kat/scripts/daily_finetune.py >> /var/www/html/ML/kat/logs/finetune.log 2>&1
"""

import sys, shutil
import numpy as np
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

import keras
from architectures.hydra import (HydraV4, HydraBlock, GatedMoE,
                                  MLAAttention, RMSNorm, TTMReflex, SovereignLoss)
from preprocess import build_dataset_streaming, KATScaler
from fetch_data  import fetch_live_kat_data

# ── DEFAULTS ──────────────────────────────────────────────────────────────────
DAYS         = 2        # Days of recent data to fine-tune on (~2,880 candles)
EPOCHS       = 5        # Fine-tune epochs — keep small (3–10)
LR           = 1e-6     # Very low LR — nudge, don't overwrite base knowledge
CTX_WIN      = 120      # Must match training context window
BATCH        = 64       # Smaller batch for fine-tune stability
FREEZE_BELOW = 9        # Freeze first N blocks (protect foundational patterns)
MODEL_FILE   = "hydra_best.keras"
SYMBOL       = "BTCUSD"
TIMEFRAME    = "1m"
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
    log("   Run train.py first to generate hydra_best.keras")
    sys.exit(1)

# ── 2. Backup yesterday's model ───────────────────────────────────────────────
BACKUP_DIR.mkdir(exist_ok=True)
date_str    = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M")
backup_path = BACKUP_DIR / f"hydra_best_{date_str}.keras"
shutil.copy2(MODEL_PATH, backup_path)
log(f"💾 Backup: {backup_path.name}")

# Prune old backups — keep last KEEP_BACKUPS
all_backups = sorted(BACKUP_DIR.glob("hydra_best_*.keras"))
for old in all_backups[:-KEEP_BACKUPS]:
    old.unlink()
    log(f"🗑️  Pruned: {old.name}")

# ── 3. Fetch fresh data ───────────────────────────────────────────────────────
log(f"📡 Fetching {CANDLES:,} fresh candles...")
try:
    df = fetch_live_kat_data(symbol=SYMBOL, n_candles=CANDLES, timeframe=TIMEFRAME)
    log(f"   ✅ {len(df):,} candles received")
except Exception as e:
    log(f"❌ Fetch failed: {e}"); sys.exit(1)

if len(df) < CTX_WIN + 50:
    log(f"❌ Insufficient data: {len(df)} candles"); sys.exit(1)

# ── 4. Load scaler ────────────────────────────────────────────────────────────
scaler_path = ROOT / "models" / "scaler_base.pkl"
scaler = KATScaler.load(str(scaler_path)) if scaler_path.exists() else None
log("📐 Scaler: " + ("loaded from training" if scaler else "fitting fresh (no saved scaler)"))

# ── 5. Streaming dataset ──────────────────────────────────────────────────────
log("🌊 Building fine-tune stream...")
ds_info  = build_dataset_streaming(df, context_window=CTX_WIN, forecast_steps=15,
                                    batch_size=BATCH, scaler=scaler)
tr_ds    = ds_info["tr_ds"]
va_ds    = ds_info["va_ds"]
steps_tr = ds_info["steps_tr"]
steps_va = ds_info["steps_va"]
log(f"   ✅ {steps_tr} train steps | {steps_va} val steps")

# ── 6. Load model ─────────────────────────────────────────────────────────────
log(f"🏗️  Loading {MODEL_FILE}...")
custom_objs = {
    "HydraV4": HydraV4, "HydraBlock": HydraBlock, "GatedMoE": GatedMoE,
    "MLAAttention": MLAAttention, "RMSNorm": RMSNorm,
    "TTMReflex": TTMReflex, "SovereignLoss": SovereignLoss,
}
model = keras.models.load_model(str(MODEL_PATH), custom_objects=custom_objs)

# ── 7. Freeze bottom blocks ───────────────────────────────────────────────────
frozen = 0
for layer in model.layers:
    if "hydra_block" in layer.name:
        idx = int(layer.name.split("_")[-1]) if layer.name[-1].isdigit() else 0
        if idx < FREEZE_BELOW:
            layer.trainable = False
            frozen += 1

trainable = sum(w.numpy().size for w in model.trainable_weights)
total     = sum(w.numpy().size for w in model.weights)
log(f"🔒 Frozen {frozen} blocks → {trainable:,} / {total:,} params active")

# ── 8. Recompile at low LR ────────────────────────────────────────────────────
model.compile(
    optimizer=keras.optimizers.AdamW(LR, weight_decay=0.001, clipnorm=0.5),
    loss=SovereignLoss(direction_weight=5.0, label_smooth=0.1),
    metrics=["mae"]
)

# ── 9. Fine-tune ──────────────────────────────────────────────────────────────
log(f"\n� Fine-tuning: {EPOCHS} epochs on latest {DAYS}-day data...")
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
    log(f"✅ Model updated — improved by {delta:.4f}")
else:
    shutil.copy2(backup_path, MODEL_PATH)
    log(f"⚠️  No improvement — restored yesterday's model")

log("=" * 55)
log(f"✅ Fine-tune complete. Brain adapted to {DAYS}-day regime.")
log("=" * 55)
