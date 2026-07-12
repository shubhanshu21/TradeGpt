"""
SOVEREIGN KRAKEN TRAINING ORCHESTRATOR (V10.7 — Predator) ⚓🚀⚡
===========================================================================
- Model: HYDRA V10.7 (128-wide, 8-block, 256-Expert MoE + SwiGLU + TurboQuant)
- RAM Strategy: One-time precomputed numpy arrays (~2.6GB) → from_tensor_slices
  Eliminates 30x epoch speed oscillation caused by from_generator recomputation.
- Context: 120 candles (2 hours) — proven stable on this hardware
- Batch: 32 — calibrated for 24GB RAM host
"""

import os, argparse, gc, glob, time
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
from core.hydra import build_kraken, IS_GPU, init_kraken_hardware, CertaintyMetric, SovereignAccuracy, SovereignLoss, certainty_loss, WarmupCosineDecay
from data.preprocess import build_dataset_streaming, build_feature_cols, KATScaler
from exchange.fetch_data import fetch_live_kat_data
import glob as _glob
from config.sovereign_config import FEE_RATE, POSITION_SIZE_PCT, CONTEXT_WINDOW, FORECAST_STEPS


class CheckpointPruner(keras.callbacks.Callback):
    """
    Enhancement #5 — Auto-prune old epoch checkpoints.
    Keeps: best val_loss checkpoint + last N epoch checkpoints.
    Prevents disk fill during 300-epoch runs.
    """
    def __init__(self, ckpt_dir: Path, model_name: str = "hydra", keep_n: int = 3):
        super().__init__()
        self.ckpt_dir   = ckpt_dir
        self.model_name = model_name
        self.keep_n     = keep_n

    def on_epoch_end(self, epoch, logs=None):
        prefix = "hydra" if self.model_name == "hydra" else self.model_name
        pattern  = str(self.ckpt_dir / f"{prefix}_checkpoint_E*.keras")
        all_ckpt = sorted(_glob.glob(pattern))
        # Keep only the last keep_n; delete the rest
        to_delete = all_ckpt[: max(0, len(all_ckpt) - self.keep_n)]
        for f in to_delete:
            # Defensive check to protect 'best' checkpoints if patterns or names overlap
            if "best" not in os.path.basename(f):
                try:
                    os.remove(f)
                except OSError:
                    pass


class EpochCheckpointSaver(keras.callbacks.Callback):
    """
    Saves periodic epoch checkpoints cleanly, independent of static batch step counts
    to avoid I/O stalls and dynamic dataset mismatch bugs.
    """
    def __init__(self, ckpt_dir: Path, model_name: str = "hydra", freq: int = 10):
        super().__init__()
        self.ckpt_dir   = ckpt_dir
        self.model_name = model_name
        self.freq       = freq

    def on_epoch_end(self, epoch, logs=None):
        actual_epoch = epoch + 1
        if actual_epoch % self.freq == 0:
            prefix = "hydra" if self.model_name == "hydra" else self.model_name
            ckpt_path = self.ckpt_dir / f"{prefix}_checkpoint_E{actual_epoch:03d}.keras"
            try:
                self.model.save(str(ckpt_path))
                print(f"\n💾 Saved epoch checkpoint to {ckpt_path.name}")
            except Exception as e:
                print(f"\n⚠️ Warning: Failed to save epoch checkpoint: {e}")


class EdgeTracker(keras.callbacks.Callback):
    """
    Statistical significance tracker for directional accuracy.

    A raw val_prediction_dir_acc reading is noisy: with ~12k validation
    samples, the 95% confidence interval on a ~51% reading is roughly
    +/-0.9 points, so epoch-to-epoch swings of a point or two (which this
    model has shown before) can be pure noise, not real learning. This
    computes a Wilson score interval each epoch so "is there a real edge
    yet" can be answered with statistics instead of eyeballing a number,
    and logs it to a clean CSV separate from the raw step-by-step log.
    """
    def __init__(self, log_path: Path, n_val_samples: int):
        super().__init__()
        self.log_path = Path(log_path)
        self.n = max(1, n_val_samples)
        self.best_acc = 0.5
        self.best_epoch = 0
        if not self.log_path.exists():
            with open(self.log_path, "w") as f:
                f.write("epoch,val_dir_acc,ci_low_95,ci_high_95,significant_edge,"
                        "epochs_since_best,val_loss,train_dir_acc\n")

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        p = float(logs.get("val_prediction_dir_acc", 0.5))

        # Wilson score interval — more reliable than a normal approximation
        # when p sits close to the 0.5 boundary we actually care about.
        z = 1.959964  # 95% two-sided
        denom  = 1.0 + (z * z) / self.n
        center = (p + (z * z) / (2 * self.n)) / denom
        margin = (z * ((p * (1 - p) / self.n + (z * z) / (4 * self.n ** 2)) ** 0.5)) / denom
        ci_low, ci_high = center - margin, center + margin
        significant = ci_low > 0.50

        if p > self.best_acc:
            self.best_acc, self.best_epoch = p, epoch + 1
        epochs_since_best = (epoch + 1) - self.best_epoch

        with open(self.log_path, "a") as f:
            f.write(f"{epoch+1},{p:.4f},{ci_low:.4f},{ci_high:.4f},{significant},"
                    f"{epochs_since_best},{logs.get('val_loss', 0.0):.4f},"
                    f"{logs.get('prediction_dir_acc', 0.0):.4f}\n")

        verdict = "✅ STATISTICALLY SIGNIFICANT EDGE" if significant else "— not significant yet (could be noise)"
        print(f"\n📐 EDGE CHECK | val_dir_acc={p*100:.2f}%  95% CI=[{ci_low*100:.2f}%, {ci_high*100:.2f}%]  "
              f"{verdict}  | best={self.best_acc*100:.2f}% @ epoch {self.best_epoch} "
              f"({epochs_since_best} epochs since best)")


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
    CTX_WIN  = args.context_window if args.context_window else CONTEXT_WINDOW
    FORECAST = args.forecast_steps if args.forecast_steps else FORECAST_STEPS

    # ── 1. Fetch / Cache ──────────────────────────────────────────────────────
    DATA_DIR.mkdir(exist_ok=True)
    CKPT_DIR.mkdir(exist_ok=True)
    LOG_DIR.mkdir(exist_ok=True)
    # Use a dynamic 'master' filename to avoid hardcoding
    CACHE_P = DATA_DIR / f"{args.symbol}_{args.timeframe}_history_master.parquet"
    
    if CACHE_P.exists():
        print(f"📖 CACHE: Loading {CANDLES:,} candles from {CACHE_P.name}...")
        df = pd.read_parquet(CACHE_P).tail(CANDLES)
    else:
        print(f"📡 No master cache found for {args.symbol}. Fetching fresh candles...")
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
    # Use exact label counts from preprocess.py (same Z-score + dynamic fee_gate logic
    # used during dataset construction — previously used pct_change which mismatched).
    print("   📊 Computing reasoning class weights from precomputed label distribution...")
    label_counts = np.maximum(ds_info["label_counts"], 1)
    total = label_counts.sum()
    class_weights = {i: min(total / (4 * label_counts[i]), 5.0) for i in range(4)}
    print(f"   ⚖️  Class weights: Bull={class_weights[0]:.2f} Bear={class_weights[1]:.2f} "
          f"FeeTrap={class_weights[2]:.2f} Noise={class_weights[3]:.2f}")

    # ── 3. Build Model ────────────────────────────────────────────────────────
    # For 1,152 experts, we use a slower, higher-quality learning profile.
    vocab_size = ds_info["vocab_size"]
    model = build_kraken(n_features=n_feat, context_window=CTX_WIN,
                        forecast_steps=FORECAST, vocab_size=vocab_size)

    # Save the return-token vocabulary AND the exact model shape this checkpoint
    # was trained with — context_window/forecast_steps/n_features differ per run
    # (e.g. --timeframe 1h uses a smaller context window), and any inference code
    # that rebuilds the model to load these weights needs to match exactly or the
    # layer shapes (e.g. MLALayer's RoPE buffers) won't align.
    import pickle as _pickle
    with open(CKPT_DIR / "return_vocab.pkl", "wb") as _f:
        _pickle.dump({"bin_edges": ds_info["bin_edges"], "bin_centers": ds_info["bin_centers"],
                      "vocab_size": vocab_size, "context_window": CTX_WIN,
                      "forecast_steps": FORECAST, "n_features": n_feat,
                      "timeframe": args.timeframe}, _f)
    
    # Custom weighted loss for reasoning to circumvent Keras 3 tf_dataset_adapter class_weight bug
    weights = [class_weights[i] for i in range(4)]
    weights_tensor = tf.constant(weights, dtype=tf.float32)
    
    def weighted_reasoning_loss(y_true, y_pred):
        # Flatten y_true to 1D to prevent dimension mismatches under dynamic shapes in tf.function compiled runs
        y_true_int = tf.reshape(tf.cast(y_true, tf.int32), [-1])
        # Apply label smoothing (0.1) on one-hot reasoning targets
        y_true_one_hot = tf.one_hot(y_true_int, depth=4)
        unweighted = tf.keras.losses.categorical_crossentropy(
            y_true_one_hot, y_pred, label_smoothing=0.1
        )
        sample_weights = tf.gather(weights_tensor, y_true_int)
        return unweighted * sample_weights

    # Rebuild LR schedule so cosine decay spans the full training run instead of
    # hitting its floor at epoch ~4 (decay_steps=10000 << 300*2996 total steps).
    # Linear warmup over the first epoch's worth of steps as cheap insurance
    # against early MoE routing instability.
    # 1e-4 peak, not 5e-6 — the old value was 20-100x below what research shows
    # is typical for training a transformer this size from scratch, and produced
    # the noise-dominated, barely-moving progress that's the textbook symptom of
    # too-low a learning rate. See matching comment in core/hydra.py.
    full_lr_schedule = WarmupCosineDecay(
        initial_learning_rate=1e-4,
        decay_steps=EPOCHS * steps_tr,
        warmup_steps=steps_tr,
        alpha=0.1   # floor = 1e-5
    )

    print("   ⚖️  Recompiling model with custom weighted sparse categorical crossentropy...")
    model.compile(
        optimizer=keras.optimizers.AdamW(
            learning_rate=full_lr_schedule,
            weight_decay=0.05,  # raised from 0.01 — train/val gap showed real overfitting signal
            clipnorm=0.5
        ),
        loss={
            "prediction": SovereignLoss(direction_weight=3.0),
            "certainty":  certainty_loss,
            "reasoning":  weighted_reasoning_loss,
            "next_token": keras.losses.SparseCategoricalCrossentropy(),
        },
        # prediction=6 restores its original ~60% share of the training signal —
        # adding next_token at weight 2 on top of the old {3,1,1} diluted
        # prediction from 60% to 43% of the gradient, likely why the model found
        # a small edge fast and then stopped improving. Scaled up to compensate.
        loss_weights={"prediction": 6.0, "certainty": 1.0, "reasoning": 1.0, "next_token": 2.0},
        metrics={
            "prediction": [SovereignAccuracy()],
            "certainty":  [CertaintyMetric()],
            "next_token": [keras.metrics.SparseCategoricalAccuracy(name="token_acc")],
        }
    )

    # ── 4. Load Weights ───────────────────────────────────────────────────────
    ckpt_name = "hydra_best.keras" if args.model == "hydra" else f"{args.model}_best.keras"
    prefix = "hydra" if args.model == "hydra" else args.model
    CKPT_BEST = CKPT_DIR / ckpt_name
    saved     = sorted(glob.glob(str(CKPT_DIR / f"{prefix}_checkpoint_E*.keras")))

    if args.resume and saved:
        print(f"📦 Loading weights from {os.path.basename(saved[-1])}")
        model.load_weights(saved[-1])
    elif args.resume and CKPT_BEST.exists():
        print(f"📦 Loading weights from {ckpt_name}")
        model.load_weights(str(CKPT_BEST))

    # ── 5. Callbacks ──────────────────────────────────────────────────────────
    # Monitor the direction-accuracy metric directly rather than total val_loss,
    # since val_loss is dominated by the certainty/reasoning heads and can keep
    # "improving" while direction accuracy (the metric that matters for trading)
    # stays flat.
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            str(CKPT_BEST), monitor="val_prediction_dir_acc", mode="max",
            save_best_only=True, verbose=1),
        EpochCheckpointSaver(ckpt_dir=CKPT_DIR, model_name=args.model, freq=10),  # Save periodic epoch checkpoints natively
        keras.callbacks.EarlyStopping(
            monitor="val_prediction_dir_acc", mode="max", patience=20,
            restore_best_weights=True, verbose=1),
        CheckpointPruner(ckpt_dir=CKPT_DIR, model_name=args.model, keep_n=3),
        EdgeTracker(log_path=LOG_DIR / "edge_tracker.csv",
                    n_val_samples=steps_va * BATCH_S * FORECAST),
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

    final_name = "hydra_final.keras" if args.model == "hydra" else f"{args.model}_final.keras"
    model.save(str(CKPT_DIR / final_name))
    print("\n✅ MISSION COMPLETE — Sovereign Alpha-Brain saved.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--symbol",    default="BTCUSD")
    p.add_argument("--timeframe", default="15m")   # 15m: maximum SNR for swing trading
    p.add_argument("--epochs",    type=int, default=300)
    p.add_argument("--model",     default="hydra")
    p.add_argument("--batch",     type=int, default=8)
    p.add_argument("--candles",   type=int, default=120000)
    p.add_argument("--context_window", type=int, default=None,
                    help="Override CONTEXT_WINDOW candle count (e.g. for non-15m timeframes)")
    p.add_argument("--forecast_steps", type=int, default=None,
                    help="Override FORECAST_STEPS candle count (e.g. for non-15m timeframes)")
    p.add_argument("--resume",    action="store_true")
    p.add_argument("--fee_rate",  type=float, default=FEE_RATE)
    args = p.parse_args()
    train_kraken(args)