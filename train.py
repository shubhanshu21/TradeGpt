"""
SOVEREIGN KRAKEN TRAINING ORCHESTRATOR — Equity Swing Edition
===========================================================================
Converted from a single-asset crypto orchestrator to a per-symbol Indian
equity swing-trading one. See src/config/sovereign_config.py's module
docstring for the full context on this conversion.

- Model: HYDRA (128-wide, 8-block, 32-Expert MoE + SwiGLU), single-input,
  4 outputs (prediction/certainty/reasoning/next_candle) — includes a
  GPT-style single-step next-candle token head, see src/core/hydra.py's
  build_kraken docstring for the single-step-not-generation design.
- Two-phase training, not from-scratch-per-symbol: (1) PRETRAIN one shared
  model on ALL universe symbols pooled together - broad general pattern
  recognition, the same idea as a trader who's watched hundreds of stocks
  before specializing. (2) FINE-TUNE a separate copy of that pretrained
  model per symbol (models/<SYMBOL>/hydra_best.keras), starting from the
  pretrained weights instead of random init, at a lower learning rate.
  This exists specifically because per-symbol-from-scratch training showed
  a real overfitting risk (val accuracy peaking at epoch 1, drifting down
  after) - each symbol alone only has ~5,000 windows, not enough to learn
  robust patterns unaided; pretraining gives every symbol's fine-tune a
  running start instead of learning everything from that symbol's data alone.
- Context: 60 trading days (~3 months) informing a swing entry held for up
  to MAX_HOLDING_DAYS (20 trading days).
"""

import os, argparse, gc, glob, time
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # silence TF C++ allocator/prefetch WARNING spam - actual errors still show
import numpy as np
import tensorflow as tf
import keras
from pathlib import Path

ROOT     = Path(__file__).parent
CKPT_DIR = ROOT / "models"
LOG_DIR  = ROOT / "logs"
PRETRAIN_DIR = CKPT_DIR / "_pretrained_base"
PRETRAIN_CKPT = PRETRAIN_DIR / "hydra_pretrained.keras"

import sys
sys.path.insert(0, str(ROOT / "src"))
from core.hydra import build_kraken, IS_GPU, init_kraken_hardware, CertaintyMetric, SovereignAccuracy, SovereignLoss, certainty_loss, WarmupCosineDecay
from data.preprocess import build_dataset_streaming, load_universe_from_cache
from config.sovereign_config import CONTEXT_WINDOW, FORECAST_STEPS, TRAINING_SYMBOLS, UNIVERSE_SYMBOLS


class CheckpointPruner(keras.callbacks.Callback):
    """
    Auto-prune old epoch checkpoints. Keeps: best val_loss checkpoint + last
    N epoch checkpoints. Prevents disk fill during long runs.
    """
    def __init__(self, ckpt_dir: Path, model_name: str = "hydra", keep_n: int = 3):
        super().__init__()
        self.ckpt_dir   = ckpt_dir
        self.model_name = model_name
        self.keep_n     = keep_n

    def on_epoch_end(self, epoch, logs=None):
        prefix = "hydra" if self.model_name == "hydra" else self.model_name
        pattern  = str(self.ckpt_dir / f"{prefix}_checkpoint_E*.keras")
        all_ckpt = sorted(glob.glob(pattern))
        to_delete = all_ckpt[: max(0, len(all_ckpt) - self.keep_n)]
        for f in to_delete:
            if "best" not in os.path.basename(f):
                try:
                    os.remove(f)
                except OSError:
                    pass


class EpochCheckpointSaver(keras.callbacks.Callback):
    """Saves periodic epoch checkpoints, independent of resumption logic."""
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
    Statistical significance tracker for directional accuracy — Wilson score
    interval on val_prediction_dir_acc each epoch, so "is there a real edge
    yet" is answered with statistics instead of eyeballing a noisy number.
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
    init_kraken_hardware()

    symbols = args.symbols if args.symbols else TRAINING_SYMBOLS

    print("\n" + "="*60)
    print(f"  {'🚀 GPU MODE' if IS_GPU else '🐌 CPU MODE'} — SOVEREIGN KRAKEN (Equity Swing)")
    print(f"  Phase 1: pretrain on all universe symbols pooled (broad experience)")
    print(f"  Phase 2: fine-tune {len(symbols)} per-symbol specialists: {', '.join(symbols)}")
    print("="*60)

    if not args.skip_pretrain and (args.force_pretrain or not PRETRAIN_CKPT.exists()):
        pretrain_base_model(args)
        keras.backend.clear_session()
        gc.collect()
    elif not args.skip_pretrain:
        print(f"\n📦 Reusing existing pretrained base at {PRETRAIN_CKPT} "
              f"(pass --force_pretrain to redo it)")

    for idx, symbol in enumerate(symbols, 1):
        print(f"\n{'#'*60}\n#  [{idx}/{len(symbols)}]  {symbol}\n{'#'*60}")
        train_one_symbol(symbol, args)
        # Each symbol builds its own model/graph/optimizer state - clear it
        # before moving to the next one so a long multi-symbol run doesn't
        # accumulate memory across models.
        keras.backend.clear_session()
        gc.collect()

    print("\n✅ ALL SYMBOLS COMPLETE — Sovereign Alpha-Brain(s) saved.")


def _compute_class_weights(ds_info):
    label_counts = np.maximum(ds_info["label_counts"], 1)
    total = label_counts.sum()
    class_weights = {i: min(total / (4 * label_counts[i]), 5.0) for i in range(4)}
    print(f"   ⚖️  Class weights: Long={class_weights[0]:.2f} Short={class_weights[1]:.2f} "
          f"FeeTrap={class_weights[2]:.2f} Noise={class_weights[3]:.2f}")
    return class_weights


def _compile_hydra(model, class_weights, epochs, steps_tr, learning_rate, weight_decay):
    """Shared compile step for both the pretrain and fine-tune phases -
    same loss/metric setup, only the learning rate/weight decay differ
    (fine-tuning uses a lower LR - standard transfer-learning practice, so
    the per-symbol phase adjusts the pretrained weights instead of
    overwriting what they already learned)."""
    weights = [class_weights[i] for i in range(4)]
    weights_tensor = tf.constant(weights, dtype=tf.float32)

    def weighted_reasoning_loss(y_true, y_pred):
        y_true_int = tf.reshape(tf.cast(y_true, tf.int32), [-1])
        y_true_one_hot = tf.one_hot(y_true_int, depth=4)
        unweighted = tf.keras.losses.categorical_crossentropy(
            y_true_one_hot, y_pred, label_smoothing=0.1
        )
        sample_weights = tf.gather(weights_tensor, y_true_int)
        return unweighted * sample_weights

    lr_schedule = WarmupCosineDecay(
        initial_learning_rate=learning_rate,
        decay_steps=max(1, epochs * steps_tr),
        warmup_steps=steps_tr,
        alpha=0.1
    )

    model.compile(
        optimizer=keras.optimizers.AdamW(
            learning_rate=lr_schedule,
            weight_decay=weight_decay,
            clipnorm=1.0
        ),
        loss={
            "prediction": SovereignLoss(direction_weight=3.0),
            "certainty":  certainty_loss,
            "reasoning":  weighted_reasoning_loss,
            # Real next-token cross-entropy, same loss GPT training uses -
            # y_true here is an integer token ID (not one-hot), so sparse.
            "next_candle": keras.losses.SparseCategoricalCrossentropy(),
        },
        # next_candle weighted lower than the actual trade-decision heads -
        # it's a genuine auxiliary task (real next-token prediction, GPT-
        # style), not the thing the trading logic gates on. Keeping its
        # weight modest stops it from competing for gradient signal against
        # prediction/reasoning, which are what ml_strategy.py actually acts on.
        loss_weights={"prediction": 6.0, "certainty": 1.0, "reasoning": 1.0, "next_candle": 0.5},
        metrics={
            "prediction": [SovereignAccuracy()],
            "certainty":  [CertaintyMetric()],
            "next_candle": [keras.metrics.SparseCategoricalAccuracy(name="next_candle_acc")],
        }
    )


def pretrain_base_model(args):
    """Phase 1: one shared model trained on ALL universe symbols pooled
    together - broad general pattern recognition across many stocks, the
    same principle as a trader who's watched hundreds of stocks before
    specializing in a few. Measured directly on this hardware at ~4,957
    steps/epoch, ~19h/epoch for the full 38-symbol pool - genuinely slow,
    but that's an accepted tradeoff here, not an oversight.
    """
    PRETRAIN_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    BATCH_S  = args.batch
    EPOCHS   = args.pretrain_epochs
    CTX_WIN  = args.context_window if args.context_window else CONTEXT_WINDOW
    FORECAST = args.forecast_steps if args.forecast_steps else FORECAST_STEPS
    pretrain_symbols = UNIVERSE_SYMBOLS

    print(f"\n🌐 PRETRAIN: loading all {len(pretrain_symbols)} universe symbols for pooled training...")
    data_by_symbol = load_universe_from_cache(pretrain_symbols)
    if not data_by_symbol:
        print("❌ No data loaded for pretraining - skipping pretrain phase, "
              "per-symbol fine-tunes will start from random init instead.")
        return

    ds_info = build_dataset_streaming(data_by_symbol, context_window=CTX_WIN,
                                       forecast_steps=FORECAST, batch_size=BATCH_S)
    tr_ds, va_ds = ds_info["tr_ds"], ds_info["va_ds"]
    steps_tr, steps_va = ds_info["steps_tr"], ds_info["steps_va"]
    n_feat = ds_info["n_features"]
    print(f"   ✅ Pooled across {len(data_by_symbol)} symbols: {steps_tr} train steps/epoch | {steps_va} val steps")

    class_weights = _compute_class_weights(ds_info)
    model = build_kraken(n_features=n_feat, context_window=CTX_WIN, forecast_steps=FORECAST)
    _compile_hydra(model, class_weights, EPOCHS, steps_tr,
                    learning_rate=1e-4, weight_decay=0.05)  # pooled data is much larger - the per-symbol 0.15 was specifically to fight per-symbol overfitting

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            str(PRETRAIN_CKPT), monitor="val_prediction_dir_acc", mode="max",
            save_best_only=True, verbose=1),
        keras.callbacks.EarlyStopping(
            # Same reasoning as the per-symbol phase's patience=25:
            # restore_best_weights=True means a generous patience can't make
            # the result worse, only spend more time looking for a better
            # epoch - and time isn't the constraint here.
            monitor="val_prediction_dir_acc", mode="max", patience=25,
            restore_best_weights=True, verbose=1),
        EdgeTracker(log_path=LOG_DIR / "edge_tracker__pretrained_base.csv",
                    n_val_samples=steps_va * BATCH_S * FORECAST),
    ]

    print(f"\n🚀 IGNITION: pretrain base | {EPOCHS}-Epoch Mission | Batch {BATCH_S} | CTX {CTX_WIN} trading days")
    model.fit(
        tr_ds, validation_data=va_ds, epochs=EPOCHS,
        steps_per_epoch=steps_tr, validation_steps=steps_va,
        callbacks=callbacks, shuffle=False, verbose=1,
    )
    print(f"\n✅ PRETRAIN COMPLETE — base weights saved to {PRETRAIN_CKPT}")


def train_one_symbol(symbol: str, args):
    ckpt_dir = CKPT_DIR / symbol
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    BATCH_S  = args.batch
    EPOCHS   = args.epochs
    CTX_WIN  = args.context_window if args.context_window else CONTEXT_WINDOW
    FORECAST = args.forecast_steps if args.forecast_steps else FORECAST_STEPS

    # ── 1. Load this symbol's real cached daily candles ─────────────────────
    print(f"📖 Loading cached daily candles for {symbol}...")
    data_by_symbol = load_universe_from_cache([symbol])
    if not data_by_symbol:
        print(f"❌ No data loaded for {symbol}. Skipping."); return

    # ── 2. Single-symbol streaming dataset ───────────────────────────────────
    ds_info = build_dataset_streaming(data_by_symbol, context_window=CTX_WIN,
                                       forecast_steps=FORECAST, batch_size=BATCH_S)
    tr_ds   = ds_info["tr_ds"]
    va_ds   = ds_info["va_ds"]
    steps_tr = ds_info["steps_tr"]
    steps_va = ds_info["steps_va"]
    n_feat   = ds_info["n_features"]

    print(f"   ✅ {steps_tr} train steps/epoch | {steps_va} val steps")

    # ── 2b. Reasoning class weights (anti-imbalance) ─────────────────────────
    class_weights = _compute_class_weights(ds_info)

    # ── 3. Build model ───────────────────────────────────────────────────────
    model = build_kraken(n_features=n_feat, context_window=CTX_WIN, forecast_steps=FORECAST)

    # Save the exact shape this checkpoint was trained with — any inference
    # code that rebuilds the model to load these weights needs to match
    # exactly or the layer shapes won't align. bin_edges/bin_centers are the
    # next-candle vocabulary THIS symbol's own data produced (build_kraken's
    # next_candle Dense layer is always a fixed 32-wide, regardless of how
    # many distinct tokens actually appear - see fit_return_vocab's
    # docstring - so this never causes a load_weights shape mismatch against
    # the pretrained base; bin_edges/bin_centers are only needed to DECODE
    # a predicted token back into an implied direction at inference time).
    import pickle as _pickle
    with open(ckpt_dir / "model_shape.pkl", "wb") as _f:
        _pickle.dump({"context_window": CTX_WIN, "forecast_steps": FORECAST,
                      "n_features": n_feat, "symbol": symbol,
                      "bin_edges": ds_info["bin_edges"], "bin_centers": ds_info["bin_centers"]}, _f)

    # ── 4. Load starting weights: resume > pretrained base > random init ────
    ckpt_name = "hydra_best.keras" if args.model == "hydra" else f"{args.model}_best.keras"
    prefix = "hydra" if args.model == "hydra" else args.model
    CKPT_BEST = ckpt_dir / ckpt_name
    saved     = sorted(glob.glob(str(ckpt_dir / f"{prefix}_checkpoint_E*.keras")))

    resumed_from_symbol_ckpt = False
    if args.resume and saved:
        print(f"📦 Resuming from this symbol's own checkpoint: {os.path.basename(saved[-1])}")
        model.load_weights(saved[-1])
        resumed_from_symbol_ckpt = True
    elif args.resume and CKPT_BEST.exists():
        print(f"📦 Resuming from this symbol's own checkpoint: {ckpt_name}")
        model.load_weights(str(CKPT_BEST))
        resumed_from_symbol_ckpt = True
    elif not args.skip_pretrain and PRETRAIN_CKPT.exists():
        print(f"📦 Starting from the pretrained base ({PRETRAIN_CKPT.name}) - "
              f"broad patterns learned across all universe symbols, now specializing on {symbol}")
        model.load_weights(str(PRETRAIN_CKPT))
    else:
        print("📦 No pretrained base available - starting from random initialization")

    # Fine-tuning from a pretrained base uses a lower learning rate than
    # training from scratch would - standard transfer-learning practice, so
    # this phase adjusts what the base already learned instead of overwriting
    # it with the higher LR that made sense for random-init training.
    learning_rate = args.finetune_lr if (not resumed_from_symbol_ckpt and not args.skip_pretrain and PRETRAIN_CKPT.exists()) else 1e-4
    print(f"   ⚖️  Recompiling model with custom weighted categorical crossentropy (lr={learning_rate})...")
    _compile_hydra(model, class_weights, EPOCHS, steps_tr,
                    learning_rate=learning_rate, weight_decay=0.15)

    # ── 5. Callbacks ──────────────────────────────────────────────────────────
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            str(CKPT_BEST), monitor="val_prediction_dir_acc", mode="max",
            save_best_only=True, verbose=1),
        EpochCheckpointSaver(ckpt_dir=ckpt_dir, model_name=args.model, freq=10),
        keras.callbacks.EarlyStopping(
            # restore_best_weights=True means the FINAL saved model is always
            # whichever epoch had the best val score, regardless of patience -
            # patience only controls how long we keep looking for a better
            # epoch before giving up, it can't make the result worse. A low
            # patience risks quitting before a real (later) peak shows up;
            # the only cost of a generous patience is extra compute time,
            # which is fine here - so keep this generous rather than tight.
            monitor="val_prediction_dir_acc", mode="max", patience=25,
            restore_best_weights=True, verbose=1),
        CheckpointPruner(ckpt_dir=ckpt_dir, model_name=args.model, keep_n=3),
        EdgeTracker(log_path=LOG_DIR / f"edge_tracker_{symbol}.csv",
                    n_val_samples=steps_va * BATCH_S * FORECAST),
    ]

    # ── 6. Ignite ─────────────────────────────────────────────────────────────
    print(f"\n🚀 IGNITION: {symbol} | {EPOCHS}-Epoch Mission | Batch {BATCH_S} | CTX {CTX_WIN} trading days")

    current_epoch = 0
    if args.resume and saved:
        try:
            latest_file = os.path.basename(saved[-1])
            import re
            match = re.search(r"E(\d+)", latest_file)
            if match:
                current_epoch = int(match.group(1))
                print(f"   🎯 RESUMPTION: Detected finished epoch {current_epoch}. Starting Epoch {current_epoch+1}...")
        except Exception as e:
            print(f"   ⚠️ Could not parse epoch from filename: {e}")
            current_epoch = len(saved)

    model.fit(
        tr_ds,
        validation_data=va_ds,
        epochs=EPOCHS,
        initial_epoch=current_epoch,
        steps_per_epoch=steps_tr,
        validation_steps=steps_va,
        callbacks=callbacks,
        shuffle=False,  # already pre-shuffled in preprocess.py; avoids Keras's tf.data.Dataset warning
        verbose=1
    )

    final_name = "hydra_final.keras" if args.model == "hydra" else f"{args.model}_final.keras"
    model.save(str(ckpt_dir / final_name))
    print(f"\n✅ {symbol} MISSION COMPLETE — saved to {ckpt_dir}/{final_name}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model",     default="hydra")
    p.add_argument("--epochs",    type=int, default=300)
    p.add_argument("--batch",     type=int, default=32)
    p.add_argument("--symbols",   nargs="+", default=None,
                    help="Which symbols to train - one independent model each, trained sequentially "
                         "(default: config/settings.yaml -> training.symbols, or the full universe.symbols "
                         "if that's null)")
    p.add_argument("--context_window", type=int, default=None,
                    help="Override CONTEXT_WINDOW trading-day count")
    p.add_argument("--forecast_steps", type=int, default=None,
                    help="Override FORECAST_STEPS trading-day count")
    p.add_argument("--resume",    action="store_true")
    p.add_argument("--pretrain_epochs", type=int, default=300,
                    help="Epoch ceiling for the phase-1 pooled pretrain - matches --epochs since "
                         "EarlyStopping+restore_best_weights means a higher ceiling only costs time, "
                         "never quality, and time isn't the constraint here")
    p.add_argument("--skip_pretrain", action="store_true",
                    help="Skip the pooled-pretrain phase entirely - each symbol trains from random init, "
                         "like before this two-phase approach existed")
    p.add_argument("--force_pretrain", action="store_true",
                    help="Redo the pretrain phase even if models/_pretrained_base/hydra_pretrained.keras already exists")
    p.add_argument("--finetune_lr", type=float, default=3e-5,
                    help="Learning rate for per-symbol fine-tuning FROM the pretrained base "
                         "(lower than the 1e-4 used for pretraining/from-scratch - standard transfer-learning practice)")
    args = p.parse_args()
    train_kraken(args)
