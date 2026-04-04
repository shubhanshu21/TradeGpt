"""
KAT Training Orchestrator
==========================
Trains all KAT model versions sequentially (or selectively).

Features:
  - Early stopping + ReduceLROnPlateau
  - Checkpoint saving (best val_loss)
  - Training curves logged to CSV
  - Out-of-sample test evaluation with direction accuracy
  - GPU-aware (runs on CPU if no GPU found)

Usage:
    python train.py --model all --epochs 50
    python train.py --model alpha --epochs 100
    python train.py --model causal_tiger --epochs 30
"""

import os, sys, argparse, time
import numpy as np
import pandas as pd
import tensorflow as tf
import keras
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).parent
DATA_DIR   = ROOT / "data"
MODEL_DIR  = ROOT / "src/architectures"
CKPT_DIR   = ROOT / "models"
LOG_DIR    = ROOT / "logs"

sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(MODEL_DIR))

from preprocess  import build_dataset, KATScaler
from alpha       import build_alpha
from titan       import build_titan
from causal      import build_causal, prepare_causal_targets
from hydra       import build_hydra
from fetch_data  import fetch_live_kat_data


# ──────────────────────────────────────────────────────────────────────────────
# GPU CONFIG
# ──────────────────────────────────────────────────────────────────────────────

def configure_gpu():
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        for g in gpus:
            tf.config.experimental.set_memory_growth(g, True)
        print(f"GPU(s) detected: {[g.name for g in gpus]}")
    else:
        print("No GPU detected — running on CPU")


# ──────────────────────────────────────────────────────────────────────────────
# METRICS
# ──────────────────────────────────────────────────────────────────────────────

def direction_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    % of predictions where model correctly predicted price direction
    (up vs down) relative to previous close.
    """
    true_dir = np.sign(np.diff(y_true))
    pred_dir = np.sign(np.diff(y_pred))
    return float(np.mean(true_dir == pred_dir))


def evaluate_model(model, X_test, y_test, scaler: KATScaler, model_name: str):
    print(f"\n── Test Evaluation: {model_name} ────────────────────")
    y_pred_scaled = model.predict(X_test, verbose=0)

    if y_pred_scaled.ndim > 1 and y_pred_scaled.shape[-1] == 1:
        y_pred_scaled = y_pred_scaled[:, -1, 0]  # CAUSAL: take last timestep

    y_pred = scaler.inverse_y(y_pred_scaled.ravel())
    y_true = scaler.inverse_y(y_test.ravel() if y_test.ndim == 1 else y_test[:, -1])

    mae  = np.mean(np.abs(y_pred - y_true))
    rmse = np.sqrt(np.mean((y_pred - y_true)**2))
    mape = np.mean(np.abs((y_pred - y_true) / (y_true + 1e-9))) * 100
    da   = direction_accuracy(y_true, y_pred) * 100

    print(f"   MAE  : ${mae:,.2f}")
    print(f"   RMSE : ${rmse:,.2f}")
    print(f"   MAPE : {mape:.3f}%")
    print(f"   Dir Acc: {da:.1f}%")
    return {"mae": mae, "rmse": rmse, "mape": mape, "dir_acc": da}


# ──────────────────────────────────────────────────────────────────────────────
# CALLBACKS
# ──────────────────────────────────────────────────────────────────────────────

def get_callbacks(model_name: str, patience: int = 10) -> list:
    CKPT_DIR.mkdir(exist_ok=True)
    LOG_DIR.mkdir(exist_ok=True)

    # DeepSeek-style structured logging: timestamped sub-run
    run_id = time.strftime("%Y%m%d-%H%M%S")
    run_dir = LOG_DIR / model_name / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    return [
        keras.callbacks.ModelCheckpoint(
            filepath=str(CKPT_DIR / f"{model_name}_best.keras"),
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=patience,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=patience // 2,
            min_lr=1e-6,
            verbose=1,
        ),
        keras.callbacks.CSVLogger(
            str(run_dir / "history.csv"),
            append=False,
        ),
        keras.callbacks.TensorBoard(
            log_dir=str(run_dir),
            histogram_freq=0,
            update_freq="epoch",
        ),
    ]


# ──────────────────────────────────────────────────────────────────────────────
# TRAIN KAT 1.3
# ──────────────────────────────────────────────────────────────────────────────

def train_alpha(ds: dict, epochs: int = 50, batch_size: int = 256):
    print("\n" + "═"*60)
    print("  ALPHA — Foundational LSTM Regression")
    print("═"*60)

    n_features = ds["n_features"]
    model = build_alpha(n_features=n_features, context_window=150)

    model.fit(
        ds["X_train"], ds["y_train"],
        validation_data=(ds["X_val"], ds["y_val"]),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=get_callbacks("alpha"),
        verbose=1,
    )

    metrics = evaluate_model(model, ds["X_test"], ds["y_test"], ds["scaler"], "ALPHA")
    model.save(str(CKPT_DIR / "alpha_final.keras"))
    return model, metrics


# ──────────────────────────────────────────────────────────────────────────────
# TRAIN KAT 1.4
# ──────────────────────────────────────────────────────────────────────────────

def train_titan(ds: dict, epochs: int = 30, batch_size: int = 128):
    print("\n" + "═"*60)
    print("  TITAN — Bidirectional LSTM + Transformer (~33M)")
    print("═"*60)

    n_features = ds["n_features"]
    model = build_titan(n_features=n_features, context_window=150)

    history = model.fit(
        ds["X_train"], ds["y_train"],
        validation_data=(ds["X_val"], ds["y_val"]),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=get_callbacks("titan", patience=8),
        verbose=1,
    )

    metrics = evaluate_model(model, ds["X_test"], ds["y_test"], ds["scaler"], "TITAN")
    model.save(str(CKPT_DIR / "titan_final.keras"))
    return model, metrics


# ──────────────────────────────────────────────────────────────────────────────
# TRAIN KAT 2 (base / lion / tiger)
# ──────────────────────────────────────────────────────────────────────────────

def train_causal(ds: dict, variant: str = "base", epochs: int = 40, batch_size: int = 64):
    ctx_map = {"base": 150, "lion": 480, "tiger": 1440}
    ctx = ctx_map[variant]

    print("\n" + "═"*60)
    print(f"  CAUSAL Engine [{variant.upper()}] — GPT Autoregressive  ctx={ctx} min")
    print("═"*60)

    n_features = ds["n_features"]
    model = build_causal(n_features=n_features, variant=variant)

    # ── Re-build dataset with correct context window ──────────────────────────
    # For lion/tiger we need a different window size — rebuild from scratch
    if variant != "base":
        print(f"  Rebuilding dataset for context_window={ctx}...")
        from generate_data import generate_synthetic
        df = generate_synthetic(n_candles=max(60_000, ctx * 80))
        ds_v = build_dataset(
            df,
            context_window=ctx,
            forecast_steps=1,
            scaler_save_path=str(CKPT_DIR / f"scaler_{variant}.pkl"),
        )
    else:
        ds_v = ds

    # Teacher-forcing targets: predict next close for every position in sequence
    X_tr_in, y_tr = prepare_causal_targets(ds_v["X_train"])
    X_va_in, y_va = prepare_causal_targets(ds_v["X_val"])

    history = model.fit(
        X_tr_in, y_tr,
        validation_data=(X_va_in, y_va),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=get_callbacks(f"causal_{variant}", patience=10),
        verbose=1,
    )

    # Evaluate: use last-timestep output vs next close
    X_te_in, y_te = prepare_causal_targets(ds_v["X_test"])
    y_pred = model.predict(X_te_in, verbose=0)[:, -1, 0]
    y_true = y_te[:, -1, 0]

    y_pred_inv = ds_v["scaler"].inverse_y(y_pred)
    y_true_inv = ds_v["scaler"].inverse_y(y_true)

    mae = np.mean(np.abs(y_pred_inv - y_true_inv))
    da  = direction_accuracy(y_true_inv, y_pred_inv) * 100
    print(f"\n── CAUSAL Engine [{variant}] Test  MAE=${mae:,.2f}  DirAcc={da:.1f}%")

    model.save(str(CKPT_DIR / f"causal_{variant}_final.keras"))

    # Demo: generate a 60-step trajectory
    print(f"\n── Generating 60-step trajectory (CAUSAL {variant}) ──")
    seed = ds_v["X_test"][0]   # (ctx, features)
    traj = model.generate(seed, steps=60, scaler=ds_v["scaler"])
    print(f"   Trajectory (USD): {traj[:10].round(2)} ...")

    return model, {"mae": mae, "dir_acc": da}


def prepare_hydra_targets(X: np.ndarray, mtp_steps: int = 5) -> tuple:
    """
    For HYDRA MTP:
      Input = X[:, :-mtp_steps, :]
      Target = (B, T, mtp_steps)
    """
    if X is None or len(X.shape) < 3 or X.shape[0] == 0:
        return X, None

    B, L, F = X.shape
    T = L - mtp_steps
    X_in = X[:, :T, :]
    
    # Target: for each position i, we want close prices at i+1, i+2, ..., i+mtp_steps
    y_blocks = []
    for s in range(1, mtp_steps + 1):
        y_blocks.append(X[:, s:T+s, 3:4]) # close is index 3
    
    y_t = np.concatenate(y_blocks, axis=-1) # (B, T, mtp_steps)
    return X_in, y_t


def train_hydra(ds: dict, epochs: int = 40, batch_size: int = 32, finetune: bool = False):
    print("\n" + "═"*60)
    print(f"  HYDRA Engine {'[FINE-TUNING]' if finetune else '[DeepSeek Ed.]'} — MTP + MoE + MLA (360-min)")
    print("═"*60)

    n_features = ds["n_features"]
    context_window = 360
    
    # ── Load existing or build new ───────────────────────────────────────────
    final_path = CKPT_DIR / "hydra_final.keras"
    if finetune and final_path.exists():
        print(f"   📂 Loading existing model for fine-tuning: {final_path}")
        model = keras.models.load_model(final_path)
        # Use a very low learning rate for fine-tuning
        # This keeps the model's 'deep' knowledge while adapting to recent trends
        model.compile(optimizer=keras.optimizers.Adam(1e-5), loss="mse")
    else:
        print("   🏗️ Building new model from scratch...")
        model = build_hydra(n_features=n_features, context_window=context_window)

    # Re-build dataset if needed, but for MTP we need 5 steps ahead
    # If the base dataset has only 1 step ahead, we might need more
    # However, if we just use the X part and slice it, we are fine as long as 
    # the original windows captured enough future.
    # build_dataset usually provides (ctx + forecast_steps)
    
    # MTP Targets (Predict next 5 steps)
    X_tr_in, y_tr = prepare_hydra_targets(ds["X_train"], mtp_steps=5)
    X_va_in, y_va = prepare_hydra_targets(ds.get("X_val"),   mtp_steps=5)
    
    # ── Final Shape Locking ──────────────────────────────────────────────────
    X_tr_in = X_tr_in.astype("float32")
    
    # Validation safety
    val_data = None
    if X_va_in is not None and y_va is not None and X_va_in.size > 0:
        X_va_in = X_va_in.astype("float32")
        y_va    = y_va.astype("float32")
        val_data = (X_va_in, y_va)
    
    y_tr = y_tr.astype("float32")
    
    print(f"   ✓ Locked Neural Input:  {X_tr_in.shape}")
    print(f"   ✓ Locked Neural Target: {y_tr.shape}")
    
    # Summary reflects established shapes from factory build
    model.summary()

    # ── Dynamic Training Density ─────────────────────────────────────────────
    steps = max(1, len(X_tr_in) // batch_size)
    print(f"   ✓ Training Strategy: {steps} steps per epoch (Dynamic)")

    history = model.fit(
        X_tr_in, y_tr,
        validation_data=val_data,
        epochs=epochs if epochs > 0 else 300, 
        batch_size=batch_size,
        steps_per_epoch=steps,
        callbacks=get_callbacks("hydra", patience=15),
        verbose=1,
    )

    # Evaluate on the first future step (comparable to CAUSAL/ALPHA)
    X_te_in, y_te = prepare_hydra_targets(ds["X_test"], mtp_steps=5)
    out = model.predict(X_te_in, verbose=0) # (B, T, 5)
    y_pred = out[:, -1, 0] # Take the T+1 prediction of the last window
    y_true = y_te[:, -1, 0]

    y_pred_inv = ds["scaler"].inverse_y(y_pred)
    y_true_inv = ds["scaler"].inverse_y(y_true)

    mae = np.mean(np.abs(y_pred_inv - y_true_inv))
    da  = direction_accuracy(y_true_inv, y_pred_inv) * 100
    print(f"\n── HYDRA Engine Test  MAE=${mae:,.2f}  DirAcc={da:.1f}%")

    model.save(str(CKPT_DIR / "hydra_final.keras"))
    return model, {"mae": mae, "dir_acc": da}


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="KAT Training Orchestrator")
    parser.add_argument(
        "--model", type=str, default="all",
        choices=["all", "alpha", "titan", "causal_base", "causal_lion", "causal_tiger", "hydra"],
        help="Which model(s) to train",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch",  type=int, default=128)
    parser.add_argument("--candles", type=int, default=60_000,
                        help="Number of real-world candles to fetch from Delta Exchange")
    parser.add_argument("--symbol", default=".DEXBTUSD", help="Symbol to fetch")
    parser.add_argument("--timeframe", default="1m", help="Timeframe (1m, 5m, 15m, 1h, etc.)")
    parser.add_argument("--finetune", action="store_true", help="Fine-tune existing model instead of scratch")
    args = parser.parse_args()

    configure_gpu()
    CKPT_DIR.mkdir(exist_ok=True)
    DATA_DIR.mkdir(exist_ok=True)

    # ── Fetch Live Data (with Caching) ─────────────────────────────────────────
    CACHE_PATH = DATA_DIR / f"{args.symbol}_{args.timeframe}_history_{args.candles}.parquet"
    
    if CACHE_PATH.exists():
        print(f"\n📂 Loading cached dataset from: {CACHE_PATH}")
        df = pd.read_parquet(CACHE_PATH)
    else:
        print(f"\n📡 No cache found. Fetching {args.candles:,} live {args.symbol} candles ({args.timeframe}) from Delta Exchange...")
        df = fetch_live_kat_data(symbol=args.symbol, n_candles=args.candles, timeframe=args.timeframe)
        # Save cache for next time
        df.to_parquet(CACHE_PATH)
        print(f"💾 Dataset cached to: {CACHE_PATH}")

    # If training HYDRA, we need 360 window
    base_window = 360 if "hydra" in args.model or args.model == "all" else 150
    print(f"\nBuilding base dataset (context={base_window})...")
    ds = build_dataset(
        df,
        context_window=base_window,
        forecast_steps=1,
        scaler_save_path=str(CKPT_DIR / "scaler_base.pkl"),
    )

    all_metrics = {}

    # ── Train requested models ───────────────────────────────────────────────
    targets = [args.model] if args.model != "all" else [
        "alpha", "titan", "causal_base", "causal_lion", "causal_tiger", "hydra"
    ]

    for target in targets:
        t0 = time.time()

        if target == "alpha":
            _, m = train_alpha(ds, epochs=args.epochs, batch_size=args.batch)
            all_metrics["ALPHA Engine"] = m

        elif target == "titan":
            _, m = train_titan(ds, epochs=args.epochs, batch_size=max(32, args.batch // 4))
            all_metrics["TITAN Engine"] = m

        elif target == "causal_base":
            _, m = train_causal(ds, variant="base", epochs=args.epochs, batch_size=args.batch)
            all_metrics["CAUSAL Base"] = m

        elif target == "causal_lion":
            _, m = train_causal(ds, variant="lion", epochs=args.epochs, batch_size=64)
            all_metrics["CAUSAL Lion"] = m

        elif target == "causal_tiger":
            _, m = train_causal(ds, variant="tiger", epochs=args.epochs, batch_size=32)
            all_metrics["CAUSAL Tiger"] = m

        elif target == "hydra":
            _, m = train_hydra(ds, epochs=args.epochs, batch_size=args.batch, finetune=args.finetune)
            all_metrics["HYDRA Engine"] = m

        elapsed = time.time() - t0
        print(f"  ⏱  {target} trained in {elapsed:.1f}s")

    # ── Summary ──────────────────────────────────────────────────────────────
    print("\n" + "═"*60)
    print("  FINAL RESULTS SUMMARY")
    print("═"*60)
    for name, m in all_metrics.items():
        parts = [f"MAE=${m.get('mae', 0):,.2f}"]
        if "dir_acc" in m:
            parts.append(f"DirAcc={m['dir_acc']:.1f}%")
        if "rmse" in m:
            parts.append(f"RMSE=${m['rmse']:,.2f}")
        print(f"  {name:<20} {' | '.join(parts)}")

    print(f"\nCheckpoints saved to: {CKPT_DIR}")


if __name__ == "__main__":
    main()