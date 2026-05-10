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
from config.sovereign_config import FEE_RATE, INITIAL_WALLET_USD, POSITION_SIZE_PCT


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

    def _update_dashboard(self, logs=None):
        logs = logs or {}
        # Fetch current training metrics if available
        v_acc = logs.get("prediction_dir_acc", 0.0)
        cert  = logs.get("certainty_certainty", 0.0)
        ts    = datetime.now().strftime("%H:%M:%S")
        
        # ── Automated Sovereign Benchmarking (V11.1) ────────────────────────
        try:
            import json
            from data.preprocess import compute_indicators, build_feature_cols
            from exchange.fetch_data import fetch_live_kat_data
            
            # Access fee_rate from args if possible, else default
            fee_rate = getattr(self, 'fee_rate', 0.0012)
            
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
                traj_scaled = outputs[0].numpy()[:, -1, 0] # Scaled price at end
                certs       = np.mean(outputs[1].numpy(), axis=1) # Avg certainty
                
                # ── V11.2: Neural De-Scaling Patch ──────────────────────────
                close_means = np.array([data[i:i+ctx, 3].mean() for i in indices])
                close_stds  = np.array([data[i:i+ctx, 3].std() + 1e-8 for i in indices])
                traj_usd    = (traj_scaled * close_stds) + close_means
                
                # Normalize certainty for bench (80-100% range)
                c_pct = (certs - certs.min()) / (certs.max() - certs.min() + 1e-9) * 100
                
                from config.sovereign_config import INITIAL_WALLET_USD
                roi_data = {"tiers": {}, "last_update": ts, "note": f"FEE GATE: {fee_rate*100:.2f}%"}
                pos_size = INITIAL_WALLET_USD
                
                # ── V11.6: Sovereign Recapitalization Filter ─────────────────
                reset_file = LOG_DIR / "sim_reset.txt"
                reset_ts = None
                if reset_file.exists():
                    try:
                        reset_ts = pd.to_datetime(reset_file.read_text().strip()).tz_localize(None)
                    except: pass

                for th in [80, 85, 90]:
                    mask = c_pct >= th
                    if reset_ts:
                        t_stamps = pd.to_datetime(df.index[np.array(indices) + ctx - 1]).tz_localize(None)
                        mask = mask & (t_stamps.values >= reset_ts)
                    n_t  = int(mask.sum())
                    if n_t > 0:
                        e_p = raw_prices[np.array(indices)[mask] + ctx - 1]
                        directions = np.sign(traj_usd[mask] - e_p)
                        gross = float((directions * (usd_diffs[mask] / e_p) * pos_size).sum())
                        fees = float(n_t * (pos_size * fee_rate))
                        roi_data["tiers"][str(th)] = {"trades": n_t, "net": gross - fees}
                
                # 3. Save detailed recent trades for dashboard feed
                recent_trades = []
                mask80 = c_pct >= 80
                if mask80.any():
                    t_indices = np.array(indices)[mask80]
                    t_traj_usd = traj_usd[mask80]
                    t_usd = usd_diffs[mask80]
                    for i in range(max(0, len(t_indices)-500), len(t_indices)):
                        idx = t_indices[i]
                        entry_p = raw_prices[idx + ctx - 1]
                        price_move_pct = (t_usd[i] / entry_p)
                        # V11.5: Tactical Deadzone Calibration
                        # Eliminate 'Noise-Driven Shorts' during Epoch 1
                        last_scaled = (entry_p - close_means[i]) / close_stds[i]
                        prediction_delta = traj_scaled[i] - last_scaled
                        
                        # Only strike if the predicted move is greater than 0.1% volatility
                        if abs(prediction_delta) < 0.001: 
                            side = "HOLD"
                        else:
                            side = "LONG" if prediction_delta > 0 else "SHORT"
                        
                        if side == "HOLD": continue
                        
                        # Apply Sovereign Reset to Feed
                        t_stamp = pd.to_datetime(df.index[idx + ctx - 1]).tz_localize(None)
                        if reset_ts and t_stamp < reset_ts: continue
                        
                        # Bug Fix #1: Invert P&L for SHORT trades
                        raw_ret = price_move_pct if side == "LONG" else -price_move_pct
                        net_ret = raw_ret - fee_rate
                        # Bug Fix #2: Deterministic Strategist (seeded by timestamp+entry)
                        prefixes = ["Neon", "Volt", "Cipher", "Ghost", "Logic", "Vector", "Pulse", "Neural", "Cyber", "Quant", "Delta", "Gamma", "Alpha", "Zenith", "Apex", "Flow"]
                        suffixes = ["Hunter", "Scout", "Tracker", "Oracle", "Sentinel", "Core", "Node", "Gate", "Shell", "Link", "Edge", "Vortex", "Matrix", "System", "Prime", "Zero"]
                        import random, hashlib
                        _seed_str = f"{df.index[idx + ctx - 1].isoformat()}{entry_p}"
                        _seed = int(hashlib.md5(_seed_str.encode()).hexdigest(), 16)
                        _rng = random.Random(_seed)
                        expert = f"{_rng.choice(prefixes)}-{_rng.choice(suffixes)} #{_rng.randint(0, 255):03d}"

                        recent_trades.append({
                            "timestamp": df.index[idx + ctx - 1].isoformat(),
                            "side": side,
                            "entry": float(entry_p),
                            "net_pct": float(net_ret * 100),
                            "expert": expert
                        })
                
                # ── Expert Fight: Live 256-Node Heatmap (V11.2 Real-Time) ──
                X_live = X[-1:] 
                outputs_live = self.model(X_live, training=False)
                # Channel 1 is the 256-expert certainty distribution
                cert_map = outputs_live[1].numpy()[0]
                # Normalize 0-1 for the heatmap visualization
                c_min, c_max = cert_map.min(), cert_map.max()
                cert_norm = (cert_map - c_min) / (c_max - c_min + 1e-9)
                
                roi_data["expert_heatmap"] = cert_norm.tolist()
                
                # Also keep the histogram for the risk Sentinel
                hist_bins = [0, 60, 65, 70, 75, 80, 85, 90, 100.1]
                counts, _ = np.histogram(cert_norm * 100, bins=hist_bins)
                roi_data["expert_fight"] = counts.tolist()

                with open(ROOT / "logs" / "latest_roi.json", "w") as f_json:
                    json.dump(roi_data, f_json, indent=4)
                with open(ROOT / "logs" / "recent_sim_trades.json", "w") as f_trades:
                    json.dump(recent_trades[::-1], f_trades, indent=4)
        except Exception as e:
            pass

    def on_batch_end(self, batch, logs=None):
        pass # V12.1: Removed batch-end updates to maximize training velocity

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        v_acc = logs.get("val_prediction_dir_acc", 0.0)
        cert  = logs.get("val_certainty_certainty", 0.0)
        ts    = datetime.now().strftime("%H:%M:%S")
        
        status = "⚓ LEARNING"
        if v_acc >= 0.54: status = "🏛️ SOVEREIGN"
        elif v_acc >= 0.53: status = "⚡ ALPHA FLOW"
        
        print(f"{ts:<10} | {epoch+1:<5} | {v_acc:<8.4f} | {cert:<10.3f} | {status}")
        self._update_dashboard(logs)


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
    mc = MissionControl()
    mc.fee_rate = args.fee_rate
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            str(CKPT_BEST), monitor="val_loss",
            save_best_only=True, verbose=1),
        keras.callbacks.ModelCheckpoint(
            str(CKPT_DIR / "hydra_checkpoint_E{epoch:03d}.keras"),
            save_freq="epoch", verbose=0),  # Save every epoch for resume support
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=20,  # 20 epochs * ~34h = ~28 days — safe for CPU
            restore_best_weights=True, verbose=1),
        CheckpointPruner(ckpt_dir=CKPT_DIR, keep_n=3),
        mc,
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
    p.add_argument("--fee_rate",  type=float, default=FEE_RATE)
    args = p.parse_args()
    train_kraken(args)