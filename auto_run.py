#!/usr/bin/env python3
"""
KAT Auto-Run
============
Single entry point for the entire KAT pipeline.

Modes:
  train   → generate data, preprocess, train all models
  predict → load latest checkpoint, run prediction on recent data
  serve   → launch FastAPI inference server
  demo    → quick end-to-end demo (small dataset, 5 epochs)

Usage:
    python auto_run.py train --model all --epochs 50
    python auto_run.py train --model causal_tiger --epochs 30
    python auto_run.py predict --model causal_tiger --steps 60
    python auto_run.py serve --port 8000
    python auto_run.py demo
"""

import sys, time
import numpy as np
import argparse
from pathlib import Path

ROOT        = Path(__file__).parent
DATA_DIR    = ROOT / "data"
MODEL_DIR   = ROOT / "src/architectures"
LOG_DIR     = ROOT / "logs"
SAVED_MODELS = ROOT / "models"

sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(MODEL_DIR))


# ──────────────────────────────────────────────────────────────────────────────
# MODES
# ──────────────────────────────────────────────────────────────────────────────

def mode_train(args):
    """Full training pipeline."""
    import subprocess
    cmd = [
        sys.executable, str(ROOT / "train.py"),
        "--model",    args.model,
        "--epochs",   str(args.epochs),
        "--batch",    str(args.batch),
        "--candles",  str(args.candles),
        "--timeframe", args.timeframe,
    ]
    print(f"Running: {' '.join(cmd)}")
    if hasattr(args, "symbol"):
        cmd.extend(["--symbol", args.symbol])
    if hasattr(args, "finetune") and args.finetune:
        cmd.append("--finetune")
    if hasattr(args, "resume") and args.resume:
        cmd.append("--resume")
    
    subprocess.run(cmd, check=True)


def mode_predict(args):
    """Load a trained model and run prediction on fresh synthetic data."""
    import numpy as np
    import tensorflow as tf
    import keras
    from data.preprocess import build_dataset_streaming as build_dataset, KATScaler, build_feature_cols
    from exchange.fetch_data     import fetch_live_kat_data

    model_file = SAVED_MODELS / "hydra_best.keras" if "hydra" in args.model else SAVED_MODELS / f"{args.model}_best.keras"
    scaler_p   = SAVED_MODELS / "scaler_base.pkl"

    if not model_file.exists():
        print(f"❌ Model file not found: {model_file}")
        return
    if scaler_p.exists():
        scaler = KATScaler.load(str(scaler_p))
    else:
        scaler = None # Phase 2 uses DLS
    print(f"📡 Fetching live data for {args.symbol} {args.model} prediction...")
    df = fetch_live_kat_data(symbol=args.symbol, n_candles=300, timeframe=args.timeframe)
    if df is None or len(df) < 120:
        print("❌ Not enough data for prediction.")
        return

    from data.preprocess import compute_indicators
    df = compute_indicators(df)
    features = build_feature_cols()
    data = df[features].values.astype("float32")
    
    # Phase 2: Dynamic Local Scaling (DLS)
    # We use the raw data directly. The scale happens inside the model logic.
    seed = data[-120:] 
    # CTX_WIN = 120

    print(f"Loading {model_file.name}...")

    custom_objs = {}
    if "hydra" in args.model:
        from core.hydra import (HydraBlock, GatedMoE, 
                                RMSNorm, TurboQuant, SwiGLU,
                                SovereignLoss, CertaintyMetric, SovereignAccuracy, MLALayer)
        custom_objs = {
            "HydraBlock":         HydraBlock,
            "GatedMoE":           GatedMoE,
            "MLALayer":           MLALayer,
            "RMSNorm":            RMSNorm,
            "TurboQuant":         TurboQuant,
            "SwiGLU":             SwiGLU,
            "SovereignLoss":      SovereignLoss,
            "CertaintyMetric":    CertaintyMetric,
            "SovereignAccuracy":  SovereignAccuracy,
        }
    if "hydra" in args.model:
        from core.hydra import build_kraken
        model = build_kraken(n_features=42)
        model.load_weights(str(model_file))
        print(f"✅ Weights loaded from {model_file.name}")

    # ── Predict ──────────────────────────────────────────────────────────────
    # ── Phase 2: Dynamic Local Scaling (DLS) ─────────────────────────────────
    if "hydra" in args.model:
        # 1. Calculate local stats for the context window
        local_mean = seed.mean(axis=0)
        local_std  = seed.std(axis=0) + 1e-8
        
        # 2. Scale locally
        seed_scaled = (seed - local_mean) / local_std
        inp = seed_scaled[np.newaxis]
        
        # 3. Predict
        outputs = model(inp, training=False)
        pred_all = outputs[0].numpy()[0] # (16, 3)
        
        # 4. Extract Trajectory
        p_anchor = pred_all[0, 0]
        p_future = pred_all[1:, 0]
        
        # Unscaled USD Price for reporting
        last_known = (seed[-1, 3]) # Seed was already raw before local scaling? 
        # Wait, auto_run data is raw until we scale it.
        # So seed is raw data.
        last_known_usd = seed[-1, 3]
        
        # Unscaled deltas (USD) = (p_future - p_anchor) * local_std[close]
        t_close = 3 
        usd_deltas = (p_future - p_anchor) * local_std[t_close]
        
        print(f"\nLast known close: ${last_known_usd:,.2f}")
        print(f"Predicted MTP-15 trajectory ({len(p_future)} steps):")
        
        try:
            import matplotlib.pyplot as plt
            plot_dir = LOG_DIR / "plots"
            plot_dir.mkdir(parents=True, exist_ok=True)
            # Manual inverse X[3] for last 30 historical close prices
            # Manual inverse prices for plotting
            hist_close = seed[-30:, 3]
            
            # Forecast visual starts at last known and applies USD deltas
            forecast_visual = [last_known_usd]
            for d in usd_deltas:
                forecast_visual.append(forecast_visual[-1] + d)
            
            forecast_x = range(len(hist_close) - 1, len(hist_close) + len(usd_deltas))
            
            plt.figure(figsize=(10, 6))
            plt.plot(range(len(hist_close)), hist_close, label="History", color="blue", marker="o", markersize=3)
            plt.plot(forecast_x, forecast_visual, label="Forecast (Deep-Predator V10.6)", color="green", linestyle="--", marker="x", markersize=4)
            
            plt.title(f"KAT Prediction: {args.model} {args.symbol} MTP-15")
            plt.xlabel("Minutes")
            plt.ylabel("Price ($)")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plot_file = plot_dir / f"{args.model}_{time.strftime('%H%M%S')}.png"
            plt.savefig(plot_file)
            print(f"📈 Visual plot saved to: {plot_file}")
            plt.close()
            
            # Update the CLI print-out to show real dollar deltas
            p_curve = usd_deltas 
            v_curve = pred_all[1:, 1]
            q_curve = pred_all[1:, 2]
        except Exception as e:
            print(f"! Could not generate visual plot: {e}")

        for i, p in enumerate(p_curve, 1):
            is_above = "↑" if p > 0 else "↓"
            print(f"  +{i:3d}min  Delta ${p:+.2f}  {is_above}  |  Vol: {v_curve[i-1]:.4f}  |  Flow: {q_curve[i-1]:.4f}")

    elif "causal" in args.model:
        if "causal" in args.model: from causal import CausalModel as GeneratorModel
        else: from hydra import Hydra as GeneratorModel
        traj = model.generate(seed, steps=args.steps, scaler=scaler)
        last_known = scaler.inverse_y(seed[-1:, 3:4].ravel())[0]
        print(f"\nLast known close: ${last_known:,.2f}")
        print(f"Predicted trajectory ({args.steps} steps):")

        # ── Visual Plotting ──────────────────────────────────────────────────
        try:
            import matplotlib.pyplot as plt
            plot_dir = LOG_DIR / "plots"
            plot_dir.mkdir(parents=True, exist_ok=True)
            
            # Get last 30 historical points for context
            hist_close = scaler.inverse_y(seed[-30:, 3:4].ravel())
            
            plt.figure(figsize=(10, 6))
            plt.plot(range(len(hist_close)), hist_close, label="History", color="blue", marker="o", markersize=3)
            
            # Forecast starts from the last historical point (connects history to forecast)
            forecast_x = range(len(hist_close) - 1, len(hist_close) + len(traj))
            plt.plot(forecast_x, [hist_close[-1]] + list(traj), label="Forecast", color="orange", linestyle="--", marker="x", markersize=4)
            
            plt.title(f"KAT Prediction: {args.model}")
            plt.xlabel("Minutes")
            plt.ylabel("Price ($)")
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plot_file = plot_dir / f"{args.model}_{time.strftime('%H%M%S')}.png"
            plt.savefig(plot_file)
            print(f"📈 Visual plot saved to: {plot_file}")
            plt.close()
            
        except Exception as e:
            print(f"! Could not generate visual plot: {e}")

        # Text output
        for i, p in enumerate(traj, 1):
            arrow = "↑" if p > (traj[i-2] if i > 1 else last_known) else "↓"
            print(f"  +{i:3d}min  ${p:,.2f}  {arrow}")
    else:
        inp = seed[np.newaxis]
        pred_s = model.predict(inp, verbose=0)[0]
        if hasattr(pred_s, "__len__"):
            pred_s = pred_s[-1]
        pred = scaler.inverse_y(np.array([float(pred_s)]))[0]
        last_known = scaler.inverse_y(seed[-1:, 3:4].ravel())[0]
        direction = "UP ↑" if pred > last_known else "DOWN ↓"
        print(f"\nLast known close : ${last_known:,.2f}")
        print(f"Predicted close  : ${pred:,.2f}  ({direction})")
        delta = pred - last_known
        sign = "+" if delta >= 0 else "-"
        print(f"Delta            : {sign}${abs(delta):,.2f}")


def mode_serve(args):
    """Launch FastAPI server."""
    try:
        import uvicorn
        print(f"Starting KAT API server on port {args.port}...")
        uvicorn.run(
            "src.api.serve:app",
            host="0.0.0.0",
            port=args.port,
            reload=False,
            app_dir=str(ROOT),
        )
    except ImportError:
        print("uvicorn not installed. Run: pip install uvicorn fastapi")


def mode_demo(args):
    """Quick end-to-end demo: train KAT 1.3 for 5 epochs, predict."""
    import subprocess

    print("=" * 60)
    print("  KAT DEMO — Quick end-to-end pipeline test")
    print("=" * 60)

    print("\n[1/2] Training ALPHA (5 epochs, 2,000 candles, LIVE)...")
    subprocess.run([
        sys.executable, str(ROOT / "train.py"),
        "--model", "alpha",
        "--epochs", "5",
        "--batch", "128",
        "--candles", "2000",
        "--live"
    ], check=True)

    print("\n[2/2] Running prediction...")
    sys.argv = ["auto_run.py", "predict", "--model", "alpha"]
    predict_args = argparse.Namespace(model="alpha", steps=1)
    mode_predict(predict_args)

    print("\n✓ Demo complete! Run `python auto_run.py train` for full training.")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="KAT Predictive Engine — Master Entry Point",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    sub = parser.add_subparsers(dest="mode", required=True)

    # ── train ─────────────────────────────────────────────────────────────────
    p_train = sub.add_parser("train", help="Train KAT models")
    p_train.add_argument("--model",   default="all",
        choices=["all","alpha","titan","causal_base","causal_lion","causal_tiger","hydra"])
    p_train.add_argument("--epochs",  type=int, default=50)
    p_train.add_argument("--batch",   type=int, default=128)  # V5.0 calibrated
    p_train.add_argument("--candles", type=int, default=120_000)
    p_train.add_argument("--symbol",  default="BTCUSD")
    p_train.add_argument("--timeframe", default="15m", help="Timeframe (15m recommended for Phase 5)")
    p_train.add_argument("--finetune", action="store_true", help="Fine-tune existing model")
    p_train.add_argument("--resume", action="store_true", help="Resume from the latest 'Decade Backup' Epoch checkpoint")

    # ── predict ───────────────────────────────────────────────────────────────
    p_pred = sub.add_parser("predict", help="Run prediction with a trained model")
    p_pred.add_argument("--model", default="causal_base",
        choices=["alpha", "titan", "causal_base", "causal_lion", "causal_tiger", "hydra"])
    p_pred.add_argument("--steps", type=int, default=60,
        help="Forecast steps (CAUSAL/HYDRA only)")
    p_pred.add_argument("--timeframe", default="15m", help="Timeframe (must match training)")
    p_pred.add_argument("--symbol",  default="BTCUSD")
    p_pred.add_argument("--live", action="store_true", default=True,
        help="Fetch live data for prediction (default: True)")

    # ── serve ─────────────────────────────────────────────────────────────────
    p_srv = sub.add_parser("serve", help="Launch FastAPI inference server")
    p_srv.add_argument("--port", type=int, default=8000)

    # ── trade ─────────────────────────────────────────────────────────────────
    p_trade = sub.add_parser("trade", help="Live autonomous Sandbox trading")
    p_trade.add_argument("--model",  default="hydra",
        choices=["alpha", "titan", "causal_base", "causal_lion", "causal_tiger", "hydra"])
    p_trade.add_argument("--symbol", default="BTCUSD")
    p_trade.add_argument("--size",   type=int, default=1)
    p_trade.add_argument("--thresh", type=float, default=0.05)
    p_trade.add_argument("--timeframe", default="15m", help="Timeframe (must match training)")
    p_trade.add_argument("--cert_thresh", type=float, default=0.70,
        help="Certainty threshold (0-1). Only trade signals above this. "
             "Higher = fewer trades but higher accuracy. Recommended: 0.70-0.85")

    args = parser.parse_args()

    MODEL_DIR = ROOT / "src/core"

    if   args.mode == "train":   mode_train(args)
    elif args.mode == "predict": mode_predict(args)
    elif args.mode == "trade":
        from trading import live_trader
        live_trader.MODEL_NAME    = args.model
        live_trader.SYMBOL        = args.symbol
        live_trader.SIZE          = args.size
        live_trader.THRESHOLD     = args.thresh
        live_trader.TIMEFRAME     = args.timeframe
        live_trader.CERT_THRESHOLD = args.cert_thresh  # High-conviction filter
        print(f"⚖️  Certainty Filter: Only trading signals with >{args.cert_thresh*100:.0f}% conviction")
        live_trader.run_pilot()
    elif args.mode == "serve":   mode_serve(args)
    elif args.mode == "demo":    mode_demo(args)


if __name__ == "__main__":
    main()