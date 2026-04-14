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
from data.preprocess import build_dataset_streaming as build_dataset, KATScaler
from exchange.fetch_data     import fetch_live_kat_data

# ... (lines 70-100 preserved)

    print(f"Loading {model_file}...")

    # Ensure custom classes are registered by importing their modules
    if   "alpha" in args.model: from core.legacy import alpha
    elif "titan" in args.model: from core.legacy import titan
    elif "causal" in args.model: from core.legacy import causal
    elif "hydra"  in args.model: 
        from core.hydra import (build_kraken, HydraBlock, GatedMoE, LightningAttention,
                                RMSNorm, TurboQuant, SwiGLU,
                                SovereignLoss, CertaintyMetric, SovereignAccuracy)
        custom_objs = {
            "HydraBlock":         HydraBlock,
            "GatedMoE":           GatedMoE,
            "LightningAttention": LightningAttention,
            "RMSNorm":            RMSNorm,
            "TurboQuant":         TurboQuant,
            "SwiGLU":             SwiGLU,
            "SovereignLoss":      SovereignLoss,
            "CertaintyMetric":    CertaintyMetric,
            "SovereignAccuracy":  SovereignAccuracy,
        }
        # V10.6: Enable unsafe deserialization for Lambda certainty aggregation
        model = keras.models.load_model(str(model_file), custom_objects=custom_objs, compile=False, safe_mode=False)

    if "hydra" not in args.model:
        model = keras.models.load_model(str(model_file))

    # ── Predict ──────────────────────────────────────────────────────────────
    if "hydra" in args.model:
        inp = seed[np.newaxis]
        pred = model.predict(inp, verbose=0)[0] # Shape (15, 3)
        p_curve = pred[:, 0]
        v_curve = pred[:, 1]
        q_curve = pred[:, 2]
        
        last_known = scaler.inverse_y(seed[-1:, 3:4].ravel())[0]
        print(f"\nLast known close: ${last_known:,.2f}")
        print(f"Predicted MTP-15 trajectory ({len(p_curve)} steps):")
        
        try:
            import matplotlib.pyplot as plt
            plot_dir = LOG_DIR / "plots"
            plot_dir.mkdir(parents=True, exist_ok=True)
            hist_close = scaler.inverse_y(seed[-30:, 3:4].ravel())
            
            plt.figure(figsize=(10, 6))
            plt.plot(range(len(hist_close)), hist_close, label="History", color="blue", marker="o", markersize=3)
            
            forecast_x = range(len(hist_close) - 1, len(hist_close) + len(p_curve))
            forecast_price = [hist_close[-1]]
            for delta in p_curve:
                forecast_price.append(forecast_price[-1] + float(delta))
                
            plt.plot(forecast_x, forecast_price, label="Forecast (Hydra V4.2)", color="green", linestyle="--", marker="x", markersize=4)
            plt.title(f"KAT Prediction: {args.model} MTP-15")
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
    p_train.add_argument("--timeframe", default="1m", help="Timeframe (1m, 5m, 15m, 1h, etc.)")
    p_train.add_argument("--finetune", action="store_true", help="Fine-tune existing model")
    p_train.add_argument("--resume", action="store_true", help="Resume from the latest 'Decade Backup' Epoch checkpoint")

    # ── predict ───────────────────────────────────────────────────────────────
    p_pred = sub.add_parser("predict", help="Run prediction with a trained model")
    p_pred.add_argument("--model", default="causal_base",
        choices=["alpha", "titan", "causal_base", "causal_lion", "causal_tiger", "hydra"])
    p_pred.add_argument("--steps", type=int, default=60,
        help="Forecast steps (CAUSAL/HYDRA only)")
    p_pred.add_argument("--timeframe", default="1m", help="Timeframe (1m, 5m, 15m, 1h, etc.)")
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
    p_trade.add_argument("--timeframe", default="1m", help="Timeframe (1m, 5m, 15m, 1h, etc.)")
    
    args = parser.parse_args()

    MODEL_DIR = ROOT / "src/core"

    if   args.mode == "train":   mode_train(args)
    elif args.mode == "predict": mode_predict(args)
    elif args.mode == "trade":
        from trading import live_trader
        live_trader.MODEL_NAME = args.model
        live_trader.SYMBOL     = args.symbol
        live_trader.SIZE       = args.size
        live_trader.THRESHOLD  = args.thresh
        live_trader.TIMEFRAME  = args.timeframe
        live_trader.run_pilot()
    elif args.mode == "serve":   mode_serve(args)
    elif args.mode == "demo":    mode_demo(args)


if __name__ == "__main__":
    main()