#!/usr/bin/env python3
"""
Sovereign Kraken Auto-Run
=========================
Single entry point for the Sovereign Kraken pipeline. Hydra is the only
architecture in the codebase — there is no `src/architectures/` directory
and no alternate model to select, so this wraps train.py and live_trader.py
directly rather than exposing dead architecture choices.

For predictions, use the dashboard (src/api/prediction_viewer/app.py) or
the scripts in src/evaluation/ (certainty_audit.py, backtest_checkup.py, etc).

Modes:
  train → generate data, preprocess, train the Hydra model
  trade → live autonomous Sandbox trading

Usage:
    python auto_run.py train --epochs 300 --timeframe 1h --candles 31430
    python auto_run.py trade --symbol BTCUSD
"""

import sys
import argparse
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))

try:
    from config.sovereign_config import CERT_THRESHOLD
except ImportError:
    CERT_THRESHOLD = 0.85


# ──────────────────────────────────────────────────────────────────────────────
# MODES
# ──────────────────────────────────────────────────────────────────────────────

def mode_train(args):
    """Full training pipeline — thin wrapper around train.py."""
    import subprocess
    cmd = [
        sys.executable, str(ROOT / "train.py"),
        "--epochs",   str(args.epochs),
        "--batch",    str(args.batch),
        "--candles",  str(args.candles),
        "--timeframe", args.timeframe,
        "--symbol",   args.symbol,
    ]
    if args.context_window:
        cmd.extend(["--context_window", str(args.context_window)])
    if args.forecast_steps:
        cmd.extend(["--forecast_steps", str(args.forecast_steps)])
    if args.resume:
        cmd.append("--resume")

    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main():
    # ── Thread Restrictions & Hardware Setup ──
    try:
        from core.hydra import init_kraken_hardware
        init_kraken_hardware()
    except Exception as e:
        print(f"⚠️ Could not initialize hardware/thread limits: {e}")

    parser = argparse.ArgumentParser(
        description="KAT Predictive Engine — Master Entry Point",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    sub = parser.add_subparsers(dest="mode", required=True)

    # ── train ─────────────────────────────────────────────────────────────────
    p_train = sub.add_parser("train", help="Train the Hydra model")
    p_train.add_argument("--epochs",  type=int, default=300)
    p_train.add_argument("--batch",   type=int, default=32)
    p_train.add_argument("--candles", type=int, default=31430)
    p_train.add_argument("--symbol",  default="BTCUSD")
    p_train.add_argument("--timeframe", default="1h")
    p_train.add_argument("--context_window", type=int, default=None)
    p_train.add_argument("--forecast_steps", type=int, default=None)
    p_train.add_argument("--resume", action="store_true", help="Resume from the latest checkpoint")

    # ── trade ─────────────────────────────────────────────────────────────────
    p_trade = sub.add_parser("trade", help="Live autonomous Sandbox trading")
    p_trade.add_argument("--symbol", default="BTCUSD")
    p_trade.add_argument("--size",   type=int, default=1)
    p_trade.add_argument("--thresh", type=float, default=0.05)
    p_trade.add_argument("--timeframe", default="1h", help="Timeframe (must match training)")
    p_trade.add_argument("--cert_thresh", type=float, default=CERT_THRESHOLD,
        help="Certainty threshold (0-1). Only trade signals above this. "
             "Higher = fewer trades but higher accuracy. Recommended: 0.70-0.85")

    args = parser.parse_args()

    if args.mode == "train":
        mode_train(args)
    elif args.mode == "trade":
        from trading import live_trader
        live_trader.SYMBOL        = args.symbol
        live_trader.SIZE          = args.size
        live_trader.THRESHOLD     = args.thresh
        live_trader.TIMEFRAME     = args.timeframe
        live_trader.CERT_THRESHOLD = args.cert_thresh  # High-conviction filter
        print(f"⚖️  Certainty Filter: Only trading signals with >{args.cert_thresh*100:.0f}% conviction")
        live_trader.run_pilot()


if __name__ == "__main__":
    main()
