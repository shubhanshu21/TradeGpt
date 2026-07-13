#!/usr/bin/env python3
"""
Sovereign Kraken Auto-Run — Equity Swing Edition
=================================================
Single entry point for the equity swing trading pipeline (converted from a
crypto perpetual-futures system — see src/config/sovereign_config.py's
module docstring for the full context). Neural-network only — no rule-based
strategies.

Modes:
  train    → train one HYDRA model per symbol on real equity data
  backtest → backtest the trained model with real Zerodha delivery costs
  paper    → paper trade (no real orders) — needs a real broker (not csv)
  live     → LIVE trade — REAL orders, REAL money. Gated, requires typed confirmation.

Usage:
    python auto_run.py train --epochs 300
    python auto_run.py backtest
    python auto_run.py paper
    python auto_run.py live
"""

import sys
import subprocess
import argparse
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))


def mode_train(args):
    """Full training pipeline — thin wrapper around train.py."""
    cmd = [
        sys.executable, str(ROOT / "train.py"),
        "--epochs", str(args.epochs),
        "--batch",  str(args.batch),
    ]
    if args.symbols:
        cmd += ["--symbols"] + args.symbols
    if args.context_window:
        cmd += ["--context_window", str(args.context_window)]
    if args.forecast_steps:
        cmd += ["--forecast_steps", str(args.forecast_steps)]
    if args.resume:
        cmd.append("--resume")

    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def mode_backtest(args):
    cmd = [sys.executable, "-m", "swing_backtest.run_swing_backtest"]
    if args.refresh:
        cmd.append("--refresh")
    subprocess.run(cmd, check=True, cwd=str(ROOT / "src"))


def mode_paper(args):
    cmd = [sys.executable, "-m", "swing_paper_trading.run_swing_paper"]
    subprocess.run(cmd, check=True, cwd=str(ROOT / "src"))


def mode_live(args):
    cmd = [sys.executable, "-m", "live_trading_swing.run_live"]
    subprocess.run(cmd, check=True, cwd=str(ROOT / "src"))


def main():
    try:
        from core.hydra import init_kraken_hardware
        init_kraken_hardware()
    except Exception as e:
        print(f"⚠️ Could not initialize hardware/thread limits: {e}")

    parser = argparse.ArgumentParser(
        description="Equity Swing Trading Pipeline — Master Entry Point",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="mode", required=True)

    p_train = sub.add_parser("train", help="Train one HYDRA model per symbol")
    p_train.add_argument("--epochs", type=int, default=300)
    p_train.add_argument("--batch",  type=int, default=32)
    p_train.add_argument("--symbols", nargs="+", default=None)
    p_train.add_argument("--context_window", type=int, default=None)
    p_train.add_argument("--forecast_steps", type=int, default=None)
    p_train.add_argument("--resume", action="store_true")

    p_bt = sub.add_parser("backtest", help="Backtest the trained model with real Zerodha delivery costs")
    p_bt.add_argument("--refresh", action="store_true")

    sub.add_parser("paper", help="Paper trade (virtual orders, real broker prices)")
    sub.add_parser("live", help="LIVE trade — REAL orders, REAL money (gated)")

    args = parser.parse_args()

    if args.mode == "train":
        mode_train(args)
    elif args.mode == "backtest":
        mode_backtest(args)
    elif args.mode == "paper":
        mode_paper(args)
    elif args.mode == "live":
        mode_live(args)


if __name__ == "__main__":
    main()
