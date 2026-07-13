"""Backtests the trained neural network (ml_swing) over the configured
universe using daily candles and real Zerodha delivery-style costs.

Usage:
    python3 -m swing_backtest.run_swing_backtest
    python3 -m swing_backtest.run_swing_backtest --refresh
"""
import argparse
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent.parent  # kat/
sys.path.insert(0, str(ROOT / "src"))

from backtest.metrics import compute_metrics  # noqa: E402
from data.fetch_historical import fetch_universe_swing  # noqa: E402
from swing_backtest.engine import SwingBacktestEngine  # noqa: E402
from strategies.swing.ml_strategy import MLSwingStrategy  # noqa: E402

REPORTS_DIR = ROOT / "reports" / "swing"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def load_config():
    with open(ROOT / "config" / "settings.yaml") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh", action="store_true", help="force re-download historical data")
    args = parser.parse_args()

    cfg = load_config()
    print("Loading daily historical data (cached after first run)...\n")
    data = fetch_universe_swing(force_refresh=args.refresh)

    if not data:
        print("No data fetched - check broker credentials / config/settings.yaml universe.")
        sys.exit(1)

    initial_capital = cfg["swing"]["initial_capital"]
    strategy = MLSwingStrategy()
    print(f"Backtesting {strategy.name}...")

    try:
        engine = SwingBacktestEngine(cfg, strategy)
        trades = engine.run(data)
    except FileNotFoundError as exc:
        print(f"\n{exc}")
        sys.exit(1)
    trades.to_csv(REPORTS_DIR / f"trades_{strategy.name}.csv", index=False)

    metrics = compute_metrics(trades, initial_capital)
    summary = pd.DataFrame([metrics]).set_index(pd.Index([strategy.name], name="strategy"))
    summary.to_csv(REPORTS_DIR / "strategy_comparison.csv")

    pd.set_option("display.width", 160)
    pd.set_option("display.max_columns", 20)
    print("\n" + "=" * 100)
    print(f"ML SWING BACKTEST  (universe={len(data)} symbols, "
          f"period={cfg['swing']['start_date']}..{cfg['swing']['end_date']}, "
          f"capital=Rs{initial_capital:,})")
    print("=" * 100)
    print(summary.to_string())

    print(f"\nFull trade log written to reports/swing/trades_{strategy.name}.csv")


if __name__ == "__main__":
    main()
