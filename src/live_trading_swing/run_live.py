"""Starts SWING LIVE trading with the trained neural network - real orders,
real money.

Do not run this until swing paper trading (swing_paper_trading/) has been
running for at least several weeks, and you've reviewed
logs/swing_paper_trades.csv and are comfortable with its real behavior, not
just the original backtest numbers.

Usage:
    python3 -m live_trading_swing.run_live
"""
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent.parent  # kat/
sys.path.insert(0, str(ROOT / "src"))

from exchange.brokers.factory import get_broker  # noqa: E402
from live_trading_swing.executor import SwingLiveExecutor  # noqa: E402
from strategies.swing.ml_strategy import MLSwingStrategy  # noqa: E402


def load_config():
    with open(ROOT / "config" / "settings.yaml") as f:
        return yaml.safe_load(f)


def main():
    cfg = load_config()
    if cfg["broker"]["name"] == "csv":
        print("broker.name is 'csv' (backtest-only, static data). Live trading "
              "needs a real broker - set broker.name to zerodha, upstox, or dhan first.")
        sys.exit(1)

    broker = get_broker(cfg)
    strategies = [MLSwingStrategy()]
    symbols = cfg["universe"]["symbols"]

    executor = SwingLiveExecutor(cfg, strategies, broker, symbols)
    executor.run()


if __name__ == "__main__":
    main()
