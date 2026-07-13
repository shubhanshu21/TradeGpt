"""Starts swing paper trading with the trained neural network (ml_swing).

Usage:
    python3 -m swing_paper_trading.run_swing_paper
"""
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent.parent  # kat/
sys.path.insert(0, str(ROOT / "src"))

from exchange.brokers.factory import get_broker  # noqa: E402
from swing_paper_trading.engine import SwingPaperTradingEngine  # noqa: E402
from strategies.swing.ml_strategy import MLSwingStrategy  # noqa: E402


def load_config():
    with open(ROOT / "config" / "settings.yaml") as f:
        return yaml.safe_load(f)


def main():
    cfg = load_config()
    if cfg["broker"]["name"] == "csv":
        print("broker.name is 'csv' (backtest-only, static data). Swing paper trading "
              "needs live prices - set broker.name to zerodha, upstox, or dhan first.")
        sys.exit(1)

    broker = get_broker(cfg)
    strategies = [MLSwingStrategy()]
    symbols = cfg["universe"]["symbols"]

    engine = SwingPaperTradingEngine(cfg, strategies, broker, symbols)
    engine.run_forever()


if __name__ == "__main__":
    main()
