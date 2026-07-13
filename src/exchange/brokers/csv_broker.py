"""Backtest-only "broker" backed by static CSVs (e.g. a Kaggle Nifty50
minute-data dump) instead of a live broker API. Use this when you want to
backtest without a paid Kite Connect subscription or before you've set up
Upstox/Dhan credentials.

It cannot provide live prices or place orders - only historical_candles()
works. Paper trading and live trading need a real broker (config broker.name:
zerodha|upstox|dhan) since they require live data and real order placement.

Expects one CSV per symbol at cache/kaggle/<SYMBOL>.csv with columns
date,open,high,low,close,volume (1-minute native resolution) - produced by
running `python3 data/import_kaggle_data.py` after downloading a dataset.
"""
from pathlib import Path

import pandas as pd

from exchange.brokers.base import Broker
from exchange.resample import resample_ohlcv


class CsvBroker(Broker):
    name = "csv"

    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)

    def _load_symbol(self, symbol: str) -> pd.DataFrame:
        path = self.data_dir / f"{symbol}.csv"
        if not path.exists():
            raise FileNotFoundError(
                f"No imported data for '{symbol}' at {path}. Download a Kaggle "
                f"Nifty50 minute-data dataset, place the raw CSV(s) in "
                f"data/kaggle_raw/, then run: python3 data/import_kaggle_data.py")
        return pd.read_csv(path, parse_dates=["date"])

    def historical_candles(self, symbol: str, interval: str, start, end) -> pd.DataFrame:
        df = self._load_symbol(symbol)
        df = df[(df["date"] >= start) & (df["date"] <= end)].reset_index(drop=True)
        if interval not in ("minute",) and not df.empty:
            df = resample_ohlcv(df, interval)
        return df

    def ltp(self, symbols: list) -> dict:
        raise NotImplementedError(
            "CsvBroker is backtest-only (static historical data) - it has no "
            "live prices. Set broker.name to zerodha/upstox/dhan for paper/live trading.")

    def place_order(self, symbol: str, direction: int, quantity: int,
                     order_type: str = "MARKET", product: str = "INTRADAY",
                     price: float = 0.0) -> str:
        raise NotImplementedError("CsvBroker is backtest-only - it cannot place orders.")

    def positions(self) -> list:
        raise NotImplementedError("CsvBroker is backtest-only - it has no positions.")
