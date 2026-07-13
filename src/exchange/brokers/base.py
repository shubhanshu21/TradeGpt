"""Broker-agnostic interface. Every strategy/backtest/paper/live component
talks to THIS interface only - never to kiteconnect/upstox/dhan SDKs
directly - so switching brokers is a one-line config change.

Canonical interval strings used everywhere in this codebase:
    "minute", "3minute", "5minute", "15minute", "60minute", "day"
Each adapter is responsible for mapping these to whatever its own API
expects (and resampling client-side if the broker doesn't natively
support a given bucket size).

Candle DataFrames returned by historical_candles() always have exactly
these columns: date, open, high, low, close, volume.
"""
from abc import ABC, abstractmethod

import pandas as pd


class Broker(ABC):
    name: str

    @abstractmethod
    def historical_candles(self, symbol: str, interval: str,
                            start, end) -> pd.DataFrame:
        """Returns OHLCV candles for `symbol` between start and end (datetimes)."""
        raise NotImplementedError

    @abstractmethod
    def ltp(self, symbols: list) -> dict:
        """Returns {symbol: last_traded_price} for a list of tradingsymbols."""
        raise NotImplementedError

    @abstractmethod
    def place_order(self, symbol: str, direction: int, quantity: int,
                     order_type: str = "MARKET", product: str = "INTRADAY",
                     price: float = 0.0) -> str:
        """direction: 1 = buy, -1 = sell. Returns broker order id."""
        raise NotImplementedError

    @abstractmethod
    def positions(self) -> list:
        """Returns current open positions as broker-native dicts."""
        raise NotImplementedError

    def place_gtt_oco(self, symbol: str, direction: int, quantity: int,
                       entry_price: float, stop_price: float, target_price: float) -> str:
        """Optional: place a broker-side one-cancels-other (stop-loss +
        target) that stays live on the BROKER'S servers, independent of
        this process running. Only meaningful for multi-day swing
        positions - an intraday position is only at risk for hours, but a
        swing position can sit open for weeks, so relying solely on this
        process polling prices is a real gap if it crashes or loses
        connectivity.

        Not every broker's API supports this the same way (or at all) -
        the default here raises NotImplementedError; live_trading/executor.py
        always falls back to its own price-polling regardless of whether
        this succeeds, so a broker without support is a reduced safety
        margin, not a hard blocker.
        """
        raise NotImplementedError(f"{self.name} adapter has no broker-side GTT/OCO support wired up")
