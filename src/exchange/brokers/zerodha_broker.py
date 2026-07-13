"""Zerodha Kite Connect adapter. Requires a daily access token generated via
scripts/login.py (see that file - Kite tokens expire every day at ~6am IST).
"""
import json
import time
from pathlib import Path

import pandas as pd
from kiteconnect import KiteConnect

from exchange.brokers.base import Broker

ROOT = Path(__file__).resolve().parent.parent.parent
CACHE_DIR = ROOT / "cache"

INTERVAL_MAP = {
    "minute": "minute", "3minute": "3minute", "5minute": "5minute",
    "15minute": "15minute", "60minute": "60minute", "day": "day",
}

CHUNK_DAYS = {"minute": 30, "3minute": 90, "5minute": 90, "15minute": 180, "60minute": 180, "day": 2000}


class ZerodhaBroker(Broker):
    name = "zerodha"

    def __init__(self, api_key: str, access_token: str):
        self.kite = KiteConnect(api_key=api_key)
        self.kite.set_access_token(access_token)
        self._instrument_cache = {}

    def _instrument_token(self, symbol: str, exchange: str = "NSE") -> int:
        if exchange in self._instrument_cache:
            instruments = self._instrument_cache[exchange]
        else:
            cache_file = CACHE_DIR / f"instruments_{exchange}.json"
            cache_file.parent.mkdir(exist_ok=True)
            if cache_file.exists():
                instruments = json.loads(cache_file.read_text())
            else:
                instruments = self.kite.instruments(exchange)
                cache_file.write_text(json.dumps(instruments, default=str))
            self._instrument_cache[exchange] = instruments

        for inst in instruments:
            if inst["tradingsymbol"] == symbol:
                return inst["instrument_token"]
        raise ValueError(f"Symbol {symbol} not found on {exchange}")

    def historical_candles(self, symbol: str, interval: str, start, end) -> pd.DataFrame:
        from datetime import timedelta
        token = self._instrument_token(symbol)
        kite_interval = INTERVAL_MAP[interval]
        chunk = timedelta(days=CHUNK_DAYS.get(interval, 90))

        frames = []
        cursor = start
        while cursor < end:
            chunk_end = min(cursor + chunk, end)
            candles = self.kite.historical_data(token, cursor, chunk_end, kite_interval)
            if candles:
                frames.append(pd.DataFrame(candles))
            cursor = chunk_end + timedelta(days=1)
            time.sleep(0.35)

        if not frames:
            return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume"])

        df = pd.concat(frames, ignore_index=True).drop_duplicates(subset="date")
        return df.sort_values("date").reset_index(drop=True)[
            ["date", "open", "high", "low", "close", "volume"]]

    def ltp(self, symbols: list) -> dict:
        keys = [f"NSE:{s}" for s in symbols]
        quotes = self.kite.ltp(keys)
        return {s: quotes[f"NSE:{s}"]["last_price"] for s in symbols if f"NSE:{s}" in quotes}

    def place_order(self, symbol: str, direction: int, quantity: int,
                     order_type: str = "MARKET", product: str = "INTRADAY",
                     price: float = 0.0) -> str:
        transaction_type = self.kite.TRANSACTION_TYPE_BUY if direction == 1 else self.kite.TRANSACTION_TYPE_SELL
        kite_product = self.kite.PRODUCT_MIS if product == "INTRADAY" else self.kite.PRODUCT_CNC
        kite_order_type = self.kite.ORDER_TYPE_MARKET if order_type == "MARKET" else self.kite.ORDER_TYPE_LIMIT

        order_id = self.kite.place_order(
            variety=self.kite.VARIETY_REGULAR,
            exchange=self.kite.EXCHANGE_NSE,
            tradingsymbol=symbol,
            transaction_type=transaction_type,
            quantity=quantity,
            product=kite_product,
            order_type=kite_order_type,
            price=price if order_type == "LIMIT" else None,
        )
        return order_id

    def positions(self) -> list:
        return self.kite.positions()["net"]

    def place_gtt_oco(self, symbol: str, direction: int, quantity: int,
                       entry_price: float, stop_price: float, target_price: float) -> str:
        """Kite Connect's GTT (Good Till Triggered) OCO order: two trigger
        prices, whichever is hit first fires its matching exit order and
        cancels the other. Lives on Zerodha's servers for up to a year,
        independent of this process - the actual safety net a multi-day
        swing position needs.

        NOT verified against a live Zerodha account in this project (only
        smoke-tested against Upstox so far) - place one manually with a
        throwaway quantity before trusting this for real capital.
        """
        exit_transaction = (self.kite.TRANSACTION_TYPE_SELL if direction == 1
                             else self.kite.TRANSACTION_TYPE_BUY)
        lower, upper = sorted([stop_price, target_price])

        trigger = self.kite.place_gtt(
            trigger_type=self.kite.GTT_TYPE_OCO,
            tradingsymbol=symbol,
            exchange=self.kite.EXCHANGE_NSE,
            trigger_values=[lower, upper],
            last_price=entry_price,
            orders=[
                {
                    "transaction_type": exit_transaction, "quantity": quantity,
                    "order_type": self.kite.ORDER_TYPE_LIMIT,
                    "product": self.kite.PRODUCT_CNC, "price": lower,
                },
                {
                    "transaction_type": exit_transaction, "quantity": quantity,
                    "order_type": self.kite.ORDER_TYPE_LIMIT,
                    "product": self.kite.PRODUCT_CNC, "price": upper,
                },
            ],
        )
        return str(trigger["trigger_id"])
