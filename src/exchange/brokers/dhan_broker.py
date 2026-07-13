"""Dhan API v2 adapter (plain REST via `requests`, no SDK dependency).

Dhan's intraday chart endpoint natively supports 1/5/15/25/60-minute
buckets; everything except 3-minute maps directly, and 3-minute is built
by resampling 1-minute data client-side.

As with the Upstox adapter: endpoint paths reflect Dhan's v2 docs
(https://dhanhq.co/docs/v2/) as of this project's writing. Broker APIs
drift - smoke-test with your own credentials before trusting this for
real orders. Zerodha's adapter (official SDK) is the most battle-tested
of the three.
"""
import time
from datetime import timedelta
from io import StringIO

import pandas as pd
import requests

from exchange.brokers.base import Broker
from exchange.resample import resample_ohlcv

BASE_URL = "https://api.dhan.co/v2"
SCRIP_MASTER_URL = "https://images.dhan.co/api-data/api-scrip-master.csv"

MINUTE_INTERVAL_MAP = {"minute": "1", "5minute": "5", "15minute": "15", "60minute": "60"}


class DhanBroker(Broker):
    name = "dhan"

    def __init__(self, client_id: str, access_token: str):
        self.client_id = client_id
        self.session = requests.Session()
        self.session.headers.update({
            "access-token": access_token,
            "client-id": client_id,
            "Content-Type": "application/json",
            "Accept": "application/json",
        })
        self._scrip_cache = None

    def _scrip_master(self) -> pd.DataFrame:
        if self._scrip_cache is not None:
            return self._scrip_cache
        resp = requests.get(SCRIP_MASTER_URL, timeout=30)
        resp.raise_for_status()
        self._scrip_cache = pd.read_csv(StringIO(resp.text), low_memory=False)
        return self._scrip_cache

    def _security_id(self, symbol: str) -> str:
        """NSE-only by design: the scrip master lists the same tradingsymbol
        on both NSE and BSE with different security IDs, and this project
        only trades NSE. Picking the wrong one silently sends a mismatched
        (exchange, security_id) pair to Dhan's API - which is exactly what
        caused an opaque 401 on the historical-candle endpoint before this
        filter was added.
        """
        df = self._scrip_master()
        exch_col = next(c for c in df.columns if "EXM_EXCH_ID" in c.upper())
        symbol_col = next(c for c in df.columns if "TRADING_SYMBOL" in c.upper())
        id_col = next(c for c in df.columns if "SECURITY_ID" in c.upper())
        instrument_col = next(c for c in df.columns if "INSTRUMENT_NAME" in c.upper())

        match = df[
            (df[symbol_col] == symbol)
            & (df[exch_col] == "NSE")
            & (df[instrument_col] == "EQUITY")
        ]
        if match.empty:
            raise ValueError(f"Symbol {symbol} not found in Dhan scrip master under NSE_EQ")
        return str(int(match.iloc[0][id_col]))

    def historical_candles(self, symbol: str, interval: str, start, end) -> pd.DataFrame:
        security_id = self._security_id(symbol)

        if interval == "day":
            payload = {
                "securityId": security_id, "exchangeSegment": "NSE_EQ",
                "instrument": "EQUITY", "fromDate": str(start.date()),
                "toDate": str(end.date()),
            }
            resp = self.session.post(f"{BASE_URL}/charts/historical", json=payload, timeout=30)
            resp.raise_for_status()
            return self._parse_columnar(resp.json())

        fetch_minute = "3" if interval == "3minute" else MINUTE_INTERVAL_MAP.get(interval, "1")
        frames = []
        cursor = start
        chunk = timedelta(days=5)  # Dhan intraday endpoint is capped to short windows per call
        while cursor < end:
            chunk_end = min(cursor + chunk, end)
            payload = {
                "securityId": security_id, "exchangeSegment": "NSE_EQ",
                "instrument": "EQUITY", "interval": fetch_minute,
                "fromDate": str(cursor.date()), "toDate": str(chunk_end.date()),
            }
            resp = self.session.post(f"{BASE_URL}/charts/intraday", json=payload, timeout=30)
            resp.raise_for_status()
            df = self._parse_columnar(resp.json())
            if not df.empty:
                frames.append(df)
            cursor = chunk_end + timedelta(days=1)
            time.sleep(0.25)

        if not frames:
            return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume"])

        df = pd.concat(frames, ignore_index=True).drop_duplicates(subset="date")
        df = df.sort_values("date").reset_index(drop=True)

        if interval == "3minute":
            df = resample_ohlcv(df, "3minute")
        return df

    @staticmethod
    def _parse_columnar(data: dict) -> pd.DataFrame:
        if not data or "open" not in data:
            return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume"])
        df = pd.DataFrame({
            "date": pd.to_datetime(data["timestamp"], unit="s"),
            "open": data["open"], "high": data["high"], "low": data["low"],
            "close": data["close"], "volume": data["volume"],
        })
        return df

    def ltp(self, symbols: list) -> dict:
        security_ids = {s: self._security_id(s) for s in symbols}
        payload = {"NSE_EQ": [int(sid) for sid in security_ids.values()]}
        resp = self.session.post(f"{BASE_URL}/marketfeed/ltp", json=payload, timeout=15)
        resp.raise_for_status()
        data = resp.json().get("data", {}).get("NSE_EQ", {})
        return {
            symbol: data[sid]["last_price"]
            for symbol, sid in security_ids.items() if sid in data
        }

    def place_order(self, symbol: str, direction: int, quantity: int,
                     order_type: str = "MARKET", product: str = "INTRADAY",
                     price: float = 0.0) -> str:
        payload = {
            "dhanClientId": self.client_id,
            "transactionType": "BUY" if direction == 1 else "SELL",
            "exchangeSegment": "NSE_EQ",
            "productType": "INTRADAY" if product == "INTRADAY" else "CNC",
            "orderType": "MARKET" if order_type == "MARKET" else "LIMIT",
            "validity": "DAY",
            "securityId": self._security_id(symbol),
            "quantity": quantity,
            "price": price if order_type == "LIMIT" else 0,
        }
        resp = self.session.post(f"{BASE_URL}/orders", json=payload, timeout=15)
        resp.raise_for_status()
        return resp.json()["orderId"]

    def positions(self) -> list:
        resp = self.session.get(f"{BASE_URL}/positions", timeout=15)
        resp.raise_for_status()
        return resp.json()
