"""Upstox API v2 adapter (plain REST via `requests`, no SDK dependency).

IMPORTANT: Upstox's minute-level historical endpoint only reliably supports
1-minute candles, so for 3/5/15-minute we pull 1-minute data and resample
client-side rather than trusting every broker to support every bucket size.

Broker REST APIs change over time. Endpoint paths below reflect Upstox's
v2 docs (https://upstox.com/developer/api-documentation) as of this
project's writing - if a call starts failing, check that page first before
assuming the strategy logic is wrong. Zerodha's adapter (official SDK) is
the most battle-tested of the three; treat this one as needing a smoke
test with your own credentials before trusting it for money.
"""
import gzip
import json
import time
from datetime import timedelta
from io import BytesIO

import pandas as pd
import requests

from exchange.brokers.base import Broker
from exchange.resample import resample_ohlcv

BASE_URL = "https://api.upstox.com/v2"
INSTRUMENTS_URL = "https://assets.upstox.com/market-quote/instruments/exchange/complete.json.gz"


class UpstoxBroker(Broker):
    name = "upstox"

    def __init__(self, access_token: str):
        self.access_token = access_token
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {access_token}",
            "Accept": "application/json",
        })
        self._instrument_cache = None

    def _headers(self):
        return {"Authorization": f"Bearer {self.access_token}", "Accept": "application/json"}

    def _instruments(self) -> list:
        if self._instrument_cache is not None:
            return self._instrument_cache
        resp = self.session.get(INSTRUMENTS_URL, timeout=30)
        resp.raise_for_status()
        raw = gzip.decompress(resp.content)
        self._instrument_cache = json.loads(raw)
        return self._instrument_cache

    def _instrument_key(self, symbol: str, exchange: str = "NSE_EQ") -> str:
        for inst in self._instruments():
            if inst.get("trading_symbol") == symbol and inst.get("segment") == exchange:
                return inst["instrument_key"]
        # Fall back through reference-data segments in turn - lets callers
        # pass a benchmark/macro symbol (e.g. "NIFTY", "USDINR", "^GSPC",
        # "GIFT NIFTY") without needing a separate code path per segment,
        # since these tickers never collide with real NSE_EQ tradingsymbols.
        fallback_segments = ["NSE_INDEX", "GLOBAL_INDICATOR", "GLOBAL_INDEX"]
        for seg in fallback_segments:
            if seg == exchange:
                continue
            for inst in self._instruments():
                if inst.get("trading_symbol") == symbol and inst.get("segment") == seg:
                    return inst["instrument_key"]
        raise ValueError(
            f"Symbol {symbol} not found in Upstox instrument master "
            f"({exchange} or {'/'.join(fallback_segments)})")

    def historical_candles(self, symbol: str, interval: str, start, end) -> pd.DataFrame:
        instrument_key = self._instrument_key(symbol)
        fetch_1min = interval != "day"
        upstox_interval = "1minute" if fetch_1min else "day"

        frames = []
        cursor = start
        chunk = timedelta(days=30 if fetch_1min else 365)
        while cursor < end:
            chunk_end = min(cursor + chunk, end)
            url = (f"{BASE_URL}/historical-candle/{instrument_key}/{upstox_interval}/"
                   f"{chunk_end.date()}/{cursor.date()}")
            resp = self.session.get(url, timeout=30)
            if not resp.ok:
                # Upstox occasionally rejects a chunk at the edge of its available
                # history window ("Invalid date range") even though neighboring
                # chunks succeed - skip it rather than aborting the whole fetch.
                print(f"  WARNING: {symbol} {cursor.date()}..{chunk_end.date()} "
                      f"skipped ({resp.status_code}: {resp.text[:150]})")
                cursor = chunk_end + timedelta(days=1)
                time.sleep(0.25)
                continue
            candles = resp.json().get("data", {}).get("candles", [])
            if candles:
                # Upstox candle rows: [timestamp, open, high, low, close, volume, oi]
                df = pd.DataFrame(candles, columns=[
                    "date", "open", "high", "low", "close", "volume", "oi"])
                frames.append(df)
            cursor = chunk_end + timedelta(days=1)
            time.sleep(0.25)

        if not frames:
            return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume"])

        df = pd.concat(frames, ignore_index=True)
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").drop_duplicates(subset="date")
        df = df[["date", "open", "high", "low", "close", "volume"]].reset_index(drop=True)

        if interval not in ("minute", "day") and fetch_1min:
            df = resample_ohlcv(df, interval)
        return df

    def ltp(self, symbols: list) -> dict:
        keys = ",".join(self._instrument_key(s) for s in symbols)
        resp = self.session.get(f"{BASE_URL}/market-quote/ltp", params={"symbol": keys}, timeout=15)
        resp.raise_for_status()
        data = resp.json().get("data", {})
        # Response keys look like "NSE_EQ:RELIANCE" - the part after ':' is
        # the tradingsymbol we asked for, not the instrument_key.
        by_symbol = {k.split(":")[-1]: v["last_price"] for k, v in data.items()}
        return {s: by_symbol[s] for s in symbols if s in by_symbol}

    def place_order(self, symbol: str, direction: int, quantity: int,
                     order_type: str = "MARKET", product: str = "INTRADAY",
                     price: float = 0.0) -> str:
        payload = {
            "quantity": quantity,
            "product": "I" if product == "INTRADAY" else "D",
            "validity": "DAY",
            "price": price if order_type == "LIMIT" else 0,
            "instrument_token": self._instrument_key(symbol),
            "order_type": "MARKET" if order_type == "MARKET" else "LIMIT",
            "transaction_type": "BUY" if direction == 1 else "SELL",
            "disclosed_quantity": 0,
            "trigger_price": 0,
            "is_amo": False,
        }
        resp = self.session.post(f"{BASE_URL}/order/place", json=payload, timeout=15)
        resp.raise_for_status()
        return resp.json()["data"]["order_id"]

    def positions(self) -> list:
        resp = self.session.get(f"{BASE_URL}/portfolio/short-term-positions", timeout=15)
        resp.raise_for_status()
        return resp.json().get("data", [])
