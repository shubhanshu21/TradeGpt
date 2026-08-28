"""Downloads and caches historical daily OHLC candles for the configured
universe, via whichever broker is set in config/settings.yaml (broker.name).

Broker historical APIs are rate-limited and capped on lookback per request,
so this delegates chunking to the adapter and caches results to CSV to
avoid re-downloading on every backtest run.
"""
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests
import yaml

ROOT = Path(__file__).resolve().parent.parent.parent  # kat/
sys.path.insert(0, str(ROOT / "src"))

from exchange.brokers.factory import get_broker  # noqa: E402

CACHE_DIR = ROOT / "cache" / "historical"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Macro/global-cue series with no long history through the configured
# broker (Upstox's GLOBAL_INDEX/GLOBAL_INDICATOR feeds only go back to
# ~March 2020) but decades of free, public history on Yahoo Finance -
# real signals Indian swing traders actually watch (overnight US market
# moves, USD/INR, oil, since India imports ~85% of its crude), not
# available anywhere else in this pipeline. cache_name is this project's
# own name for the series (matches what preprocess.py's load_macro_cues
# looks for); yahoo_ticker is Yahoo's own symbol for it.
YAHOO_MACRO_SERIES = {
    "SP500": "^GSPC",
    "USDINR_LONG": "INR=X",
    "CRUDE": "CL=F",
}


def fetch_yahoo_symbol(cache_name: str, yahoo_ticker: str, start: datetime, end: datetime,
                        force_refresh: bool = False) -> pd.DataFrame:
    """Fetches daily OHLCV for a Yahoo Finance ticker via its public chart
    endpoint (no API key, no extra dependency - same requests library
    already used elsewhere) and caches it the same way as broker data."""
    cache_file = CACHE_DIR / f"yahoo_{cache_name}_day_{start.date()}_{end.date()}.csv"
    if cache_file.exists() and not force_refresh:
        return pd.read_csv(cache_file, parse_dates=["date"])

    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{yahoo_ticker}"
    params = {"period1": int(start.timestamp()), "period2": int(end.timestamp()), "interval": "1d"}
    headers = {"User-Agent": "Mozilla/5.0"}
    resp = requests.get(url, params=params, headers=headers, timeout=20)
    resp.raise_for_status()
    result = resp.json().get("chart", {}).get("result")
    if not result:
        return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume"])

    r = result[0]
    quote = r["indicators"]["quote"][0]
    df = pd.DataFrame({
        "date": pd.to_datetime(r["timestamp"], unit="s"),
        "open": quote["open"], "high": quote["high"], "low": quote["low"],
        "close": quote["close"], "volume": quote["volume"],
    }).dropna(subset=["close"]).reset_index(drop=True)

    if not df.empty:
        df.to_csv(cache_file, index=False)
    return df


def load_config():
    with open(ROOT / "config" / "settings.yaml") as f:
        return yaml.safe_load(f)


def fetch_symbol(broker, symbol: str, interval: str, start: datetime, end: datetime,
                  force_refresh: bool = False) -> pd.DataFrame:
    cache_file = CACHE_DIR / f"{broker.name}_{symbol}_{interval}_{start.date()}_{end.date()}.csv"
    if cache_file.exists() and not force_refresh:
        return pd.read_csv(cache_file, parse_dates=["date"])

    df = broker.historical_candles(symbol, interval, start, end)
    if not df.empty:
        df.to_csv(cache_file, index=False)
    return df


def fetch_universe_swing(force_refresh: bool = False) -> dict:
    """Returns {symbol: DataFrame} of daily candles for every symbol in the
    configured universe, over swing's own date range (config/settings.yaml
    -> swing). Also fetches the Nifty 50 index itself (see
    preprocess.py's load_benchmark_index/rel_strength_20d feature) -
    not part of universe.symbols since it's a benchmark reference, not a
    tradeable stock, but needs the same caching/refresh treatment.
    """
    cfg = load_config()
    broker = get_broker(cfg)
    interval = cfg["swing"].get("interval", "day")
    start = datetime.strptime(cfg["swing"]["start_date"], "%Y-%m-%d")
    end = datetime.strptime(cfg["swing"]["end_date"], "%Y-%m-%d")
    # NIFTY, INDIA VIX, and GIFT NIFTY fetched/cached alongside the universe
    # but NOT returned in data - they're benchmark/regime/macro-cue
    # references (see preprocess.py's load_benchmark_index/load_vix_index/
    # load_macro_cues), not tradeable symbols, so they must never reach the
    # backtest/paper/live engines as if they were one. GIFT NIFTY only has
    # ~2020-onward history via this broker (see YAHOO_MACRO_SERIES above
    # for the longer-history macro cues fetched separately, from Yahoo
    # Finance instead - GIFT NIFTY itself has no long free third-party
    # equivalent, so it stays broker-only and shorter-history).
    benchmark_symbols = ["NIFTY", "INDIA VIX", "GIFT NIFTY"]
    symbols = cfg["universe"]["symbols"] + benchmark_symbols

    data = {}
    for symbol in symbols:
        print(f"Fetching {symbol} via {broker.name} ({interval}, {start.date()} -> {end.date()})...")
        try:
            df = fetch_symbol(broker, symbol, interval, start, end, force_refresh=force_refresh)
        except Exception as e:
            # One symbol's transient API failure shouldn't abort the whole
            # universe fetch - log and move on, same as the empty-df case.
            print(f"  WARNING: fetch failed for {symbol} ({e}), skipping")
            continue
        if df.empty:
            print(f"  WARNING: no data returned for {symbol}, skipping")
            continue
        if symbol not in benchmark_symbols:
            data[symbol] = df
        print(f"  {len(df)} candles cached")

    for cache_name, yahoo_ticker in YAHOO_MACRO_SERIES.items():
        print(f"Fetching {cache_name} ({yahoo_ticker}) via Yahoo Finance ({start.date()} -> {end.date()})...")
        try:
            df = fetch_yahoo_symbol(cache_name, yahoo_ticker, start, end, force_refresh=force_refresh)
            print(f"  {len(df)} candles cached" if not df.empty else "  WARNING: no data returned, skipping")
        except Exception as e:
            print(f"  WARNING: fetch failed for {cache_name} ({e}), skipping")
        time.sleep(0.5)  # be polite to Yahoo's public endpoint, no auth/key to rate-limit us

    return data


if __name__ == "__main__":
    fetch_universe_swing()
