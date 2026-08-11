"""Downloads and caches historical daily OHLC candles for the configured
universe, via whichever broker is set in config/settings.yaml (broker.name).

Broker historical APIs are rate-limited and capped on lookback per request,
so this delegates chunking to the adapter and caches results to CSV to
avoid re-downloading on every backtest run.
"""
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent.parent  # kat/
sys.path.insert(0, str(ROOT / "src"))

from exchange.brokers.factory import get_broker  # noqa: E402

CACHE_DIR = ROOT / "cache" / "historical"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


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
    # NIFTY and INDIA VIX fetched/cached alongside the universe but NOT
    # returned in data - they're benchmark/regime references (see
    # preprocess.py's load_benchmark_index/load_vix_index), not tradeable
    # symbols, so they must never reach the backtest/paper/live engines as
    # if they were one.
    benchmark_symbols = ["NIFTY", "INDIA VIX"]
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
    return data


if __name__ == "__main__":
    fetch_universe_swing()
