"""Shared OHLCV resampling, used by any adapter whose source data is finer-
grained than what was requested (Upstox/Dhan only give certain native
minute buckets; CSV imports are typically 1-minute).
"""
import pandas as pd

RULE_MAP = {"3minute": "3min", "5minute": "5min", "15minute": "15min", "60minute": "60min"}


def resample_ohlcv(df: pd.DataFrame, interval: str) -> pd.DataFrame:
    rule = RULE_MAP[interval]
    return df.set_index("date").resample(rule, origin="start_day").agg({
        "open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum",
    }).dropna().reset_index()
