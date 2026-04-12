"""
SOVEREIGN KRAKEN — Data Engine (V4.7 Abyss-Streamer)
=====================================================
- Zero-copy TF Dataset generator (no RAM materialization)
- Full 23-feature engineering inline
- Proven stable on 24GB CPU hosts
"""

import numpy as np
import pandas as pd
import pickle
import tensorflow as tf

# ── Settings ──────────────────────────────────────────────────────────────────
WINDOW_SIZE  = 120   # 2-hour context — max safe on 24GB CPU (CTX² attention)
TARGET_STEPS = [1, 5, 15, 30, 60]

def build_feature_cols():
    """
    24-dimension input vector — V4.7 Enhanced.
    'count' replaced by 'funding_rate_proxy': perpetual futures premium proxy
    (close vs sma_99 z-score). Highly predictive of 15-min direction.
    """
    return [
        'open', 'high', 'low', 'close', 'volume', 'quote_volume',
        'funding_rate_proxy',                      # ← NEW (was 'count': always 0)
        'taker_buy_volume', 'taker_buy_quote_volume',
        'sma_7', 'sma_25', 'sma_99', 'rsi', 'macd', 'macd_signal', 'macd_hist',
        'bollinger_upper', 'bollinger_lower', 'atr', 'obv', 'volatility', 'adx', 'cci',
        'bb_width',                                # ← NEW: Bollinger Band width (squeeze detector)
        'vwap_dist',                               # ← NEW: Scalpers Mean-Reversion Anchor
        'stoch_rsi',                               # ← NEW: Fast Momentum Trigger
        'cvd',                                     # ← NEW: Cumulative Volume Delta Proxy
    ]


import pandas_ta as ta

def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all technical indicators using high-speed pandas_ta engines."""
    df = df.copy()
    
    # ── V6.3 Iron-Clad: Force Base Columns ──────────────────────────────────
    required_base = ['open', 'high', 'low', 'close', 'volume', 'quote_volume', 
                     'taker_buy_volume', 'taker_buy_quote_volume']
    for col in required_base:
        if col not in df.columns:
            df[col] = 0.0 if col != 'close' else df.get('close', 0.0)
    
    # ── Index Enforcement ────────────────────────────────────────────────────
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
    elif not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(np.arange(len(df)), unit='m')

    # ── Standard Momentum (Pandas-TA) ────────────────────────────────────────
    df.ta.rsi(length=14, append=True)
    df.ta.macd(fast=12, slow=26, signal=9, append=True)
    df.ta.adx(length=14, append=True)
    df.ta.cci(length=20, append=True)
    df.ta.obv(append=True)
    
    # ── Fuzzy Renaming Logic ───────────────
    # We find columns by searching for partial name matches
    def find_col(options):
        return next((c for c in df.columns if c.split('_')[0].upper() in [o.upper() for o in options]), None)

    mapping = {
        'rsi':           ['RSI'],
        'macd':          ['MACD'],
        'macd_signal':   ['MACDs', 'MACDS'],
        'macd_hist':     ['MACDh', 'MACDH'],
        'adx':           ['ADX'],
        'cci':           ['CCI'],
        'obv':           ['OBV']
    }
    for target, options in mapping.items():
        found = find_col(options)
        if found: df[target] = df[found]
        else:     df[target] = 0.0

    # ── Robust Manual Calculations ───────────────────────────────────────────
    close = df["close"]
    # Bollinger Bands
    df["sma_20"] = close.rolling(20, min_periods=1).mean()
    std20 = close.rolling(20, min_periods=1).std().fillna(1e-9)
    df["bollinger_upper"] = df["sma_20"] + 2 * std20
    df["bollinger_lower"] = df["sma_20"] - 2 * std20
    df["bb_width"] = (df["bollinger_upper"] - df["bollinger_lower"]) / (df["sma_20"] + 1e-9)

    # ATR
    prev_c = close.shift(1).fillna(close)
    tr = pd.concat([df["high"] - df["low"],
                    (df["high"] - prev_c).abs(),
                    (df["low"]  - prev_c).abs()], axis=1).max(axis=1)
    df["atr"] = tr.rolling(14, min_periods=1).mean()

    # ── Scalping Core (Pandas-TA) ────────────────────────────────────────────
    try:
        df.ta.vwap(append=True)
        vcol = next((c for c in df.columns if "VWAP" in c), None)
        df["vwap_dist"] = (close - df[vcol]) / (df[vcol] + 1e-9) if vcol else 0.0
    except Exception:
        df["vwap_dist"] = 0.0

    df.ta.stochrsi(length=14, rsi_length=14, k=3, d=3, append=True)
    sr_col = next((c for c in df.columns if "STOCHRSI" in c.upper() and "K" in c.upper()), None)
    df["stoch_rsi"] = df[sr_col] if sr_col else 0.5

    # ── Custom Institutional Proxies ─────────────────────────────────────────
    # Funding Rate Proxy
    df["sma_99"] = close.rolling(99, min_periods=1).mean()
    premium = (close - df["sma_99"]) / (df["sma_99"] + 1e-9)
    p_mean  = premium.rolling(288, min_periods=1).mean()
    p_std   = premium.rolling(288, min_periods=1).std().fillna(1e-9)
    df["funding_rate_proxy"] = ((premium - p_mean) / (p_std + 1e-9)).clip(-3, 3)

    # CVD Proxy
    diff = close.diff().fillna(0)
    df["cvd"] = (df["volume"] * np.sign(diff)).rolling(60, min_periods=1).sum()

    # Volatility
    df["volatility"] = close.pct_change().rolling(20, min_periods=1).std().fillna(0)
    
    # ── SMA Set ──────────────────────────────────────────────────────────────
    df["sma_7"]  = close.rolling(7,  min_periods=1).mean()
    df["sma_25"] = close.rolling(25, min_periods=1).mean()

    # Final Safety: Ensure all build_feature_cols exist and are clean (V8.5 Hammer)
    for col in build_feature_cols():
        if col not in df.columns:
            df[col] = 0.0
    
    # Force clean numerical types and scrub Inf/NaN
    df = df.replace([np.inf, -np.inf], np.nan)
    return df.fillna(0).clip(-1e9, 1e9).astype("float32")


class KATScaler:
    """Incremental Z-score scaler — pickle-safe."""
    def __init__(self):
        self.mean = None
        self.std  = None

    def fit(self, data: np.ndarray):
        self.mean = data.mean(axis=0)
        self.std  = data.std(axis=0) + 1e-8

    def transform_X(self, data):
        # Prevent division by zero if std is 0 (V8.4 Patch)
        safe_std = np.copy(self.std)
        safe_std[safe_std == 0] = 1.0
        return (data - self.mean) / safe_std

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str):
        with open(path, "rb") as f:
            return pickle.load(f)


def build_dataset_streaming(df, context_window=60, forecast_steps=15,
                             batch_size=256, scaler=None, scaler_save_path=None):
    """
    ABYSS-STREAMER (V4.7) — TF generator pipeline, zero RAM materialization.

    Returns dict with:
      tr_ds, va_ds  : tf.data.Dataset (batched, prefetched)
      steps_tr, steps_va : int
      n_features    : int
      scaler        : fitted KATScaler
    """
    print(f"   ⚓ Abyss-Streamer V4.7: {len(df):,} candles → CTX={context_window} FORECAST={forecast_steps}")

    # Step 1: Feature engineering
    df_feat  = compute_indicators(df)
    features = build_feature_cols()
    data     = df_feat[features].values.astype("float32")   # (N, 23)
    n        = len(data)

    # Step 2: Fit scaler on train slice only
    n_win    = n - context_window - forecast_steps + 1
    tr_end   = int(n_win * 0.8)
    va_end   = int(n_win * 0.9)

    if scaler is None:
        scaler = KATScaler()
        scaler.fit(data[:tr_end + context_window])
    if scaler_save_path:
        scaler.save(scaler_save_path)

    scaled = scaler.transform_X(data)    # (N, 23) — tiny 11 MB array

    t_close = 3    # 'close'
    t_vol   = 20   # 'volatility'
    t_volu  = 4    # 'volume'

    # Step 3: Generator factories (V10.0 Entry-Anchored)
    def make_gen(start, end):
        def gen():
            for i in range(start, end):
                x  = scaled[i : i + context_window]                             # (T, 27)
                end_i = i + context_window + forecast_steps
                # Include i+context-1 as the 'entry price' anchor
                yc = scaled[i + context_window - 1 : end_i, t_close]            # (F+1,)
                yv = scaled[i + context_window - 1 : end_i, t_vol]              # (F+1,)
                yu = scaled[i + context_window - 1 : end_i, t_volu]             # (F+1,)
                y  = np.stack([yc, yv, yu], axis=-1)                            # (F+1, 3)
                
                # Dual Target for certainty tracking (dummy)
                c_dummy = np.zeros((1,), dtype=np.float32)
                yield x, (y, c_dummy)
        return gen

    sig = (
        tf.TensorSpec(shape=(context_window, len(features)), dtype=tf.float32),
        (
            tf.TensorSpec(shape=(forecast_steps + 1, 3),        dtype=tf.float32),
            tf.TensorSpec(shape=(1,),                         dtype=tf.float32),
        )
    )

    tr_ds = (tf.data.Dataset.from_generator(make_gen(0,      tr_end), output_signature=sig)
             .shuffle(20000, reshuffle_each_iteration=True)
             .batch(batch_size).prefetch(tf.data.AUTOTUNE))
    va_ds = (tf.data.Dataset.from_generator(make_gen(tr_end, va_end), output_signature=sig)
             .batch(batch_size).prefetch(tf.data.AUTOTUNE))

    steps_tr = max(1, tr_end // batch_size)
    steps_va = max(1, (va_end - tr_end) // batch_size)

    print(f"   📊 Stream ready: {tr_end:,} train | {va_end - tr_end:,} val windows. "
          f"Steps/epoch: {steps_tr}")

    return {
        "tr_ds":     tr_ds,
        "va_ds":     va_ds,
        "steps_tr":  steps_tr,
        "steps_va":  steps_va,
        "n_features": len(features),
        "scaler":    scaler,
    }


# ── Legacy alias kept for any other callers ───────────────────────────────────
def build_dataset(df, context_window=60, forecast_steps=15, scaler=None, scaler_save_path=None):
    """Thin wrapper — returns streaming dict (V4.7 compatible)."""
    return build_dataset_streaming(df, context_window=context_window,
                                   forecast_steps=forecast_steps,
                                   scaler=scaler, scaler_save_path=scaler_save_path)