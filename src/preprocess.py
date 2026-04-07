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
    ]


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all 23 technical indicator columns in-place."""
    df = df.copy()
    close = df["close"].astype(float)
    high  = df["high"].astype(float)
    low   = df["low"].astype(float)
    vol   = df["volume"].astype(float)

    # Fill any missing raw cols
    for col in ['quote_volume', 'taker_buy_volume', 'taker_buy_quote_volume']:
        if col not in df.columns:
            df[col] = 0.0

    # Moving averages
    df["sma_7"]  = close.rolling(7,  min_periods=1).mean()
    df["sma_25"] = close.rolling(25, min_periods=1).mean()
    df["sma_99"] = close.rolling(99, min_periods=1).mean()

    # ── Funding Rate Proxy (Enhancement #4) ──────────────────────────────────
    # Measures the premium/discount of spot vs fair value (sma_99).
    # In perpetual futures, positive = market paying longs → bearish pressure.
    premium = (close - df["sma_99"]) / (df["sma_99"] + 1e-9)
    # Z-score to keep in [-3, 3] range, then clip
    p_mean  = premium.rolling(288, min_periods=1).mean()   # 5-hour z-score window
    p_std   = premium.rolling(288, min_periods=1).std().fillna(1e-9)
    df["funding_rate_proxy"] = ((premium - p_mean) / (p_std + 1e-9)).clip(-3, 3)

    # RSI (14)
    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(14, min_periods=1).mean()
    loss  = (-delta.clip(upper=0)).rolling(14, min_periods=1).mean()
    df["rsi"] = 100 - (100 / (1 + gain / (loss + 1e-9)))

    # MACD
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df["macd"]        = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"]   = df["macd"] - df["macd_signal"]

    # Bollinger Bands (20, 2σ)
    sma20 = close.rolling(20, min_periods=1).mean()
    std20 = close.rolling(20, min_periods=1).std().fillna(0)
    df["bollinger_upper"] = sma20 + 2 * std20
    df["bollinger_lower"] = sma20 - 2 * std20

    # ── BB Width — Squeeze Detector (Enhancement #4b) ─────────────────────────
    # Narrow bands = coiling energy → breakout imminent. Wide bands = trending.
    df["bb_width"] = (df["bollinger_upper"] - df["bollinger_lower"]) / (sma20 + 1e-9)

    # ATR (14)
    prev_c = close.shift(1).fillna(close)
    tr = pd.concat([high - low,
                    (high - prev_c).abs(),
                    (low  - prev_c).abs()], axis=1).max(axis=1)
    df["atr"] = tr.rolling(14, min_periods=1).mean()

    # OBV
    df["obv"] = (vol * np.sign(close.diff().fillna(0))).cumsum()

    # Volatility
    df["volatility"] = close.pct_change().rolling(20, min_periods=1).std().fillna(0)

    # ADX (14)
    plus_dm  = (high.diff()).clip(lower=0)
    minus_dm = (-low.diff()).clip(lower=0)
    plus_dm  = plus_dm.where(plus_dm  > minus_dm, 0)
    minus_dm = minus_dm.where(minus_dm > plus_dm,  0)
    atr14    = tr.rolling(14, min_periods=1).mean()
    pdi = 100 * (plus_dm.rolling(14,  min_periods=1).mean() / (atr14 + 1e-9))
    mdi = 100 * (minus_dm.rolling(14, min_periods=1).mean() / (atr14 + 1e-9))
    dx  = 100 * ((pdi - mdi).abs() / (pdi + mdi + 1e-9))
    df["adx"] = dx.rolling(14, min_periods=1).mean()

    # CCI (20)
    tp = (high + low + close) / 3
    df["cci"] = (tp - tp.rolling(20, min_periods=1).mean()) / \
                (0.015 * tp.rolling(20, min_periods=1).std().fillna(1e-9))

    return df.fillna(0)


class KATScaler:
    """Incremental Z-score scaler — pickle-safe."""
    def __init__(self):
        self.mean = None
        self.std  = None

    def fit(self, data: np.ndarray):
        self.mean = data.mean(axis=0)
        self.std  = data.std(axis=0) + 1e-8

    def transform_X(self, data: np.ndarray) -> np.ndarray:
        return (data - self.mean) / self.std

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

    # Step 3: Generator factories (zero-copy slicing)
    def make_gen(start, end):
        def gen():
            for i in range(start, end):
                x  = scaled[i : i + context_window]                            # (T, 23)
                end_i = i + context_window + forecast_steps
                yc = scaled[i + context_window : end_i, t_close]               # (F,)
                yv = scaled[i + context_window : end_i, t_vol]                 # (F,)
                yu = scaled[i + context_window : end_i, t_volu]                # (F,)
                y  = np.stack([yc, yv, yu], axis=-1)                           # (F, 3)
                yield x, y
        return gen

    sig = (
        tf.TensorSpec(shape=(context_window, len(features)), dtype=tf.float32),
        tf.TensorSpec(shape=(forecast_steps, 3),             dtype=tf.float32),
    )

    tr_ds = (tf.data.Dataset.from_generator(make_gen(0,      tr_end), output_signature=sig)
             .batch(batch_size).prefetch(2))
    va_ds = (tf.data.Dataset.from_generator(make_gen(tr_end, va_end), output_signature=sig)
             .batch(batch_size).prefetch(1))

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