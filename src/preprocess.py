"""
KAT 3 Core Infrastructure — Preprocessing & Data Pipelines
==========================================================
Includes:
 - Universal feature generator (47 features)
 - KAT Scaler (Invertible data normalization)
 - Memory-efficient dataset building (TensorFlow Data)
"""

import os
import numpy as np
import pandas as pd
import tensorflow as tf
import pickle
from pathlib import Path

# ──────────────────────────────────────────────────────────────────────────────
# SYSTEM SETTINGS
# ──────────────────────────────────────────────────────────────────────────────
WINDOW_SIZE = 360 
TARGET_STEPS = [1, 5, 15, 30, 60]

def build_feature_cols():
    """Defines the standard KAT 3 input feature set."""
    return [
        "open", "high", "low", "close", "volume",
        "rsi", "rsi_signal", "macd", "macd_signal", "macd_hist",
        "bollinger_h", "bollinger_l", "ema_short", "ema_mid", "ema_long",
        "volatility", "volume_ma", "return_short", "return_long",
        "hour_sin", "hour_cos", "day_sin", "day_cos"
    ]

def add_derived_features(df):
    """Enrich raw candles with indicators for KAT 3."""
    df = df.copy()
    
    # Returns
    df["return_short"] = df["close"].pct_change(5)
    df["return_long"] = df["close"].pct_change(15)
    
    # EMAs
    df["ema_short"] = df["close"].ewm(span=12).mean()
    df["ema_mid"] = df["close"].ewm(span=26).mean()
    df["ema_long"] = df["close"].ewm(span=50).mean()
    
    # MACD
    df["macd"] = df["ema_short"] - df["ema_mid"]
    df["macd_signal"] = df["macd"].ewm(span=9).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]
    
    # RSI
    delta = df["close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-9)
    df["rsi"] = 100 - (100 / (1 + rs))
    df["rsi_signal"] = df["rsi"].rolling(window=9).mean()
    
    # Bollinger
    df["sma_20"] = df["close"].rolling(window=20).mean()
    df["std_20"] = df["close"].rolling(window=20).std()
    df["bollinger_h"] = df["sma_20"] + (df["std_20"] * 2)
    df["bollinger_l"] = df["sma_20"] - (df["std_20"] * 2)
    
    # Volatility & Volume
    df["volatility"] = df["close"].rolling(window=15).std()
    df["volume_ma"] = df["volume"].rolling(window=15).mean()
    
    # Time Cycles
    if 'timestamp' in df.columns:
        dt = pd.to_datetime(df['timestamp'], unit='s')
        df["hour_sin"] = np.sin(2 * np.pi * dt.dt.hour / 24)
        df["hour_cos"] = np.cos(2 * np.pi * dt.dt.hour / 24)
        df["day_sin"] = np.sin(2 * np.pi * dt.dt.dayofweek / 7)
        df["day_cos"] = np.cos(2 * np.pi * dt.dt.dayofweek / 7)
    else:
        df["hour_sin"] = df["hour_cos"] = df["day_sin"] = df["day_cos"] = 0

    return df.fillna(0)

# ──────────────────────────────────────────────────────────────────────────────
# SCALING
# ──────────────────────────────────────────────────────────────────────────────

class KATScaler:
    """Universal Scaler for features and price targets."""
    def __init__(self):
        self.mean = None
        self.scale = None

    def fit(self, X):
        self.mean = np.mean(X, axis=0)
        self.scale = np.std(X, axis=0) + 1e-8
        
    def transform_X(self, X):
        if self.mean is None: raise ValueError("Not fitted")
        return (X - self.mean) / self.scale

    def transform_y(self, y):
        if self.mean is None: raise ValueError("Not fitted")
        return (y - self.mean[3]) / self.scale[3]

    def inverse_y(self, y_scaled):
        if self.mean is None: raise ValueError("Not fitted")
        return (y_scaled * self.scale[3]) + self.mean[3]

    @classmethod
    def load(cls, path):
        with open(path, 'rb') as f: return pickle.load(f)

    def save(self, path):
        with open(path, 'wb') as f: pickle.dump(self, f)

# ──────────────────────────────────────────────────────────────────────────────
# DATASET BUILDING
# ──────────────────────────────────────────────────────────────────────────────

def build_dataset(df, context_window=360, forecast_steps=1, scaler=None, scaler_save_path=None):
    """
    Universal dataset builder with Memory-Efficient splitting.
    Returns: Dict containing train/val/test splits + scaler.
    """
    if df is None or len(df) <= context_window + forecast_steps:
        return None

    # 1. Feature Engineering
    df = add_derived_features(df)
    features = build_feature_cols()
    X_raw = df[features].values

    # 2. Scaling
    if scaler is None:
        scaler = KATScaler()
        scaler.fit(X_raw)
    
    if scaler_save_path:
        scaler.save(scaler_save_path)
        print(f"   💾 Scaler cached: {scaler_save_path}")

    X_scaled = scaler.transform_X(X_raw)

    # 3. Create Windows
    Xs, ys = [], []
    for i in range(context_window, len(df) - forecast_steps):
        Xs.append(X_scaled[i-context_window : i])
        ys.append(X_scaled[i + forecast_steps - 1, 3])

    Xs = np.array(Xs, dtype="float32")
    ys = np.array(ys, dtype="float32")

    # 4. Standard Chronological Split
    n = len(Xs)
    tr_idx = int(n * 0.8)
    va_idx = int(n * 0.9)

    ds = {
        "X_train": Xs[:tr_idx], "y_train": ys[:tr_idx],
        "X_val":   Xs[tr_idx:va_idx], "y_val":   ys[tr_idx:va_idx],
        "X_test":  Xs[va_idx:], "y_test":  ys[va_idx:],
        "n_features": len(features),
        "scaler":  scaler
    }
    
    print(f"   📊 Combined Dataset: {n:,} windows | Features: {len(features)}")
    return ds

def create_tf_dataset(Xs, ys, batch_size=64, shuffle=True):
    """
    Converts raw numpy arrays to optimized TensorFlow Dataset.
    """
    dataset = tf.data.Dataset.from_tensor_slices((Xs, ys))
    if shuffle:
        dataset = dataset.shuffle(buffer_size=min(len(Xs), 10000))
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset