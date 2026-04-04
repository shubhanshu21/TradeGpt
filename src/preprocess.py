"""
KAT 3 Core Infrastructure — Preprocessing & Data Pipelines
==========================================================
Includes:
 - Universal feature generator (23 features)
 - KAT Scaler (Invertible data normalization)
 - Stride-trick memory efficient dataset building
"""

import os, gc
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
    df["ema_short"] = df["close"].ewm(span=12, adjust=False).mean()
    df["ema_mid"] = df["close"].ewm(span=26, adjust=False).mean()
    df["ema_long"] = df["close"].ewm(span=50, adjust=False).mean()
    
    # MACD
    df["macd"] = df["ema_short"] - df["ema_mid"]
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]
    
    # RSI (Pandas-friendly calculation)
    delta = df["close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss.replace(0, 1e-9))
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

class KATScaler:
    def __init__(self):
        self.mean = None
        self.scale = None

    def fit(self, X):
        self.mean = np.mean(X, axis=0)
        self.scale = np.std(X, axis=0) + 1e-8
        
    def transform_X(self, X):
        if self.mean is None: return X
        return (X - self.mean) / self.scale

    def inverse_y(self, y):
        # Target is at index 3 (close)
        return y * self.scale[3] + self.mean[3]
    
    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self, f)
            
    @staticmethod
    def load(path):
        with open(path, "rb") as f:
            return pickle.load(f)

def build_dataset(df, context_window=360, forecast_steps=1, scaler=None, scaler_save_path=None):
    if df is None or len(df) <= context_window + forecast_steps:
        return None

    df_feat = add_derived_features(df)
    features = build_feature_cols()
    data = df_feat[features].values.astype("float32")

    if scaler is None:
        scaler = KATScaler()
        scaler.fit(data)
    if scaler_save_path:
        scaler.save(scaler_save_path)

    scaled_data = scaler.transform_X(data)
    
    # Memory Efficient Slinding Windows (using stride tricks)
    from numpy.lib.stride_tricks import sliding_window_view
    Xs = sliding_window_view(scaled_data, window_shape=(context_window, len(features)))
    Xs = Xs.squeeze(axis=1) # Remove the extra dimension from window_shape
    
    # Slice to align with targets
    Xs_view = Xs[:-forecast_steps]
    ys_view = scaled_data[context_window + forecast_steps - 1:, 3]
    
    # Standard array for split
    Xs_final = np.array(Xs_view)
    ys_final = np.array(ys_view)
    
    n = len(Xs_final)
    tr_idx = int(n * 0.8)
    va_idx = int(n * 0.9)

    ds = {
        "X_train": Xs_final[:tr_idx], "y_train": ys_final[:tr_idx],
        "X_val":   Xs_final[tr_idx:va_idx], "y_val":   ys_final[tr_idx:va_idx],
        "X_test":  Xs_final[va_idx:], "y_test":  ys_final[va_idx:],
        "n_features": len(features),
        "scaler":  scaler
    }
    
    gc.collect()
    print(f"   📊 Dataset Built: {n:,} windows | RSS Memory Reclaimed")
    return ds

def create_tf_dataset(Xs, ys, batch_size=32, shuffle=True):
    dataset = tf.data.Dataset.from_tensor_slices((Xs, ys))
    if shuffle:
        dataset = dataset.shuffle(buffer_size=min(len(Xs), 10000))
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset