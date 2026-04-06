"""
ALPHA — Foundational Supervised Predictor (Stacked LSTM)
========================================================
Architecture:
  - Stacked LSTM → Dense Regression
  - Predicts ONE next-candle close price
  - Baseline for market volatility modeling
"""

import tensorflow as tf
import keras
from keras import layers, regularizers
import numpy as np

@keras.utils.register_keras_serializable()
class Alpha(keras.Model):
    def __init__(self, n_features: int, context_window: int = 150, lstm_units: int = 128, dropout: float = 0.2, name: str = "ALPHA"):
        super().__init__(name=name)
        self.context_window, self.n_features = context_window, n_features
        self.lstm = keras.Sequential([
            layers.LSTM(lstm_units, return_sequences=True, kernel_regularizer=regularizers.l2(1e-4)),
            layers.Dropout(dropout),
            layers.LSTM(lstm_units // 2, return_sequences=True, kernel_regularizer=regularizers.l2(1e-4)),
            layers.Dropout(dropout),
            layers.LSTM(lstm_units // 4, return_sequences=False),
            layers.Dropout(dropout),
            layers.Dense(64, activation="relu"),
            layers.Dense(32, activation="relu"),
            layers.Dense(1, activation="linear")
        ])

    def call(self, x, training=False):
        return tf.squeeze(self.lstm(x, training=training), axis=-1)

    def get_config(self):
        return {"n_features": self.n_features, "context_window": self.context_window}

def build_alpha(n_features: int, context_window: int = 150) -> keras.Model:
    model = Alpha(n_features=n_features, context_window=context_window)
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse", metrics=["mae"])
    model(tf.zeros((1, context_window, n_features)))
    print(f"ALPHA Engine — Built | Params: {model.count_params():,}")
    return model
