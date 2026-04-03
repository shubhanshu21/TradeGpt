"""
TITAN — High-Dimensional Deep Regression (~33M parameters)
=========================================================
Architecture:
  - Bidirectional LSTM + Multi-Head Attention
  - Deep Dense Tower
  - Hybrid Architecture for non-linear market regimes
"""

import tensorflow as tf
import keras
from keras import layers, regularizers
import numpy as np

@keras.utils.register_keras_serializable()
class TitanBlock(layers.Layer):
    def __init__(self, d_model: int, n_heads: int, ff_dim: int, dropout: float = 0.1):
        super().__init__()
        self.attn = layers.MultiHeadAttention(num_heads=n_heads, key_dim=d_model // n_heads)
        self.ffn = keras.Sequential([layers.Dense(ff_dim, activation="gelu"), layers.Dense(d_model), layers.Dropout(dropout)])
        self.norm1, self.norm2, self.drop = layers.LayerNormalization(epsilon=1e-6), layers.LayerNormalization(epsilon=1e-6), layers.Dropout(dropout)

    def call(self, x, training=False):
        x = self.norm1(x + self.drop(self.attn(x, x), training=training))
        return self.norm2(x + self.ffn(x, training=training))

@keras.utils.register_keras_serializable()
class Titan(keras.Model):
    def __init__(self, n_features: int, context_window: int = 150, d_model: int = 512, n_heads: int = 8, n_blocks: int = 4, lstm_units: int = 512, dropout: float = 0.15, name: str = "TITAN"):
        super().__init__(name=name)
        self.context_window, self.n_features = context_window, n_features
        self.input_proj, self.pos_embed = layers.Dense(d_model), layers.Embedding(context_window, d_model)
        self.bilstm = layers.Bidirectional(layers.LSTM(lstm_units, return_sequences=True), name="bilstm_core")
        self.lstm_proj = layers.Dense(d_model)
        self.blocks = [TitanBlock(d_model, n_heads, d_model * 4, dropout) for _ in range(n_blocks)]
        self.global_pool = layers.GlobalAveragePooling1D()
        self.tower = keras.Sequential([layers.Dense(1024, activation="gelu"), layers.Dropout(dropout), layers.Dense(512, activation="gelu"), layers.Dense(128, activation="relu"), layers.Dense(1, activation="linear")])

    def call(self, x, training=False):
        x = self.input_proj(x) + self.pos_embed(tf.range(tf.shape(x)[1]))
        x = self.lstm_proj(self.bilstm(x, training=training))
        for block in self.blocks: x = block(x, training=training)
        return tf.squeeze(self.tower(self.global_pool(x), training=training), axis=-1)

    def get_config(self):
        return {"n_features": self.n_features, "context_window": self.context_window}

def build_titan(n_features: int, context_window: int = 150) -> keras.Model:
    model = Titan(n_features=n_features, context_window=context_window)
    model.compile(optimizer=keras.optimizers.Adam(5e-4, clipnorm=1.0), loss="huber", metrics=["mae"])
    model(tf.zeros((1, context_window, n_features)))
    print(f"TITAN Engine — Built | Params: {model.count_params():,}")
    return model
