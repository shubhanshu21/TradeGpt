"""
CAUSAL — GPT-Style Autoregressive Forecaster (200K params)
==========================================================
Architecture:
  - Causal (decoder-only) transformer
  - Predicts next price token, feeds prediction back as input
  - Generates a full multi-step trajectory
"""

import tensorflow as tf
import keras
from keras import layers, regularizers
import numpy as np

@keras.utils.register_keras_serializable()
class CausalSelfAttention(layers.Layer):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.n_heads, self.d_head, self.d_model = n_heads, d_model // n_heads, d_model
        self.Wq, self.Wk, self.Wv, self.Wo = [layers.Dense(d_model, use_bias=False) for _ in range(4)]
        self.drop = layers.Dropout(dropout)

    def call(self, x, training=False):
        B, T = tf.shape(x)[0], tf.shape(x)[1]
        Q, K, V = self._split(self.Wq(x)), self._split(self.Wk(x)), self._split(self.Wv(x))
        scores = tf.matmul(Q, K, transpose_b=True) / tf.math.sqrt(tf.cast(self.d_head, tf.float32))
        scores += (1.0 - tf.linalg.band_part(tf.ones((T, T)), -1, 0)) * -1e9
        out = tf.reshape(tf.transpose(tf.matmul(tf.nn.softmax(scores, axis=-1), V), perm=[0, 2, 1, 3]), (B, T, self.d_model))
        return self.Wo(out)

    def _split(self, x):
        B, T = tf.shape(x)[0], tf.shape(x)[1]
        return tf.transpose(tf.reshape(x, (B, T, self.n_heads, self.d_head)), perm=[0, 2, 1, 3])

@keras.utils.register_keras_serializable()
class CausalBlock(layers.Layer):
    def __init__(self, d_model: int, n_heads: int, ff_mult: int = 4, dropout: float = 0.1):
        super().__init__()
        self.attn, self.norm1, self.norm2 = CausalSelfAttention(d_model, n_heads, dropout), layers.LayerNormalization(epsilon=1e-6), layers.LayerNormalization(epsilon=1e-6)
        self.ff = keras.Sequential([layers.Dense(d_model * ff_mult, activation="gelu"), layers.Dense(d_model), layers.Dropout(dropout)])

    def call(self, x, training=False):
        x = self.norm1(x + self.attn(x, training=training))
        return self.norm2(x + self.ff(x, training=training))

@keras.utils.register_keras_serializable()
class CausalModel(keras.Model):
    def __init__(self, n_features: int, context_window: int = 150, d_model: int = 128, n_heads: int = 4, n_layers: int = 4, dropout: float = 0.1, name: str = "CAUSAL"):
        super().__init__(name=name)
        self.context_window, self.n_features = context_window, n_features
        self.input_proj, self.pos_embed = layers.Dense(d_model), layers.Embedding(context_window + 512, d_model)
        self.blocks = [CausalBlock(d_model, n_heads, dropout=dropout) for _ in range(n_layers)]
        self.norm_out, self.head = layers.LayerNormalization(epsilon=1e-6), layers.Dense(1, activation="linear")

    def call(self, x, training=False):
        x = self.input_proj(x) + self.pos_embed(tf.range(tf.shape(x)[1]))
        for block in self.blocks: x = block(x, training=training)
        return self.head(self.norm_out(x))

    def generate(self, seed, steps=60, scaler=None):
        ctx = seed.copy()
        preds = []
        for _ in range(steps):
            next_p = float(self(tf.constant(ctx[np.newaxis], dtype=tf.float32), training=False)[0, -1, 0])
            preds.append(next_p)
            new_row = ctx[-1].copy()
            new_row[3] = next_p
            ctx = np.vstack([ctx[1:], new_row])
        return scaler.inverse_y(np.array(preds)) if scaler else np.array(preds)

    def get_config(self):
        return {"n_features": self.n_features, "context_window": self.context_window}

def build_causal(n_features, variant="base"):
    ctx_map = {"base": 150, "lion": 480, "tiger": 1440}
    ctx = ctx_map.get(variant, 150)
    model = CausalModel(n_features=n_features, context_window=ctx)
    model.compile(optimizer=keras.optimizers.Adam(1e-3, clipnorm=1.0), loss="mse", metrics=["mae"])
    model(tf.zeros((1, ctx, n_features)))
    print(f"CAUSAL Engine [{variant}] — Built | Params: {model.count_params():,} | Context: {ctx}min")
    return model

def prepare_causal_targets(X):
    return X[:, :-1, :], X[:, 1:, 3:4]
