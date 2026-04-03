"""
HYDRA — Next-Gen Autoregressive Forecaster (MoE + MLA + MTP)
===========================================================
Architecture:
  - Multi-Head Latent Attention (MLA)
  - Mixture of Experts (MoE)
  - Multi-Token Prediction (MTP)
"""

import tensorflow as tf
import keras
from keras import layers, regularizers
import numpy as np

@keras.saving.register_keras_serializable(package="KAT")
def directional_huber_loss(y_true, y_pred, delta=1.0, direction_weight=2.0):
    err = y_true - y_pred
    is_small_error = tf.abs(err) <= delta
    huber = tf.where(is_small_error, 0.5 * tf.square(err), delta * (tf.abs(err) - 0.5 * delta))
    return tf.reduce_mean(tf.where(tf.sign(y_true) != tf.sign(y_pred), direction_weight * huber, huber))

@keras.utils.register_keras_serializable(package="KAT")
class MLAAttention(layers.Layer):
    def __init__(self, d_model: int, n_heads: int, latent_dim: int = 32, dropout: float = 0.1):
        super().__init__()
        self.n_heads, self.d_head, self.latent_dim = n_heads, d_model // n_heads, latent_dim
        self.kv_compress = layers.Dense(latent_dim, use_bias=False)
        self.k_up, self.v_up, self.q_proj, self.o_proj = [layers.Dense(d_model, use_bias=False) for _ in range(4)]
        self.drop = layers.Dropout(dropout)

    def call(self, x, training=False):
        B, T = tf.shape(x)[0], tf.shape(x)[1]
        latent = self.kv_compress(x)
        K, V, Q = self._split_heads(self.k_up(latent)), self._split_heads(self.v_up(latent)), self._split_heads(self.q_proj(x))
        scores = tf.matmul(Q, K, transpose_b=True) / tf.math.sqrt(tf.cast(self.d_head, tf.float32))
        scores += (1.0 - tf.linalg.band_part(tf.ones((T, T)), -1, 0)) * -1e9
        attn_out = tf.reshape(tf.transpose(tf.matmul(tf.nn.softmax(scores, axis=-1), V), perm=[0, 2, 1, 3]), (B, T, -1))
        return self.o_proj(attn_out)

    def _split_heads(self, x):
        B, T = tf.shape(x)[0], tf.shape(x)[1]
        return tf.transpose(tf.reshape(x, (B, T, self.n_heads, self.d_head)), perm=[0, 2, 1, 3])

    def get_config(self):
        return {**super().get_config(), "d_model": self.n_heads * self.d_head, "n_heads": self.n_heads, "latent_dim": self.latent_dim}

@keras.utils.register_keras_serializable(package="KAT")
class TinyMoE(layers.Layer):
    def __init__(self, d_model: int, n_experts: int = 4, expert_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        self.n_experts = n_experts
        self.router = layers.Dense(n_experts, activation="softmax")
        self.experts = [keras.Sequential([layers.Dense(expert_dim, activation="gelu"), layers.Dense(d_model), layers.Dropout(dropout)]) for _ in range(n_experts)]

    def call(self, x, training=False):
        weights = self.router(x)
        stacked = tf.stack([self.experts[i](x, training=training) for i in range(self.n_experts)], axis=2)
        return tf.reduce_sum(stacked * tf.expand_dims(weights, -1), axis=2)

    def get_config(self):
        return {**super().get_config(), "n_experts": self.n_experts}

@keras.utils.register_keras_serializable(package="KAT")
class HydraBlock(layers.Layer):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.attn, self.moe = MLAAttention(d_model, n_heads, dropout=dropout), TinyMoE(d_model, expert_dim=d_model*2, dropout=dropout)
        self.norm1, self.norm2 = layers.LayerNormalization(epsilon=1e-6), layers.LayerNormalization(epsilon=1e-6)

    def call(self, x, training=False):
        x = self.norm1(x + self.attn(x, training=training))
        return self.norm2(x + self.moe(x, training=training))

@keras.utils.register_keras_serializable(package="KAT")
class Hydra(keras.Model):
    def __init__(self, n_features: int, context_window: int = 360, d_model: int = 128, mtp_steps: int = 5, dropout: float = 0.1, name: str = "HYDRA"):
        super().__init__(name=name)
        self.n_features, self.context_window, self.mtp_steps = n_features, context_window, mtp_steps
        self.input_proj, self.pos_embed = layers.Dense(d_model), layers.Embedding(context_window + 512, d_model)
        self.blocks = [HydraBlock(d_model, 4, dropout) for _ in range(4)]
        self.norm_out, self.mtp_head = layers.LayerNormalization(epsilon=1e-6), layers.Dense(mtp_steps, activation="linear")

    def call(self, x, training=False):
        x = self.input_proj(x) + self.pos_embed(tf.range(tf.shape(x)[1]))
        for block in self.blocks: x = block(x, training=training)
        return self.mtp_head(self.norm_out(x))

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
        return {"n_features": self.n_features, "context_window": self.context_window, "mtp_steps": self.mtp_steps}

def build_hydra(n_features: int, context_window: int = 360) -> Hydra:
    model = Hydra(n_features=n_features, context_window=context_window)
    model.compile(optimizer=keras.optimizers.Adam(1e-3, clipnorm=1.0), loss=directional_huber_loss, metrics=["mae"])
    model(tf.zeros((1, context_window, n_features)))
    print(f"HYDRA Engine — Built with Directional Huber Loss | Params: {model.count_params():,}")
    return model
