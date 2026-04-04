"""
HYDRA — Next-Gen Autoregressive Forecaster (MoE + MLA + MTP + AttnRes)
====================================================================
Architecture:
  - Multi-Head Latent Attention (MLA)
  - Mixture of Experts (MoE) 
  - Multi-Token Prediction (MTP)
  - Attention Residuals (AttnRes)
  - RMSNorm (Stability for extreme depth)
  - Depth: 16 Blocks (Elite Configuration)
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
class RMSNorm(layers.Layer):
    def __init__(self, epsilon=1e-6, **kwargs):
        super().__init__(**kwargs)
        self.eps = epsilon

    def build(self, input_shape):
        if not hasattr(self, 'scale'):
            self.scale = self.add_weight(name="scale", shape=(input_shape[-1],), initializer="ones")
        super().build(input_shape)

    def call(self, x):
        variance = tf.reduce_mean(tf.square(x), axis=-1, keepdims=True)
        x = x * tf.math.rsqrt(variance + self.eps)
        return self.scale * x

@keras.utils.register_keras_serializable(package="KAT")
class AttnRes(layers.Layer):
    def __init__(self, d_model: int, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.proj = layers.Dense(1, use_bias=False)
        self.norm = RMSNorm()

    def build(self, input_shape):
        self.proj.build(input_shape)
        self.norm.build(input_shape)
        super().build(input_shape)

    def call(self, memory_list, current_x):
        all_states = memory_list + [current_x]
        V = tf.stack(all_states, axis=1) # [B, N+1, T, D]
        K = self.norm(V)
        logits = self.proj(K)
        weights = tf.nn.softmax(logits, axis=1)
        return tf.reduce_sum(weights * V, axis=1)

    def get_config(self):
        return {**super().get_config(), "d_model": self.d_model}

@keras.utils.register_keras_serializable(package="KAT")
class MLAAttention(layers.Layer):
    def __init__(self, d_model: int, n_heads: int, latent_dim: int = 32, dropout: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.n_heads, self.d_head, self.latent_dim = n_heads, d_model // n_heads, latent_dim
        self.kv_compress = layers.Dense(latent_dim, use_bias=False)
        self.k_up, self.v_up, self.q_proj, self.o_proj = [layers.Dense(d_model, use_bias=False) for _ in range(4)]
        self.drop = layers.Dropout(dropout)

    def build(self, input_shape):
        B, T, D = input_shape
        self.kv_compress.build(input_shape)
        self.k_up.build((B, T, self.latent_dim))
        self.v_up.build((B, T, self.latent_dim))
        self.q_proj.build(input_shape)
        self.o_proj.build(input_shape)
        super().build(input_shape)

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
    def __init__(self, d_model: int, n_experts: int = 4, expert_dim: int = 64, dropout: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.n_experts = n_experts
        self.router = layers.Dense(n_experts, activation="softmax")
        self.experts = [keras.Sequential([layers.Dense(expert_dim, activation="gelu"), layers.Dense(d_model), layers.Dropout(dropout)]) for _ in range(n_experts)]

    def build(self, input_shape):
        self.router.build(input_shape)
        for expert in self.experts: expert.build(input_shape)
        super().build(input_shape)

    def call(self, x, training=False):
        weights = self.router(x)
        stacked = tf.stack([self.experts[i](x, training=training) for i in range(self.n_experts)], axis=2)
        return tf.reduce_sum(stacked * tf.expand_dims(weights, -1), axis=2)

    def get_config(self):
        return {**super().get_config(), "n_experts": self.n_experts}

@keras.utils.register_keras_serializable(package="KAT")
class HydraBlock(layers.Layer):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.attn, self.moe = MLAAttention(d_model, n_heads, dropout=dropout), TinyMoE(d_model, expert_dim=d_model*2, dropout=dropout)
        self.norm1, self.norm2 = RMSNorm(), RMSNorm()

    def build(self, input_shape):
        self.attn.build(input_shape)
        self.moe.build(input_shape)
        self.norm1.build(input_shape)
        self.norm2.build(input_shape)
        super().build(input_shape)

    def call(self, x, training=False):
        x = self.norm1(x + self.attn(x, training=training))
        return self.norm2(x + self.moe(x, training=training))

@keras.utils.register_keras_serializable(package="KAT")
class Hydra(keras.Model):
    def __init__(self, n_features: int, context_window: int = 360, d_model: int = 128, mtp_steps: int = 5, n_blocks: int = 16, dropout: float = 0.1, name: str = "HYDRA", **kwargs):
        super().__init__(name=name, **kwargs)
        self.n_features, self.context_window, self.mtp_steps, self.n_blocks = n_features, context_window, mtp_steps, n_blocks
        self.input_proj = layers.Dense(d_model)
        self.pos_embed = layers.Embedding(context_window + 512, d_model)
        self.res_controllers = [AttnRes(d_model, name=f"attn_res_{i}") for i in range(n_blocks)]
        self.blocks = [HydraBlock(d_model, 4, dropout, name=f"hydra_block_{i}") for i in range(n_blocks)]
        self.norm_out, self.mtp_head = RMSNorm(), layers.Dense(mtp_steps, activation="linear")

    def build(self, input_shape):
        B, T, F = input_shape
        self.input_proj.build(input_shape)
        d_model = self.input_proj.compute_output_shape(input_shape)[-1]
        dummy_shape = (B, T, d_model)
        for res in self.res_controllers: res.build(dummy_shape)
        for block in self.blocks: block.build(dummy_shape)
        self.norm_out.build(dummy_shape)
        self.mtp_head.build(dummy_shape)
        super().build(input_shape)

    def call(self, x, training=False):
        x = self.input_proj(x) + self.pos_embed(tf.range(tf.shape(x)[1]))
        memory = [x] # Start memory with the projected input
        for i in range(self.n_blocks):
            # All blocks now engage their AttnRes controller
            # at i=0, it attends over the input projection itself
            x = self.res_controllers[i](memory, x)
            x = self.blocks[i](x, training=training)
            memory.append(x)
        return self.mtp_head(self.norm_out(x))

    def generate(self, seed, steps=60, scaler=None):
        ctx = seed.copy()
        preds = []
        for _ in range(steps):
            inp = tf.constant(ctx[np.newaxis], dtype=tf.float32)
            out = self(inp, training=False)
            next_p = float(out[0, -1, 0])
            preds.append(next_p)
            new_row = ctx[-1].copy()
            new_row[3] = next_p
            ctx = np.vstack([ctx[1:], new_row])
        return scaler.inverse_y(np.array(preds)) if scaler else np.array(preds)

    def get_config(self):
        return {"n_features": self.n_features, "context_window": self.context_window, "mtp_steps": self.mtp_steps, "n_blocks": self.n_blocks}

def build_hydra(n_features: int, context_window: int = 360) -> Hydra:
    model = Hydra(n_features=n_features, context_window=context_window, n_blocks=16)
    model.compile(optimizer=keras.optimizers.Adam(1e-3, clipnorm=1.0), loss=directional_huber_loss, metrics=["mae"])
    model(tf.zeros((1, context_window, n_features)))
    print(f"HYDRA Engine (16B-Elite) — Built with RMSNorm & AttnRes | Params: {model.count_params():,}")
    return model
