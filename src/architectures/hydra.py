"""
HYDRA — Sovereign Sovereign Quant Engine (VSN + DualScale + KimiMoE + MoBA)
========================================================================
Architecture:
  - Variable Selection Network (VSN): Dynamic feature-level importance gating.
  - Dual-Scale Fusion: Fusing 1m High-Frequency with 15m Macro-Trends.
  - Active Expert Balancing: Expert-diversity loss for optimal regime mastery.
  - MoBA-inspired Temporal Gating: Selective Attention over 30-min Blocks.
  - Kimi-Linear inspired Routing: Stably gated Mixture of Experts.
"""

import tensorflow as tf
import keras
from keras import layers, regularizers
import numpy as np

@keras.saving.register_keras_serializable(package="KAT")
class SovereignLoss(keras.losses.Loss):
    def __init__(self, delta=1.0, direction_weight=2.0, balance_weight=0.1, **kwargs):
        super().__init__(**kwargs)
        self.delta, self.direction_weight, self.balance_weight = delta, direction_weight, balance_weight

    def call(self, y_true, y_pred):
        err = y_true - y_pred
        is_small_error = tf.abs(err) <= self.delta
        huber = tf.where(is_small_error, 0.5 * tf.square(err), self.delta * (tf.abs(err) - 0.5 * self.delta))
        base_loss = tf.reduce_mean(tf.where(tf.sign(y_true) != tf.sign(y_pred), self.direction_weight * huber, huber))
        return base_loss

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

    def compute_output_shape(self, input_shape):
        return input_shape

@keras.utils.register_keras_serializable(package="KAT")
class VSN(layers.Layer):
    def __init__(self, d_model: int, n_features: int, **kwargs):
        super().__init__(**kwargs)
        self.d_model, self.n_features = d_model, n_features
        self.feature_projs = [layers.Dense(d_model) for _ in range(n_features)]
        self.gate_proj = layers.Dense(n_features, activation="softmax")

    def build(self, input_shape):
        for proj in self.feature_projs: proj.build((input_shape[0], input_shape[1], 1))
        self.gate_proj.build(input_shape)
        super().build(input_shape)

    def call(self, x):
        projs = []
        for i in range(self.n_features):
            feat = tf.expand_dims(x[:, :, i], -1)
            projs.append(self.feature_projs[i](feat))
        weights = self.gate_proj(x)
        stacked = tf.stack(projs, axis=2) 
        return tf.reduce_sum(stacked * tf.expand_dims(weights, -1), axis=2)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], self.d_model)

    def get_config(self):
        return {**super().get_config(), "d_model": self.d_model, "n_features": self.n_features}

@keras.utils.register_keras_serializable(package="KAT")
class DualScaleFusion(layers.Layer):
    def __init__(self, d_model: int, scale_factor=15, **kwargs):
        super().__init__(**kwargs)
        self.scale_factor = scale_factor
        self.macro_proj = layers.Dense(d_model)
        self.fusion = layers.Dense(d_model)

    def build(self, input_shape):
        self.macro_proj.build(input_shape)
        self.fusion.build((input_shape[0], input_shape[1], input_shape[2]*2))
        super().build(input_shape)

    def call(self, x):
        macro_x = layers.AveragePooling1D(pool_size=self.scale_factor, strides=1, padding="same")(x)
        macro_feat = self.macro_proj(macro_x)
        return self.fusion(tf.concat([x, macro_feat], axis=-1))

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        return {**super().get_config(), "scale_factor": self.scale_factor}

@keras.utils.register_keras_serializable(package="KAT")
class TemporalGating(layers.Layer):
    def __init__(self, n_blocks=5, **kwargs):
        super().__init__(**kwargs)
        self.n_blocks = n_blocks
        self.gate_proj = layers.Dense(n_blocks, activation="softmax")

    def build(self, input_shape):
        self.gate_proj.build((input_shape[0], self.n_blocks, input_shape[2]))
        super().build(input_shape)

    def call(self, x):
        T_dyn, D_dyn, B_dyn = tf.shape(x)[1], tf.shape(x)[2], tf.shape(x)[0]
        block_size = T_dyn // self.n_blocks
        target_len = self.n_blocks * block_size
        x_sliced = x[:, :target_len, :]
        reshaped = tf.reshape(x_sliced, (B_dyn, self.n_blocks, block_size, D_dyn))
        block_means = tf.reduce_mean(reshaped, axis=2) 
        importance = self.gate_proj(block_means) 
        importance = tf.reduce_mean(importance, axis=-1, keepdims=True) 
        gated = reshaped * tf.expand_dims(importance, -1) 
        gated_flat = tf.reshape(gated, (B_dyn, target_len, D_dyn))
        pad_size = T_dyn - target_len
        gated_flat = tf.pad(gated_flat, [[0, 0], [0, pad_size], [0, 0]])
        gated_flat.set_shape(x.shape)
        return gated_flat

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        return {**super().get_config(), "n_blocks": self.n_blocks}

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
        V = tf.stack(all_states, axis=1) 
        K = self.norm(V)
        logits = self.proj(K)
        weights = tf.nn.softmax(logits, axis=1)
        return tf.reduce_sum(weights * V, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape

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

    def compute_output_shape(self, input_shape):
        return input_shape

    def _split_heads(self, x):
        B, T = tf.shape(x)[0], tf.shape(x)[1]
        return tf.transpose(tf.reshape(x, (B, T, self.n_heads, self.d_head)), perm=[0, 2, 1, 3])

    def get_config(self):
        return {**super().get_config(), "d_model": self.n_heads * self.d_head, "n_heads": self.n_heads, "latent_dim": self.latent_dim}

@keras.utils.register_keras_serializable(package="KAT")
class GatedMoE(layers.Layer):
    def __init__(self, d_model: int, n_experts: int = 4, expert_dim: int = 64, dropout: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.n_experts = n_experts
        self.router = layers.Dense(n_experts, activation="softmax")
        self.gate = layers.Dense(d_model, activation="sigmoid") 
        self.experts = [keras.Sequential([layers.Dense(expert_dim, activation="gelu"), layers.Dense(d_model), layers.Dropout(dropout)]) for _ in range(n_experts)]

    def build(self, input_shape):
        self.router.build(input_shape)
        self.gate.build(input_shape)
        for expert in self.experts: expert.build(input_shape)
        super().build(input_shape)

    def call(self, x, training=False):
        weights = self.router(x)
        if training:
            avg_weights = tf.reduce_mean(weights, axis=[0, 1])
            balance_loss = tf.reduce_sum(avg_weights * tf.math.log(avg_weights + 1e-9))
            self.add_loss(0.01 * balance_loss) 
        x_gated = x * self.gate(x)
        stacked = tf.stack([self.experts[i](x_gated, training=training) for i in range(self.n_experts)], axis=2)
        return tf.reduce_sum(stacked * tf.expand_dims(weights, -1), axis=2)

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        return {**super().get_config(), "n_experts": self.n_experts}

@keras.utils.register_keras_serializable(package="KAT")
class HydraBlock(layers.Layer):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.t_gate = TemporalGating(n_blocks=5)
        self.dual_scale = DualScaleFusion(d_model, scale_factor=15)
        self.attn, self.moe = MLAAttention(d_model, n_heads, dropout=dropout), GatedMoE(d_model, expert_dim=d_model*2, dropout=dropout)
        self.norm1, self.norm2 = RMSNorm(), RMSNorm()

    def build(self, input_shape):
        self.t_gate.build(input_shape)
        self.dual_scale.build(input_shape)
        self.attn.build(input_shape)
        self.moe.build(input_shape)
        self.norm1.build(input_shape)
        self.norm2.build(input_shape)
        super().build(input_shape)

    def call(self, x, training=False):
        x_filtered = self.t_gate(x)
        x_fusion   = self.dual_scale(x_filtered)
        x = self.norm1(x + self.attn(x_fusion, training=training))
        return self.norm2(x + self.moe(x, training=training))

    def compute_output_shape(self, input_shape):
        return input_shape

@keras.utils.register_keras_serializable(package="KAT")
class Hydra(keras.Model):
    def __init__(self, n_features: int, context_window: int = 360, d_model: int = 128, mtp_steps: int = 5, n_blocks: int = 16, dropout: float = 0.1, name: str = "HYDRA", **kwargs):
        super().__init__(name=name, **kwargs)
        self.n_features, self.context_window, self.mtp_steps, self.n_blocks = n_features, context_window, mtp_steps, n_blocks
        self.vsn = VSN(d_model, n_features)
        self.pos_embed = layers.Embedding(context_window + 512, d_model)
        self.res_controllers = [AttnRes(d_model, name=f"attn_res_{i}") for i in range(n_blocks)]
        self.blocks = [HydraBlock(d_model, 4, dropout, name=f"hydra_block_{i}") for i in range(n_blocks)]
        self.norm_out, self.mtp_head = RMSNorm(), layers.Dense(mtp_steps, activation="linear")

    def build(self, input_shape):
        batch, seq, feat = input_shape
        self.vsn.build(input_shape)
        vsn_out_shape = self.vsn.compute_output_shape(input_shape)
        d_model = vsn_out_shape[2]
        dummy_shape = (batch, seq, d_model)
        for res in self.res_controllers: res.build(dummy_shape)
        for block in self.blocks: block.build(dummy_shape)
        self.norm_out.build(dummy_shape)
        self.mtp_head.build(dummy_shape)
        super().build(input_shape)

    def call(self, x, training=False):
        x = self.vsn(x) + self.pos_embed(tf.range(tf.shape(x)[1]))
        memory = [x]
        for i in range(self.n_blocks):
            # FIXED: All res_controllers must be used to participate in training
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
    model.compile(optimizer=keras.optimizers.Adam(1e-3, clipnorm=1.0), loss=SovereignLoss(), metrics=["mae"])
    model(tf.zeros((1, context_window - 5, n_features))) 
    print(f"HYDRA Sovereign Quant Engine (V3.5) — Built with VSN & DualScale Fusion | Params: {model.count_params():,}")
    return model
