"""
HYDRA — Sovereign Quant Engine (V3.7.1 HARDENED)
==================================================
Mission: Stability & Serialization Fixes.
Status: Hardened for 300-epoch mission saving.
"""

import tensorflow as tf
import keras
from keras import layers, regularizers
import numpy as np

@keras.saving.register_keras_serializable(package="KAT")
class SovereignLoss(keras.losses.Loss):
    def __init__(self, delta=1.0, direction_weight=2.0, **kwargs):
        super().__init__(**kwargs)
        self.delta, self.direction_weight = delta, direction_weight
    def call(self, y_true, y_pred):
        err = y_true - y_pred
        huber = tf.where(tf.abs(err) <= self.delta, 0.5 * tf.square(err), self.delta * (tf.abs(err) - 0.5 * self.delta))
        return tf.reduce_mean(tf.where(tf.sign(y_true) != tf.sign(y_pred), self.direction_weight * huber, huber))
    def get_config(self):
        config = super().get_config()
        config.update({"delta": self.delta, "direction_weight": self.direction_weight})
        return config

@keras.utils.register_keras_serializable(package="KAT")
class RMSNorm(layers.Layer):
    def __init__(self, epsilon=1e-6, **kwargs):
        super().__init__(**kwargs)
        self.eps = epsilon
    def build(self, input_shape):
        self.scale = self.add_weight(name="scale", shape=(input_shape[-1],), initializer="ones")
        super().build(input_shape)
    def call(self, x):
        v = tf.reduce_mean(tf.square(x), axis=-1, keepdims=True)
        return self.scale * tf.math.rsqrt(v + self.eps) * x
    def get_config(self):
        config = super().get_config()
        config.update({"epsilon": self.eps})
        return config

@keras.utils.register_keras_serializable(package="KAT")
class VSN(layers.Layer):
    def __init__(self, d_model: int, n_features: int, **kwargs):
        super().__init__(**kwargs)
        self.d_model, self.n_features = d_model, n_features
        self.feature_projs = [layers.Dense(d_model) for _ in range(n_features)]
        self.gate_proj = layers.Dense(n_features, activation="softmax")
    def build(self, input_shape):
        self.gate_proj.build(input_shape)
        for p in self.feature_projs: p.build((None, None, 1))
        super().build(input_shape)
    def call(self, x):
        p_list = [self.feature_projs[i](tf.expand_dims(x[:, :, i], -1)) for i in range(self.n_features)]
        w = self.gate_proj(x)
        return tf.reduce_sum(tf.stack(p_list, axis=2) * tf.expand_dims(w, -1), axis=2)
    def get_config(self):
        config = super().get_config()
        config.update({"d_model": self.d_model, "n_features": self.n_features})
        return config

@keras.utils.register_keras_serializable(package="KAT")
class DualScaleFusion(layers.Layer):
    def __init__(self, d_model: int, scale_factor=15, **kwargs):
        super().__init__(**kwargs)
        self.d_model, self.scale_factor = d_model, scale_factor
        self.macro_proj = layers.Dense(d_model)
        self.fusion = layers.Dense(d_model)
    def build(self, input_shape):
        self.macro_proj.build(input_shape)
        self.fusion.build((None, None, self.d_model*2))
        super().build(input_shape)
    def call(self, x):
        m_x = layers.AveragePooling1D(pool_size=self.scale_factor, strides=1, padding="same")(x)
        return self.fusion(tf.concat([x, self.macro_proj(m_x)], axis=-1))
    def get_config(self):
        config = super().get_config()
        config.update({"d_model": self.d_model, "scale_factor": self.scale_factor})
        return config

@keras.utils.register_keras_serializable(package="KAT")
class TemporalGating(layers.Layer):
    def __init__(self, n_blocks=5, **kwargs):
        super().__init__(**kwargs)
        self.n_blocks = n_blocks
        self.gate_proj = layers.Dense(n_blocks, activation="softmax")
    def build(self, input_shape):
        self.gate_proj.build((None, self.n_blocks, input_shape[-1]))
        super().build(input_shape)
    def call(self, x):
        B, T, D = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2]
        size = T // self.n_blocks
        t_len = self.n_blocks * size
        r = tf.reshape(x[:, :t_len, :], (B, self.n_blocks, size, D))
        imp = tf.reduce_mean(self.gate_proj(tf.reduce_mean(r, axis=2)), axis=-1, keepdims=True)
        g_f = tf.reshape(r * tf.expand_dims(imp, -1), (B, t_len, D))
        g_f = tf.pad(g_f, [[0, 0], [0, T - t_len], [0, 0]])
        g_f.set_shape(x.shape)
        return g_f
    def get_config(self):
        config = super().get_config()
        config.update({"n_blocks": self.n_blocks})
        return config

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
        s = tf.stack(memory_list + [current_x], axis=1) 
        w = tf.nn.softmax(self.proj(self.norm(s)), axis=1)
        return tf.reduce_sum(w * s, axis=1)
    def get_config(self):
        config = super().get_config()
        config.update({"d_model": self.d_model})
        return config

@keras.utils.register_keras_serializable(package="KAT")
class MLAAttention(layers.Layer):
    def __init__(self, d_model: int, n_heads: int, latent_dim: int = 32, **kwargs):
        super().__init__(**kwargs)
        self.d_model, self.n_heads, self.latent_dim = d_model, n_heads, latent_dim
        self.d_head = d_model // n_heads
        self.kv_compress = layers.Dense(latent_dim, use_bias=False)
        self.k_up, self.v_up, self.q_proj, self.o_proj = [layers.Dense(d_model, use_bias=False) for _ in range(4)]
    def build(self, input_shape):
        self.kv_compress.build(input_shape)
        self.k_up.build((None, None, self.latent_dim))
        self.v_up.build((None, None, self.latent_dim))
        self.q_proj.build(input_shape)
        self.o_proj.build(input_shape)
        super().build(input_shape)
    def call(self, x, training=False):
        B, T = tf.shape(x)[0], tf.shape(x)[1]
        l = self.kv_compress(x)
        def sp(t): return tf.transpose(tf.reshape(t, (B, T, self.n_heads, self.d_head)), perm=[0, 2, 1, 3])
        K, V, Q = sp(self.k_up(l)), sp(self.v_up(l)), sp(self.q_proj(x))
        s = tf.matmul(Q, K, transpose_b=True) / tf.math.sqrt(tf.cast(self.d_head, tf.float32))
        s += (1.0 - tf.linalg.band_part(tf.ones((T, T)), -1, 0)) * -1e9
        out = tf.reshape(tf.transpose(tf.matmul(tf.nn.softmax(s, axis=-1), V), perm=[0, 2, 1, 3]), (B, T, -1))
        return self.o_proj(out)
    def get_config(self):
        config = super().get_config()
        config.update({"d_model": self.d_model, "n_heads": self.n_heads, "latent_dim": self.latent_dim})
        return config

@keras.utils.register_keras_serializable(package="KAT")
class GatedMoE(layers.Layer):
    def __init__(self, d_model: int, n_experts: int = 4, expert_dim: int = 64, **kwargs):
        super().__init__(**kwargs)
        self.d_model, self.n_experts, self.expert_dim = d_model, n_experts, expert_dim
        self.router = layers.Dense(n_experts, activation="softmax")
        self.gate = layers.Dense(d_model, activation="sigmoid") 
        self.experts = [keras.Sequential([layers.Dense(expert_dim, activation="gelu"), layers.Dense(d_model)]) for _ in range(n_experts)]
    def build(self, input_shape):
        self.router.build(input_shape)
        self.gate.build(input_shape)
        for e in self.experts: e.build(input_shape)
        super().build(input_shape)
    def call(self, x, training=False):
        w = self.router(x)
        if training:
            aw = tf.reduce_mean(w, axis=[0, 1])
            self.add_loss(0.02 * tf.reduce_sum(aw * tf.math.log(aw + 1e-9))) 
        xg = x * self.gate(x)
        eo = tf.stack([e(xg) for e in self.experts], axis=2)
        return tf.reduce_sum(eo * tf.expand_dims(w, -1), axis=2)
    def get_config(self):
        config = super().get_config()
        config.update({"d_model": self.d_model, "n_experts": self.n_experts, "expert_dim": self.expert_dim})
        return config

@keras.utils.register_keras_serializable(package="KAT")
class HydraBlock(layers.Layer):
    def __init__(self, d_model: int, n_heads: int, **kwargs):
        super().__init__(**kwargs)
        self.d_model, self.n_heads = d_model, n_heads
        self.t_gate, self.dual_scale = TemporalGating(5), DualScaleFusion(d_model)
        self.attn, self.moe = MLAAttention(d_model, n_heads), GatedMoE(d_model)
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
        x_s = self.dual_scale(self.t_gate(x))
        x = self.norm1(x + self.attn(x_s, training=training))
        return self.norm2(x + self.moe(x, training=training))
    def get_config(self):
        config = super().get_config()
        config.update({"d_model": self.d_model, "n_heads": self.n_heads})
        return config

@keras.utils.register_keras_serializable(package="KAT")
class Hydra(keras.Model):
    def __init__(self, n_features: int, context_window: int = 360, d_model: int = 128, mtp_steps: int = 5, n_blocks: int = 8, **kwargs):
        super().__init__(**kwargs)
        self.n_features, self.context_window, self.d_model, self.mtp_steps, self.n_blocks = n_features, context_window, d_model, mtp_steps, n_blocks
        self.vsn = VSN(d_model, n_features)
        self.pos_embed = layers.Embedding(context_window + 512, d_model)
        self.res_controllers = [AttnRes(d_model) for _ in range(n_blocks)]
        self.blocks = [HydraBlock(d_model, 4) for _ in range(n_blocks)]
        self.norm_out, self.mtp_head = RMSNorm(), layers.Dense(mtp_steps, activation="linear")
    def build(self, input_shape):
        self.vsn.build(input_shape)
        dummy = (input_shape[0], input_shape[1], self.d_model)
        for res in self.res_controllers: res.build(dummy)
        for block in self.blocks: block.build(dummy)
        self.norm_out.build(dummy)
        self.mtp_head.build(dummy)
        super().build(input_shape)
    def call(self, x, training=False):
        x = self.vsn(x) + self.pos_embed(tf.range(tf.shape(x)[1]))
        memory = [x]
        for i in range(self.n_blocks):
            x = self.res_controllers[i](memory, x)
            x = self.blocks[i](x, training=training)
            memory.append(x)
        return self.mtp_head(self.norm_out(x[:, -1:, :]))
    def get_config(self):
        config = super().get_config()
        config.update({
            "n_features": self.n_features,
            "context_window": self.context_window,
            "d_model": self.d_model,
            "mtp_steps": self.mtp_steps,
            "n_blocks": self.n_blocks,
        })
        return config

def build_hydra(n_features: int, context_window: int = 360) -> Hydra:
    model = Hydra(n_features=n_features, context_window=context_window, n_blocks=8)
    model.compile(optimizer=keras.optimizers.Adam(5e-4, clipnorm=1.0), loss=SovereignLoss(), metrics=["mae"])
    model(tf.zeros((1, context_window - 5, n_features))) 
    return model
