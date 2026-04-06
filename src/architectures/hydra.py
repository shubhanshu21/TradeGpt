"""
HYDRA SOVEREIGN KRAKEN (V4.2) - MASTERY EDITION ⚓🚀⚡
==========================================================
- Architecture: MoE + MLA + RoPE + MTP (Adaptive)
- Features: 
    1. RoPE (Rotary Positional Embeddings) for relative time awareness.
    2. Auxiliary Head (MTP-3): Predicts Price, Volatility, and Volume Imbalance.
    3. Regime-Aware Gating: Contextual routing for experts.
- Hardware: Auto-Detect (CPU / NVIDIA A40)
"""

import os, time
import numpy as np
import tensorflow as tf
import keras
from keras import layers, ops
from typing import Optional

# Lazy Hardware Initializer
IS_GPU = len(tf.config.list_physical_devices('GPU')) > 0

def init_kraken_hardware():
    if IS_GPU:
        try:
            from tensorflow.keras import mixed_precision
            mixed_precision.set_global_policy('mixed_float16')
            print(f"🚀 KRAKEN: NVIDIA GPU DETECTED. Mixed Precision ENABLED (A40 Mode).")
        except:
            pass
    else:
        print(f"🐌 KRAKEN: NO GPU DETECTED. Running in CPU-Lite Mode.")

def apply_rope(x, head_dim):
    """
    Apply Rotary Positional Embeddings (RoPE).
    Ensures the model understands the RELATIVE distance between candles.
    """
    B, T, H, D = ops.shape(x)[0], ops.shape(x)[1], ops.shape(x)[2], ops.shape(x)[3]
    
    # Generate frequencies
    inv_freq = 1.0 / (10000**(ops.cast(ops.arange(0, D, 2), "float32") / D))
    t = ops.cast(ops.arange(T), "float32")
    freqs = ops.outer(t, inv_freq)
    
    # Pre-calculated sin/cos
    sin_f = ops.sin(freqs)
    cos_f = ops.cos(freqs)
    
    # Reshape for broadcasting
    sin_f = ops.reshape(sin_f, (1, T, 1, D // 2))
    cos_f = ops.reshape(cos_f, (1, T, 1, D // 2))
    
    # Split queries/keys into two halves
    x1, x2 = ops.split(x, 2, axis=-1)
    
    # Effective rotation: [x1*cos - x2*sin, x1*sin + x2*cos]
    rx1 = x1 * cos_f - x2 * sin_f
    rx2 = x1 * sin_f + x2 * cos_f
    
    return ops.concatenate([rx1, rx2], axis=-1)

@keras.saving.register_keras_serializable(package="KAT")
class RMSNorm(layers.Layer):
    def __init__(self, eps=1e-6, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
    def build(self, input_shape):
        self.gamma = self.add_weight(shape=(input_shape[-1],), initializer="ones")
    def call(self, x):
        norm_x = ops.sqrt(ops.mean(ops.square(x), axis=-1, keepdims=True) + self.eps)
        return self.gamma * (x / norm_x)
    def get_config(self):
        config = super().get_config()
        config.update({"eps": self.eps})
        return config

@keras.saving.register_keras_serializable(package="KAT")
class MLAAttention(layers.Layer):
    """DeepSeek-style Multi-head Latent Attention with RoPE integration."""
    def __init__(self, d_model=128, n_heads=8, d_latent=64, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_latent = d_latent
        self.head_dim = d_model // n_heads

    def build(self, input_shape):
        self.kv_downproj = layers.Dense(self.d_latent)
        self.kv_upproj   = layers.Dense(self.d_model * 2)
        self.q_proj      = layers.Dense(self.d_model)
        self.out_proj    = layers.Dense(self.d_model)

    def call(self, x, training=False):
        B, T, D = ops.shape(x)[0], ops.shape(x)[1], ops.shape(x)[2]
        
        q = self.q_proj(x)
        kv_latent = self.kv_downproj(x)
        kv = self.kv_upproj(kv_latent)
        
        q = ops.reshape(q, (B, T, self.n_heads, self.head_dim))
        kv = ops.reshape(kv, (B, T, self.n_heads, self.head_dim * 2))
        k, v = ops.split(kv, 2, axis=-1)
        
        # Inject Relative Time Awareness via RoPE
        q = apply_rope(q, self.head_dim)
        k = apply_rope(k, self.head_dim)
        
        att = ops.matmul(q, ops.transpose(k, (0, 1, 3, 2))) / ops.sqrt(float(self.head_dim))
        att = ops.softmax(att, axis=-1)
        
        out = ops.matmul(att, v)
        out = ops.reshape(out, (B, T, self.d_model))
        return self.out_proj(out)

    def get_config(self):
        config = super().get_config()
        config.update({"d_model": self.d_model, "n_heads": self.n_heads, "d_latent": self.d_latent})
        return config

@keras.saving.register_keras_serializable(package="KAT")
class GatedMoE(layers.Layer):
    """Regime-Aware Multi-Expert Router (V4.2)"""
    def __init__(self, d_model=128, n_experts=None, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.n_experts = n_experts if n_experts else (32 if IS_GPU else 8)

    def build(self, input_shape):
        # We use the raw sequence plus a 'Regime Context' for routing
        self.gate = layers.Dense(self.n_experts, activation="softmax")
        self.expert_w = self.add_weight(
            shape=(self.n_experts, self.d_model, self.d_model),
            initializer="glorot_uniform", name="expert_weights"
        )

    def call(self, x, context=None):
        # If context is provided (Regime info), use it for routing, otherwise use x
        route_input = context if context is not None else x
        if len(ops.shape(route_input)) == 2: # Broadcast context to sequence
           route_input = ops.expand_dims(route_input, axis=1)
           route_input = ops.repeat(route_input, ops.shape(x)[1], axis=1)

        gate_scores = self.gate(route_input) 
        all_experts = ops.tensordot(x, self.expert_w, axes=[[-1], [1]]) 
        gate_scores = ops.expand_dims(gate_scores, axis=-1)
        return ops.sum(all_experts * gate_scores, axis=2)

    def get_config(self):
        config = super().get_config()
        config.update({"d_model": self.d_model, "n_experts": self.n_experts})
        return config

@keras.saving.register_keras_serializable(package="KAT")
class HydraBlock(layers.Layer):
    def __init__(self, d_model=128, n_heads=8, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.n_heads = n_heads
    def build(self, input_shape):
        self.norm1 = RMSNorm()
        self.attn  = MLAAttention(d_model=self.d_model, n_heads=self.n_heads)
        self.norm2 = RMSNorm()
        self.moe   = GatedMoE(d_model=self.d_model)
    def call(self, x, training=False, context=None):
        x = x + self.attn(self.norm1(x))
        x = x + self.moe(self.norm2(x), context=context)
        return x
    def get_config(self):
        config = super().get_config()
        config.update({"d_model": self.d_model, "n_heads": self.n_heads})
        return config

@keras.saving.register_keras_serializable(package="KAT")
class HydraV4(keras.Model):
    """The KRAKEN Mastery Engine (RoPE + MLA + Aux-MTP)"""
    def __init__(self, n_features=23, d_model=None, n_blocks=None, **kwargs):
        super().__init__(**kwargs)
        self.n_features = n_features
        self.d_model = d_model if d_model else (128 if IS_GPU else 96)
        self.n_blocks = n_blocks if n_blocks else (16 if IS_GPU else 8)
        self.mtp_steps = 15 
        self.aux_targets = 3 # Price, Volatility, Volume Imbalance

        self.feat_proj = layers.Dense(self.d_model)
        # Regime Extractor (Compresses features into a routing context)
        self.regime_ext = layers.Dense(self.d_model // 2, activation="tanh")
        
        self.layer_names = [f"b_{i}" for i in range(self.n_blocks)]
        for name in self.layer_names:
            setattr(self, name, HydraBlock(d_model=self.d_model))
        
        self.norm_final = RMSNorm()
        # Multi-Target Head: (Batch, MTP_Steps, 3)
        self.head = layers.Dense(self.mtp_steps * self.aux_targets) 

    def call(self, x, training=False):
        # Extract Regime Context from the last candle
        last_candle = x[:, -1, :]
        regime_ctx = self.regime_ext(last_candle)

        x = self.feat_proj(x)
        for name in self.layer_names:
            x = getattr(self, name)(x, training=training, context=regime_ctx)
        
        x = self.norm_final(x)
        last_step = x[:, -1, :] 
        
        out = self.head(last_step)
        # Reshape to (Batch, 15, 3) -> Predicts Price, Vol, and Volume Balance
        return ops.reshape(out, (-1, self.mtp_steps, self.aux_targets))

    def get_config(self):
        return {"n_features": self.n_features, "d_model": self.d_model, "n_blocks": self.n_blocks}

    @classmethod
    def from_config(cls, config):
        return cls(**config)

@keras.saving.register_keras_serializable(package="KAT")
class SovereignLoss(keras.losses.Loss):
    """Directional + Auxiliary Consistency Loss"""
    def __init__(self, direction_weight=3.0, aux_weight=0.5, **kwargs):
        super().__init__(**kwargs)
        self.direction_weight = direction_weight
        self.aux_weight = aux_weight

    def call(self, y_true, y_pred):
        # y_true: (B, 15, 3), y_pred: (B, 15, 3)
        # Index 0: Price, 1: Volatility, 2: Volume
        
        # 1. Price MSE + Directional Error
        p_true, p_pred = y_true[:, :, 0], y_pred[:, :, 0]
        mse_price = ops.mean(ops.square(p_true - p_pred))
        
        dir_true = ops.sign(p_true[:, 1:] - p_true[:, :-1])
        dir_pred = ops.sign(p_pred[:, 1:] - p_pred[:, :-1])
        dir_err  = ops.mean(ops.cast(ops.not_equal(dir_true, dir_pred), "float32"))
        
        # 2. Auxiliary Losses (Volatility & Volume Imbalance)
        v_true, v_pred = y_true[:, :, 1:], y_pred[:, :, 1:]
        mse_aux = ops.mean(ops.square(v_true - v_pred))
        
        return mse_price + (self.direction_weight * dir_err) + (self.aux_weight * mse_aux)

    def get_config(self):
        config = super().get_config()
        config.update({"direction_weight": self.direction_weight, "aux_weight": self.aux_weight})
        return config

def build_kraken(n_features=23):
    model = HydraV4(n_features=n_features)
    # Adaptive Learning Rate: Lower for CPUs to avoid gradient explosions
    lr = 1e-3 if IS_GPU else 5e-4
    model.compile(
        optimizer=keras.optimizers.AdamW(lr, weight_decay=0.01, clipnorm=1.0),
        loss=SovereignLoss(),
        metrics=["mae"]
    )
    return model
