"""
HYDRA SOVEREIGN KRAKEN (V4.0) - MULTI-HARDWARE EDITION ⚓🚀⚡
==========================================================
- Architecture: MoE + MLA + MTP (Adaptive)
- Hardware: Auto-Detect (CPU / NVIDIA A40)
- Intelligence: 15-Minute Future Prophecy (MTP-15)
- Memory: Elastic Context (150 - 1440 minutes)
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
    """DeepSeek-style Multi-head Latent Attention (Adaptive for A40)"""
    def __init__(self, d_model=128, n_heads=8, d_latent=64, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_latent = d_latent
        self.head_dim = d_model // n_heads

    def build(self, input_shape):
        # Latent Compression logic
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
        
        # Fast attention (A40 will use FlashAttention if available)
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
    """Adaptive Gated Mixture of Experts (Auto-sizes to Hardware)"""
    def __init__(self, d_model=128, n_experts=32 if IS_GPU else 8, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.n_experts = n_experts

    def build(self, input_shape):
        self.gate = layers.Dense(self.n_experts, activation="softmax")
        # Optimization: Matrix Sharding of Experts for Memory Speed
        self.expert_w = self.add_weight(
            shape=(self.n_experts, self.d_model, self.d_model),
            initializer="glorot_uniform", name="expert_weights"
        )

    def __init__(self, d_model=128, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        # CPU Mode gets 8 specialists | GPU gets 32 specialists (Vectorized)
        self.n_experts = 32 if IS_GPU else 8

    def build(self, input_shape):
        self.gate = layers.Dense(self.n_experts, activation="softmax")
        self.expert_w = self.add_weight(
            shape=(self.n_experts, self.d_model, self.d_model),
            initializer="glorot_uniform", name="expert_weights"
        )

    def call(self, x):
        # gate_scores: (B, T, n_experts)
        gate_scores = self.gate(x) 
        # Vectorized MoE: Single Matrix Contract (No Loops)
        # x: (B, T, D), weight: (E, D, D) -> results: (B, T, E, D)
        all_experts = ops.tensordot(x, self.expert_w, axes=[[-1], [1]]) 
        # result: (B, T, E, D) -> reduce with gate: (B, T, D)
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
    def call(self, x, training=False):
        x = x + self.attn(self.norm1(x))
        x = x + self.moe(self.norm2(x))
        return x
    def get_config(self):
        config = super().get_config()
        config.update({"d_model": self.d_model, "n_heads": self.n_heads})
        return config

@keras.saving.register_keras_serializable(package="KAT")
class HydraV4(keras.Model):
    """The KRAKEN Multi-Hardware Engine (MoE 32 + MLA 8 + MTP-15)"""
    def __init__(self, n_features=23, d_model=None, n_blocks=None, **kwargs):
        super().__init__(**kwargs)
        self.n_features = n_features
        # Allow dynamic override: 128/16 for GPU Power, 96/8 for CPU Stability
        self.d_model = d_model if d_model else (128 if IS_GPU else 96)
        self.n_blocks = n_blocks if n_blocks else (16 if IS_GPU else 8)
        self.mtp_steps = 15 

        # Dynamic Unbundling for Full Summary Visibility & GPU Scaling
        self.feat_proj = layers.Dense(self.d_model)
        self.layer_names = [f"b_{i}" for i in range(self.n_blocks)]
        for name in self.layer_names:
            setattr(self, name, HydraBlock(d_model=self.d_model))
        
        self.head = layers.Dense(self.mtp_steps) 
        self.norm_final = RMSNorm()

    def call(self, x, training=False):
        x = self.feat_proj(x)
        # Sequential Execution of Dynamically Unbundled Sovereigns
        for name in self.layer_names:
            x = getattr(self, name)(x, training=training)
        
        x = self.norm_final(x)
        last_step = x[:, -1, :] 
        return self.head(last_step)

    def get_config(self):
        return {"n_features": self.n_features, "d_model": self.d_model, "n_blocks": self.n_blocks}

    @classmethod
    def from_config(cls, config):
        return cls(**config)

@keras.saving.register_keras_serializable(package="KAT")
class SovereignLoss(keras.losses.Loss):
    """The Kraken Loss: Rewards Directional Prophecy over Mean Error"""
    def __init__(self, direction_weight=3.0, **kwargs):
        super().__init__(**kwargs)
        self.direction_weight = direction_weight
    def call(self, y_true, y_pred):
        mse = ops.mean(ops.square(y_true - y_pred))
        # Penalty for guessing the wrong Delta-Direction
        dir_true = ops.sign(y_true[:, 1:] - y_true[:, :-1])
        dir_pred = ops.sign(y_pred[:, 1:] - y_pred[:, :-1])
        dir_err = ops.mean(ops.cast(ops.not_equal(dir_true, dir_pred), "float32"))
        return mse + (self.direction_weight * dir_err)
    def get_config(self):
        config = super().get_config()
        config.update({"direction_weight": self.direction_weight})
        return config

def build_kraken(n_features=23):
    model = HydraV4(n_features=n_features)
    # Use higher learning rate on GPU to burn through the $7.00 faster
    lr = 1e-3 if IS_GPU else 5e-4
    model.compile(
        optimizer=keras.optimizers.Adam(lr, clipnorm=1.0),
        loss=SovereignLoss(),
        metrics=["mae"]
    )
    return model
