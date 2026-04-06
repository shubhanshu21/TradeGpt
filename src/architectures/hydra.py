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

# ── HARDWARE AUTODETECT ───────────────────────────────────────────────────────
GPUS = tf.config.list_physical_devices('GPU')
if GPUS:
    try:
        # A40 Optimization: Enable Mixed Precision for massive speedup
        from tensorflow.keras import mixed_precision
        mixed_precision.set_global_policy('mixed_float16')
        print(f"🚀 KRAKEN: NVIDIA GPU DETECTED. Mixed Precision ENABLED (A40 Mode).")
        IS_GPU = True
    except:
        IS_GPU = False
else:
    print(f"🐌 KRAKEN: NO GPU DETECTED. Running in CPU-Lite Mode.")
    IS_GPU = False

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
        self.kv_upproj   = layers.Dense(self.d_model)
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

    def call(self, x):
        # x is (B, T, D)
        gate_scores = self.gate(x) # (B, T, n_experts)
        
        # Batch Matrix Multiply across the expert dimension
        # A40 Mode: This is where the 48GB VRAM unleashes the Specialists
        outputs = []
        for i in range(self.n_experts):
            outputs.append(ops.matmul(x, self.expert_w[i]))
        
        all_experts = ops.stack(outputs, axis=2) # (B, T, n_experts, D)
        
        gate_scores = ops.expand_dims(gate_scores, axis=-1) # (B, T, n_experts, 1)
        combined = ops.sum(all_experts * gate_scores, axis=2)
        return combined

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
    def __init__(self, n_features=23, d_model=128 if IS_GPU else 96, n_blocks=12 if IS_GPU else 6, **kwargs):
        super().__init__(**kwargs)
        self.n_features = n_features
        self.d_model = d_model
        self.n_blocks = n_blocks
        self.mtp_steps = 15 # Predict 15 minutes of Alpha

    def build(self, input_shape):
        self.feat_proj = layers.Dense(self.d_model)
        self.blocks = [HydraBlock(d_model=self.d_model) for _ in range(self.n_blocks)]
        self.head = layers.Dense(self.mtp_steps) # Output 15 future steps
        self.norm_final = RMSNorm()

    def call(self, x, training=False):
        x = self.feat_proj(x)
        for block in self.blocks:
            x = block(x, training=training)
        
        # Aggregate Context (Mean Pool) + Final Residual Link
        x = self.norm_final(x)
        # We take the VERY LAST step to predict the future curve
        last_step = x[:, -1, :] # (B, D)
        return self.head(last_step) # (B, 15)

    def get_config(self):
        return {"n_features": self.n_features, "d_model": self.d_model, "n_blocks": self.n_blocks}

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
