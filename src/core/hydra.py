"""
HYDRA SOVEREIGN KRAKEN (V10.6) - PREDATOR EDITION ⚓🚀⚡
==========================================================
- Architecture: Sparse MoE + Lightning Attention + RoPE + SwiGLU + Tactical Reasoning
- Features:
    1. Lightning Attention: Linear scaling for massive context windows (up to 2,000 candles).
    2. GatedMoE: 256 specialized experts with entropy-balancing.
    3. SwiGLU Activation: Superior non-linear signal filtering (Gemma/Llama DNA).
    4. Tactical Reasoning: Native output head for market regime classification.
    5. Knowledge Distillation: Teacher-Student Distiller class for HFT deployment.
- Hardware: Auto-Detect (CPU / NVIDIA A40)
"""

import os, time
import numpy as np
import tensorflow as tf
import keras
from keras import layers, ops
from typing import Optional, Dict, Any

@keras.saving.register_keras_serializable(package="KAT")
class SwiGLU(layers.Layer):
    """V10.6: SwiGLU Activation (Gemma/Llama DNA). 
    Combines Swish + Gating for superior non-linear signal filtering.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.d_model = input_shape[-1]
        self.w1 = layers.Dense(self.d_model)
        self.w2 = layers.Dense(self.d_model)

    def call(self, x):
        # Cache w1(x) to avoid computing it twice (correct SwiGLU: SiLU(gate) * value)
        gate = self.w1(x)
        return gate * ops.sigmoid(gate) * self.w2(x)

    def get_config(self):
        return super().get_config()

@keras.saving.register_keras_serializable(package="KAT")
class Distiller(keras.Model):
    """V10.6: Knowledge Distiller (Teacher-Student Pipeline).
    Trains a student model to mimic the expert consensus of a heavyweight teacher.
    """
    def __init__(self, student, teacher):
        super().__init__()
        self.student = student
        self.teacher = teacher

    def compile(self, optimizer, metrics, prediction_loss_fn, distillation_loss_fn, alpha=0.1, temperature=3):
        super().compile(optimizer=optimizer, metrics=metrics)
        self.prediction_loss_fn = prediction_loss_fn
        self.distillation_loss_fn = distillation_loss_fn
        self.alpha = alpha
        self.temperature = temperature

    def compute_loss(self, x=None, y=None, y_pred=None, sample_weight=None, allow_no_y=False):
        # Teacher inference (No gradients)
        teacher_preds = self.teacher(x, training=False)
        student_preds = self.student(x, training=True)
        
        # 1. Standard task loss
        student_loss = self.prediction_loss_fn(y["prediction"], student_preds[0])
        
        # 2. Distillation loss (Soft labels)
        distill_loss = self.distillation_loss_fn(
            ops.softmax(teacher_preds[0] / self.temperature, axis=-1),
            ops.softmax(student_preds[0] / self.temperature, axis=-1)
        )

        return self.alpha * student_loss + (1 - self.alpha) * distill_loss

    def call(self, x):
        return self.student(x)

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
class TurboQuant(layers.Layer):
    """V10.5: Google TurboQuant (PolarQuant) Vector Compression. 
    Uses Random Rotation + Polar Mapping to compress context without accuracy loss.
    """
    def __init__(self, d_model=128, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        # Robust Fixed-Seed Orthogonal Matrix (QR Decomposition)
        rng = np.random.RandomState(42)
        H = rng.randn(d_model, d_model)
        Q, _ = np.linalg.qr(H)
        self.rot_init = Q.astype("float32")

    def build(self, input_shape):
        self.rotation = self.add_weight(
            name="jl_rotation", shape=(self.d_model, self.d_model),
            initializer=keras.initializers.Constant(self.rot_init),
            trainable=False
        )
        self.scale = self.add_weight(name="polar_scale", shape=(self.d_model,), initializer="ones")

    def call(self, x):
        # 1. Random Rotation (Spreads outliers for quantization stability)
        x = ops.matmul(x, self.rotation)
        
        # 2. Polar Mapping Approximation (Magnitude-Phase split)
        mag = ops.sqrt(ops.mean(ops.square(x), axis=-1, keepdims=True) + 1e-6)
        phase = x / mag
        
        # 3. Simulate 4-bit Quantization during training for robustness
        phase = ops.stop_gradient(ops.round(phase * 8.0) / 8.0 - phase) + phase
        
        return phase * mag * self.scale

    def get_config(self):
        config = super().get_config()
        config.update({"d_model": self.d_model})
        return config

def apply_rope(x, head_dim):
    """Rotary Positional Embeddings for relative temporal awareness."""
    B, T, H, D = ops.shape(x)[0], ops.shape(x)[1], ops.shape(x)[2], ops.shape(x)[3]
    inv_freq = 1.0 / (10000**(ops.cast(ops.arange(0, D, 2), "float32") / D))
    t = ops.cast(ops.arange(T), "float32")
    freqs = ops.outer(t, inv_freq)
    sin_f = ops.reshape(ops.sin(freqs), (1, T, 1, D // 2))
    cos_f = ops.reshape(ops.cos(freqs), (1, T, 1, D // 2))
    x1, x2 = ops.split(x, 2, axis=-1)
    return ops.concatenate([x1 * cos_f - x2 * sin_f, x1 * sin_f + x2 * cos_f], axis=-1)

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
class LightningAttention(layers.Layer):
    """V10.4: Linear-Scaling Fast Attention using the kernel trick for long sequences."""
    def __init__(self, d_model=128, n_heads=8, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

    def build(self, input_shape):
        self.q_proj = layers.Dense(self.d_model)
        self.k_proj = layers.Dense(self.d_model)
        self.v_proj = layers.Dense(self.d_model)
        self.out_proj = layers.Dense(self.d_model)

    def call(self, x, training=False):
        B = ops.shape(x)[0]
        T = ops.shape(x)[1]
        
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        q = ops.reshape(q, (B, T, self.n_heads, self.head_dim))
        k = ops.reshape(k, (B, T, self.n_heads, self.head_dim))
        v = ops.reshape(v, (B, T, self.n_heads, self.head_dim))
        
        # Apply RoPE to Q and K
        q = apply_rope(q, self.head_dim)
        k = apply_rope(k, self.head_dim)
        
        # Linear Attention via Kernel Trick: (Q @ (K.T @ V))
        # ELU + 1 activation for positive feature maps
        q = ops.elu(q) + 1.0
        k = ops.elu(k) + 1.0
        
        # Transpose for kernel computation: (B, H, T, D)
        q = ops.transpose(q, (0, 2, 1, 3))
        k = ops.transpose(k, (0, 2, 1, 3))
        v = ops.transpose(v, (0, 2, 1, 3))
        
        # Global context accumulation (B, H, D, D)
        context = ops.matmul(ops.transpose(k, (0, 1, 3, 2)), v)
        
        # Attention Output: (B, H, T, D)
        out = ops.matmul(q, context)
        
        # Normalization factor
        k_sum = ops.sum(k, axis=2, keepdims=True)
        z = 1.0 / (ops.matmul(q, ops.transpose(k_sum, (0, 1, 3, 2))) + 1e-6)
        out = out * z
        
        out = ops.transpose(out, (0, 2, 1, 3))
        out = ops.reshape(out, (B, T, self.d_model))
        return self.out_proj(out)

@keras.saving.register_keras_serializable(package="KAT")
class GatedMoE(layers.Layer):
    """V10.4: Predator 256-Expert Ensemble with Entropy Load Balancing."""
    def __init__(self, d_model=128, n_experts=256, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.n_experts = n_experts

    def build(self, input_shape):
        self.gate = layers.Dense(self.n_experts, activation="softmax")
        self.expert_w = self.add_weight(
            shape=(self.n_experts, self.d_model, self.d_model),
            initializer="glorot_uniform", name="expert_weights"
        )
        self.swiglu = SwiGLU()

    def call(self, x, context=None):
        route_input = context if context is not None else x
        if len(ops.shape(route_input)) == 2:
           route_input = ops.repeat(ops.expand_dims(route_input, axis=1), ops.shape(x)[1], axis=1)

        gate_scores = self.gate(route_input) 
        top_k_vals, top_k_idx = ops.top_k(gate_scores, k=2) 
        top_k_weights = top_k_vals / (ops.sum(top_k_vals, axis=-1, keepdims=True) + 1e-6)
        
        expert_outputs = ops.einsum("btd,edo->bteo", x, self.expert_w)
        
        mask = ops.one_hot(top_k_idx, self.n_experts) 
        mask = ops.sum(mask * ops.expand_dims(top_k_weights, axis=-1), axis=2) 
        mask_expanded = ops.expand_dims(mask, axis=-1) 
        
        weighted_avg = ops.sum(expert_outputs * mask_expanded, axis=2)
        weighted_avg = self.swiglu(weighted_avg) # Apply SwiGLU to expert consensus
        
        diff_sq = ops.square(expert_outputs - ops.expand_dims(weighted_avg, axis=2))
        weighted_var = ops.sum(diff_sq * mask_expanded, axis=2)
        consensus = ops.exp(-ops.mean(weighted_var, axis=-1))

        # Entropy Balancing: Loss to prevent expert collapse
        entropy = -ops.mean(ops.sum(gate_scores * ops.log(gate_scores + 1e-9), axis=-1))
        self.add_loss(-0.01 * entropy)

        return weighted_avg, consensus

@keras.saving.register_keras_serializable(package="KAT")
class HydraBlock(layers.Layer):
    def __init__(self, d_model=128, n_heads=8, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.n_heads = n_heads
    def build(self, input_shape):
        self.norm1 = RMSNorm()
        self.attn  = LightningAttention(d_model=self.d_model, n_heads=self.n_heads)
        self.tq    = TurboQuant(d_model=self.d_model) # NEW: TurboQuant Sentinel
        self.swiglu = SwiGLU() # NEW: SwiGLU Gating
        self.norm2 = RMSNorm()
        self.moe   = GatedMoE(d_model=self.d_model, n_experts=256)
    def call(self, x, training=False, context=None):
        attn_out = self.tq(self.attn(self.norm1(x)))
        x = x + self.swiglu(attn_out) # Applied SwiGLU Gating
        moe_out, consensus = self.moe(self.norm2(x), context=context)
        x = x + moe_out
        return x, consensus

@keras.saving.register_keras_serializable(package="KAT")
class SovereignLoss(keras.losses.Loss):
    def __init__(self, direction_weight=10.0, **kwargs):
        super().__init__(**kwargs)
        self.direction_weight = direction_weight
    def call(self, y_true, y_pred):
        p_true = y_true[:, :, 0]
        p_pred = y_pred[:, :, 0]
        mse = ops.mean(ops.square(p_true - p_pred))
        p_entry = p_true[:, 0:1]
        raw_true = p_true[:, 1:] - p_entry
        raw_pred = p_pred[:, 1:] - p_entry
        dir_loss = ops.mean(ops.square(ops.sign(raw_true) - ops.tanh(raw_pred * 10.0)))
        return mse + (self.direction_weight * dir_loss)

@keras.saving.register_keras_serializable(package="KAT")
class CertaintyMetric(keras.metrics.Metric):
    def __init__(self, name="certainty", **kwargs):
        super().__init__(name=name, **kwargs)
        self.cert_sum = self.add_weight(name="cert_sum", initializer="zeros")
        self.count = self.add_weight(name="count", initializer="zeros")
    def update_state(self, y_true, y_pred, sample_weight=None):
        self.cert_sum.assign_add(ops.sum(y_pred))
        self.count.assign_add(ops.cast(ops.shape(y_pred)[0], "float32"))
    def result(self):
        return self.cert_sum / (self.count + 1e-6)

@keras.saving.register_keras_serializable(package="KAT")
class SovereignAccuracy(keras.metrics.Metric):
    def __init__(self, name="dir_acc", **kwargs):
        super().__init__(name=name, **kwargs)
        self.total = self.add_weight(name="total", initializer="zeros")
        self.count = self.add_weight(name="count", initializer="zeros")
    def update_state(self, y_true, y_pred, sample_weight=None):
        p_true = y_true[:, :, 0]
        p_pred = y_pred[:, :, 0]
        p_entry = p_true[:, 0:1]
        correct = ops.equal(ops.sign(p_true[:, 1:] - p_entry), ops.sign(p_pred[:, 1:] - p_entry))
        self.total.assign_add(ops.sum(ops.cast(correct, "float32")))
        self.count.assign_add(ops.cast(ops.size(correct), "float32"))
    def result(self):
        return self.total / (self.count + 1e-9)

def build_kraken(n_features=27, context_window=120, forecast_steps=15):
    """V10.4: Predator Unified Initializer (Linear Attention + MoE-256)"""
    inputs = layers.Input(shape=(context_window, n_features))
    
    # Core Path
    x = layers.Dense(128)(inputs)
    x = RMSNorm()(x)
    
    all_consensus = []
    # 8 Predator Blocks (Deeper but faster due to Linear Attention)
    for _ in range(8):
        x, c = HydraBlock(d_model=128, n_heads=8)(x)
        all_consensus.append(c)
    
    avg_consensus = layers.Lambda(lambda cs: ops.mean(ops.stack(cs, axis=1), axis=1), name="certainty")(all_consensus)
    
    x = RMSNorm()(x)
    last_step = layers.GlobalAveragePooling1D()(x)
    
    # Output 1: Prediction (Price, Vol, Imbalance)
    preds_flat = layers.Dense((forecast_steps + 1) * 3)(last_step)
    preds = layers.Reshape((forecast_steps + 1, 3), name="prediction")(preds_flat)
    
    # Output 2: Tactical Reasoning (4 Regimes: Bull, Bear, Sideways, News)
    reasoning = layers.Dense(4, activation="softmax", name="reasoning")(last_step)
    
    model = keras.Model(inputs, [preds, avg_consensus, reasoning], name="predator_v10_4")
    
    model.compile(
        optimizer=keras.optimizers.AdamW(learning_rate=2e-4, weight_decay=0.01),
        loss={"prediction": SovereignLoss(), "certainty": None, "reasoning": "sparse_categorical_crossentropy"},
        metrics={"prediction": [SovereignAccuracy()], "certainty": CertaintyMetric()}
    )
    return model
