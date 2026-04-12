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
    def __init__(self, d_model=128, n_heads=8, d_latent=32, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_latent = d_latent # TurboQuant: Reduced Latent Dim (LVQ)
        self.head_dim = d_model // n_heads

    def build(self, input_shape):
        T = input_shape[1]
        self.kv_downproj = layers.Dense(self.d_latent)
        self.kv_upproj   = layers.Dense(self.d_model * 2)
        self.q_proj      = layers.Dense(self.d_model)
        self.out_proj    = layers.Dense(self.d_model)
        
        # Pre-compute LogSparse Mask (Constant) - Use NumPy to avoid FuncGraph scope errors
        idx = np.arange(T)
        dist = np.abs(np.expand_dims(idx, -1) - np.expand_dims(idx, 0))
        mask_np = -0.1 * np.log(dist + 1.0)
        mask_np = mask_np.reshape(1, 1, T, T).astype("float32")

        self.log_sparse_mask = self.add_weight(
            name="log_sparse_mask", shape=(1, 1, T, T),
            initializer=keras.initializers.Constant(mask_np),
            trainable=False
        )

    def call(self, x, training=False):
        B = ops.shape(x)[0]
        T = ops.shape(x)[1]
        
        q = self.q_proj(x)
        kv_latent = self.kv_downproj(x)
        kv = self.kv_upproj(kv_latent)
        
        # Reshape to (B, T, n_heads, head_dim)
        q = ops.reshape(q, (B, T, self.n_heads, self.head_dim))
        kv = ops.reshape(kv, (B, T, self.n_heads, self.head_dim * 2))
        k, v = ops.split(kv, 2, axis=-1)
        
        # Inject Relative Time Awareness via RoPE
        q = apply_rope(q, self.head_dim)
        k = apply_rope(k, self.head_dim)
        
        # Transpose for temporal attention: (B, n_heads, T, head_dim)
        q = ops.transpose(q, (0, 2, 1, 3))
        k = ops.transpose(k, (0, 2, 1, 3))
        v = ops.transpose(v, (0, 2, 1, 3))
        
        # Compute proper temporal attention: (B, n_heads, T, T)
        att = ops.matmul(q, ops.transpose(k, (0, 1, 3, 2))) / ops.sqrt(float(self.head_dim))
        
        # Add-on 2: LogSparse + Time-Decay (V6.0 SOTA)
        indices = ops.arange(T, dtype="float32")
        dist = ops.expand_dims(indices, axis=1) - ops.expand_dims(indices, axis=0)
        decay_mask = ops.exp(-ops.maximum(dist, 0.0) * 0.01) # Geometric Priority
        att = att + self.log_sparse_mask[:, :, :T, :T]
        att = att + ops.log(ops.expand_dims(ops.expand_dims(decay_mask, axis=0), axis=0) + 1e-9)
        
        att = ops.softmax(att, axis=-1)
        out = ops.matmul(att, v) # (B, n_heads, T, head_dim)
        
        # Add-on 1: Infini-Attention (Temporal Compressive Memory)
        memory_k = ops.mean(k, axis=2, keepdims=True) # (B, n_heads, 1, head_dim)
        memory_v = ops.mean(v, axis=2, keepdims=True) # (B, n_heads, 1, head_dim)
        
        # Gate the memory anchor with Sigmoid
        memory_scores = ops.matmul(q, ops.transpose(memory_k, (0, 1, 3, 2)))
        memory_att = ops.sigmoid(memory_scores) # (B, n_heads, T, 1)
        
        infini_out = ops.matmul(memory_att, memory_v) # (B, n_heads, T, head_dim)
        
        # Fuse Local (LogSparse) + Global (Infini-Memory)
        out = out + (0.2 * infini_out)
        
        # Transpose back to (B, T, n_heads, head_dim)
        out = ops.transpose(out, (0, 2, 1, 3))
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
        self.n_experts = n_experts if n_experts else 16 # Grand Mastery: Higher Specialization

    def build(self, input_shape):
        # We use the raw sequence plus a 'Regime Context' for routing
        self.gate = layers.Dense(self.n_experts, activation="softmax")
        self.expert_w = self.add_weight(
            shape=(self.n_experts, self.d_model, self.d_model),
            initializer="glorot_uniform", name="expert_weights"
        )

    def call(self, x, context=None):
        route_input = context if context is not None else x
        if len(ops.shape(route_input)) == 2:
           route_input = ops.repeat(ops.expand_dims(route_input, axis=1), ops.shape(x)[1], axis=1)

        gate_scores = self.gate(route_input) 
        top_k_vals, top_k_idx = ops.top_k(gate_scores, k=2) 
        top_k_weights = top_k_vals / (ops.sum(top_k_vals, axis=-1, keepdims=True) + 1e-6)
        
        # Expert Outputs (B, T, E, O)
        expert_outputs = ops.einsum("btd,edo->bteo", x, self.expert_w)
        
        # Select Top-K Weighted Average
        mask = ops.one_hot(top_k_idx, self.n_experts) 
        mask = ops.sum(mask * ops.expand_dims(top_k_weights, axis=-1), axis=2) 
        mask_expanded = ops.expand_dims(mask, axis=-1) 
        
        weighted_avg = ops.sum(expert_outputs * mask_expanded, axis=2)

        # ── Expert Consensus (New V6.0 feature) ──────────────────────────────
        # Measures how much the experts disagree. High variance = Low consensus.
        # Square of differences weighted by gate mask.
        diff_sq = ops.square(expert_outputs - ops.expand_dims(weighted_avg, axis=2))
        weighted_var = ops.sum(diff_sq * mask_expanded, axis=2)
        consensus = ops.exp(-ops.mean(weighted_var, axis=-1)) # 1.0 = perfect agreement

        return weighted_avg, consensus

    def get_config(self):
        config = super().get_config()
        config.update({"d_model": self.d_model, "n_experts": self.n_experts})
        return config

@keras.saving.register_keras_serializable(package="KAT")
class TTMReflex(layers.Layer):
    """Add-on 3: Tiny Time Mixer (IBM) - Ultra Fast MLP Reflex"""
    def __init__(self, d_model=128, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
    def build(self, input_shape):
        self.mixer = layers.Dense(self.d_model, activation="gelu")
        self.norm = RMSNorm()
    def call(self, x):
        # A lightning-fast dense track bypassing all attention layers
        return self.norm(x + self.mixer(x))
    def get_config(self):
        config = super().get_config()
        config.update({"d_model": self.d_model})
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
        moe_out, consensus = self.moe(self.norm2(x), context=context)
        x = x + moe_out
        return x, consensus
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
        # V5.0 CPU-Optimal: 128-wide, 12-block depth (hierarchical BTC reasoning)
        # Width→128 gives 9× speedup vs 384. Depth→12 preserves full learning capacity.
        self.d_model  = d_model  if d_model  else 128
        self.n_blocks = n_blocks if n_blocks else 12
        self.mtp_steps = 15 
        self.aux_targets = 3 # Price, Volatility, Volume Imbalance

        self.feat_proj = layers.Dense(self.d_model)
        
        # TTM Reflex Circuit
        self.ttm_reflex = TTMReflex(d_model=self.d_model)
        
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
        
        # Compute the Instant Reflex (TTM)
        reflex_track = self.ttm_reflex(x)
        
        # Compute the Deep Path with Consensus Tracking
        all_consensus = []
        for name in self.layer_names:
            x, c = getattr(self, name)(x, training=training, context=regime_ctx)
            all_consensus.append(c)
        
        # V6.0 SOTA: Internal Expert Consensus Penalty
        # We penalize high variance (low consensus) directly inside the model
        consensus_score = ops.mean(ops.stack(all_consensus))
        # Penalty: loss increases as consensus decreases
        self.add_loss(0.1 * (1.0 - consensus_score)) 

        # Fuse Deep Intelligence with Fast Reflexes
        x = x + reflex_track
        
        x = self.norm_final(x)
        last_step = x[:, -1, :] 
        
        preds = self.head(last_step)
        preds = ops.reshape(preds, (-1, self.mtp_steps, self.aux_targets))
        
        return preds

    def get_config(self):
        return {"n_features": self.n_features, "d_model": self.d_model, "n_blocks": self.n_blocks}

    @classmethod
    def from_config(cls, config):
        return cls(**config)

@keras.saving.register_keras_serializable(package="KAT")
class SovereignLoss(keras.losses.Loss):
    """
    Sovereign Loss V2 — Label-Smoothed Directional + Auxiliary Consistency

    Enhancements over V1:
    - Label smoothing (smooth=0.1): soft directional targets prevent overconfidence
      on noisy BTC micro-structure (e.g. ±1 → ±0.9)
    - Confidence-weighted direction: penalises wrong predictions harder when
      the model is highly confident (steep predicted gradient)
    - Huber price loss: less sensitive to outlier wick candles than pure MSE
    """
    def __init__(self, direction_weight=5.0, aux_weight=0.5, label_smooth=0.1,
                 huber_delta=1.0, **kwargs):
        super().__init__(**kwargs)
        self.direction_weight = direction_weight
        self.aux_weight       = aux_weight
        self.label_smooth     = label_smooth   # NEW: soft targets
        self.huber_delta      = huber_delta    # NEW: robust price loss

    def call(self, y_true, y_pred):
        p_true = y_true[:, :, 0]   # (B, 15)
        p_pred = y_pred[:, :, 0]

        # 1. Huber price loss (robust to candle wicks)
        err    = p_true - p_pred
        abs_e  = ops.abs(err)
        huber  = ops.where(abs_e <= self.huber_delta,
                           0.5 * ops.square(err),
                           self.huber_delta * (abs_e - 0.5 * self.huber_delta))
        loss_price = ops.mean(huber)

        # 2. Label-smoothed directional loss
        p_entry  = p_true[:, 0:1]
        raw_true = p_true[:, 1:] - p_entry
        raw_pred = p_pred[:, 1:] - p_entry

        # Soft target: sign direction with smoothing  (±1 → ±(1-smooth))
        smooth    = self.label_smooth
        dir_true  = ops.sign(raw_true) * (1.0 - smooth)   # in [-0.9, 0.9]

        # Predicted direction score (tanh squashes to [-1,1])
        dir_pred_score = ops.tanh(raw_pred * 50.0)

        # Confidence-weighted MSE between soft labels and predictions
        dir_loss = ops.mean(ops.square(dir_true - dir_pred_score))

        # 3. Auxiliary losses: volatility + volume (Huber)
        a_true = y_true[:, :, 1:]
        a_pred = y_pred[:, :, 1:]
        ae     = a_true - a_pred
        abs_ae = ops.abs(ae)
        huber_a = ops.where(abs_ae <= self.huber_delta,
                            0.5 * ops.square(ae),
                            self.huber_delta * (abs_ae - 0.5 * self.huber_delta))
        loss_aux = ops.mean(huber_a)

        return loss_price + (self.direction_weight * dir_loss) + (self.aux_weight * loss_aux)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({
            "direction_weight": self.direction_weight,
            "aux_weight":       self.aux_weight,
            "label_smooth":     self.label_smooth,
            "huber_delta":      self.huber_delta,
        })
        return cfg

@keras.saving.register_keras_serializable(package="KAT")
class SovereignAccuracy(keras.metrics.Metric):
    """
    Calculates Directional Win-Rate for Sovereign Kraken.
    Counts a 'win' if sign(pred) == sign(true) for price changes.
    """
    def __init__(self, name="dir_acc", **kwargs):
        super().__init__(name=name, **kwargs)
        self.total = self.add_weight(name="total", initializer="zeros")
        self.count = self.add_weight(name="count", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        p_true = y_true[:, :, 0]
        p_pred = y_pred[:, :, 0]
        
        # Look at returns from entry (aligns with new loss)
        p_entry  = p_true[:, 0:1]
        raw_true = p_true[:, 1:] - p_entry
        raw_pred = p_pred[:, 1:] - p_entry
        
        # Compare signs
        correct = ops.equal(ops.sign(raw_true), ops.sign(raw_pred))
        correct = ops.cast(correct, "float32")
        
        self.total.assign_add(ops.sum(correct))
        self.count.assign_add(ops.cast(ops.size(correct), "float32"))

    def result(self):
        return self.total / self.count

    def reset_state(self):
        self.total.assign(0.0)
        self.count.assign(0.0)


def build_kraken(n_features=23, total_steps=None, initial_step=0):
    """
    Build & compile the Sovereign Kraken model.
    total_steps: if provided, uses CosineDecay LR schedule.
    initial_step: used to resume the schedule at the correct point.
    """
    custom_objs = {"TTMReflex": TTMReflex}
    with keras.saving.custom_object_scope(custom_objs):
        model = HydraV4(n_features=n_features)

    # ── Cosine Annealing LR ───────────────────────────────────────────────────
    initial_lr  = 1e-3 if IS_GPU else 5e-4
    min_lr      = 1e-6

    if total_steps:
        lr_schedule = keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=initial_lr,
            decay_steps=total_steps,
            alpha=min_lr / initial_lr,
        )
        print(f"   📉 Cosine Annealing: Resuming at step {initial_step:,} / {total_steps:,}")
    else:
        lr_schedule = initial_lr

    # ── Optimizer ─────────────────────────────────────────────────────────────
    base_optimizer = keras.optimizers.AdamW(
        lr_schedule, weight_decay=0.01, clipnorm=1.0
    )
    
    try:
        optimizer = keras.optimizers.GradientAccumulationOptimizer(
            inner_optimizer=base_optimizer, accumulation_steps=4
        )
    except AttributeError:
        optimizer = base_optimizer

    # Set iterations to resume LR schedule
    if total_steps and initial_step > 0:
        optimizer.iterations.assign(initial_step)

    model.compile(
        optimizer=optimizer,
        loss=SovereignLoss(direction_weight=10.0, label_smooth=0.1),
        metrics=["mae", SovereignAccuracy()]
    )
    return model
