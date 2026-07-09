"""
HYDRA SOVEREIGN KRAKEN (V11.0) - DEEP PREDATOR PHASE 3 ⚓🚀⚡
=================================================================
Architecture: causal softmax attention (QK-Norm + RoPE, latent KV bottleneck)
+ MoE-256 (top-4 routed + shared expert) + DLS + expanded SwiGLU + Dropout.
V11.0 changes over V10.7:
  1. Real causal softmax attention — replaces the earlier ELU+1 linear-attention
     approximation. Softmax is what GPT/DeepSeek actually use; linear attention
     trades sharpness for O(T) cost, which isn't needed at this 120-step window.
  2. QK-Norm on Q/K before RoPE — current (2024-2025) attention stability practice.
  3. SwiGLU now expands to 2x d_model before gating/projecting down, instead of
     staying at d_model — real LLaMA/Gemma SwiGLU blocks expand ~2.7-4x; this
     was previously a capacity bottleneck relative to the architecture it's modeled on.
  4. Shared-expert path in GatedMoE (DeepSeekMoE-style) — always-active capacity
     alongside the top-4 routed experts, so routed experts don't relearn common
     patterns from scratch.
  5. Linear LR warmup before cosine decay — cheap insurance against early MoE
     routing instability (this model already uses Pre-Norm, which reduces but
     doesn't eliminate the need for this).
  Prior V10.7 upgrades retained: RoPE, Dropout(0.1), Volatility-Weighted Loss,
  Label Smoothing(0.1), Gradient Clipping, Input Noise Augmentation, top-4 routing.
"""

import os
import numpy as np
import tensorflow as tf
import keras
from keras import layers, ops
from typing import Optional

try:
    from config.sovereign_config import CONTEXT_WINDOW, FORECAST_STEPS
except ImportError:
    try:
        from src.config.sovereign_config import CONTEXT_WINDOW, FORECAST_STEPS
    except ImportError:
        CONTEXT_WINDOW = 120
        FORECAST_STEPS = 15

# ── Hardware ──────────────────────────────────────────────────────────────────
IS_GPU = len(tf.config.list_physical_devices('GPU')) > 0

def init_kraken_hardware():
    # Limit CPU threading to prevent CPU thrashing/server hangs
    try:
        tf.config.threading.set_intra_op_parallelism_threads(2)
        tf.config.threading.set_inter_op_parallelism_threads(2)
        print("⚡ KRAKEN: Thread limits applied (intra=2, inter=2) to prevent server hangs.")
    except Exception as e:
        print(f"⚠️ KRAKEN: Could not set thread limits: {e}")

    if IS_GPU:
        try:
            from tensorflow.keras import mixed_precision
            mixed_precision.set_global_policy('mixed_float16')
            print("🚀 KRAKEN: NVIDIA GPU DETECTED. Mixed Precision ENABLED.")
        except:
            pass
    else:
        print("🐌 KRAKEN: NO GPU DETECTED. Running in CPU-Lite Mode.")



@keras.saving.register_keras_serializable(package="KAT")
class WarmupCosineDecay(keras.optimizers.schedules.LearningRateSchedule):
    """
    Linear LR warmup for `warmup_steps`, then hands off to cosine decay.
    Standard practice in GPT/DeepSeek-style training recipes as cheap insurance
    against early-training instability (MoE routing in particular tends to be
    unstable in the first steps). This model already uses Pre-Norm (RMSNorm
    before attention/MoE), which reduces — but doesn't eliminate — the need
    for this, so it's kept short rather than a large fraction of training.
    """
    def __init__(self, initial_learning_rate, decay_steps, warmup_steps, alpha=0.1, **kwargs):
        super().__init__(**kwargs)
        self.initial_learning_rate = initial_learning_rate
        self.decay_steps = decay_steps
        self.warmup_steps = max(1, warmup_steps)
        self.alpha = alpha
        self.cosine = keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=initial_learning_rate,
            decay_steps=max(1, decay_steps - warmup_steps),
            alpha=alpha)

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        warmup_steps = tf.cast(self.warmup_steps, tf.float32)
        warmup_lr = self.initial_learning_rate * (step / warmup_steps)
        decayed_lr = self.cosine(step - warmup_steps)
        return tf.where(step < warmup_steps, warmup_lr, decayed_lr)

    def get_config(self):
        return {
            "initial_learning_rate": self.initial_learning_rate,
            "decay_steps": self.decay_steps,
            "warmup_steps": self.warmup_steps,
            "alpha": self.alpha,
        }


# ── Building Blocks ───────────────────────────────────────────────────────────

@keras.saving.register_keras_serializable(package="KAT")
class SwiGLU(layers.Layer):
    """
    SwiGLU Activation (Gemma/Llama DNA). Gated feed-forward.
    Expands to `expansion * d_model` before gating then projects back down —
    real LLaMA/Gemma SwiGLU blocks expand ~2.7-4x; without expansion the FFN
    sublayer has much less capacity than the architecture it's modeled on.
    """
    def __init__(self, expansion=2, **kwargs):
        super().__init__(**kwargs)
        self.expansion = expansion

    def build(self, input_shape):
        self.d_model = input_shape[-1]
        hidden = int(self.d_model * self.expansion)
        self.w1 = layers.Dense(hidden)
        self.w2 = layers.Dense(hidden)
        self.w3 = layers.Dense(self.d_model)

    def call(self, x):
        # SwiGLU: w3(w1(x) * sigmoid(w1(x)) * w2(x)) - gated, then projected back down
        w1_x = self.w1(x)
        gated = w1_x * ops.sigmoid(w1_x) * self.w2(x)
        return self.w3(gated)

    def get_config(self):
        config = super().get_config()
        config.update({"expansion": self.expansion})
        return config


@keras.saving.register_keras_serializable(package="KAT")
class RMSNorm(layers.Layer):
    """Root Mean Square Layer Normalization (faster than LayerNorm)."""
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
class TurboQuant(layers.Layer):
    """PolarQuant 2.0: Random orthogonal rotation + INT8 clip simulation."""
    def __init__(self, d_model=128, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        rng = np.random.RandomState(42)
        H = rng.randn(d_model, d_model)
        Q, _ = np.linalg.qr(H)
        self.rot_init = Q.astype("float32")

    def build(self, input_shape):
        self.rotation = self.add_weight(
            name="jl_rotation", shape=(self.d_model, self.d_model),
            initializer=keras.initializers.Constant(self.rot_init), trainable=False
        )
        self.scale = self.add_weight(
            name="polar_scale", shape=(self.d_model,), initializer="ones")

    def call(self, x):
        x = ops.matmul(x, self.rotation)
        mag = ops.sqrt(ops.mean(ops.square(x), axis=-1, keepdims=True) + 1e-6)
        phase = x / mag
        # INT8 simulation via straight-through estimator (STE)
        quant = ops.clip(phase * 127.0, -127.0, 127.0) / 127.0
        phase = ops.stop_gradient(quant - phase) + phase
        return phase * mag * self.scale

    def get_config(self):
        config = super().get_config()
        config.update({"d_model": self.d_model})
        return config


@keras.saving.register_keras_serializable(package="KAT")
class MLALayer(layers.Layer):
    """
    V11.0: Latent-bottleneck attention with RoPE and real causal softmax attention.
    - Compresses KV into a latent bottleneck (90% memory saving in the K/V projections).
    - RoPE applied to Q & K for temporal position awareness.
    - Causal SCALED DOT-PRODUCT (softmax) attention with an explicit mask — replaces
      the earlier ELU+1 linear-attention approximation. Linear attention trades away
      sharp, selective attention for O(T) cost, which matters for long sequences but
      not for a 120-step window (softmax here is a cheap O(T^2) = 14,400 ops/head).
      Softmax attention is also what GPT/DeepSeek actually use, unlike the linear
      approximation this replaces.
    - Attention-probability dropout added (previously only present after the MoE).
    """
    def __init__(self, d_model=128, n_heads=8, kv_lora_rank=32, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.kv_lora_rank = kv_lora_rank
        self.dropout_rate = dropout_rate

    def build(self, input_shape):
        self.q_proj      = layers.Dense(self.d_model)
        self.kv_down_proj = layers.Dense(self.kv_lora_rank)
        self.kv_up_proj  = layers.Dense(self.d_model * 2)
        self.out_proj    = layers.Dense(self.d_model)
        self.attn_dropout = layers.Dropout(self.dropout_rate)
        # QK-Norm: normalize Q/K per-head before RoPE — current (2024-2025) stability
        # practice for softmax attention, keeps attention logits well-scaled.
        self.q_norm = RMSNorm()
        self.k_norm = RMSNorm()

        # Precompute RoPE cos/sin as non-trainable weights to avoid dynamic tracing
        T = input_shape[1] if input_shape[1] is not None else CONTEXT_WINDOW
        half_d = self.head_dim // 2

        freq = 1.0 / (10000.0 ** (np.arange(half_d, dtype="float32") / half_d))
        t    = np.arange(T, dtype="float32")
        freqs = np.outer(t, freq)               # (T, D//2)
        cos_f_val = np.reshape(np.cos(freqs), (1, 1, T, half_d))
        sin_f_val = np.reshape(np.sin(freqs), (1, 1, T, half_d))

        self.cos_f = self.add_weight(
            name="rope_cos", shape=cos_f_val.shape,
            initializer=keras.initializers.Constant(cos_f_val),
            trainable=False
        )
        self.sin_f = self.add_weight(
            name="rope_sin", shape=sin_f_val.shape,
            initializer=keras.initializers.Constant(sin_f_val),
            trainable=False
        )

        # Causal mask (T, T): 1.0 where key position j is in the future of query i
        idx = np.arange(T)
        causal_val = (idx[None, :] > idx[:, None]).astype("float32")
        self.causal_bias = self.add_weight(
            name="causal_bias", shape=causal_val.shape,
            initializer=keras.initializers.Constant(causal_val),
            trainable=False
        )

    def _apply_rope(self, x):
        """Rotary Positional Embedding applied to (B, H, T, D)."""
        x1, x2 = ops.split(x, 2, axis=-1)
        return ops.concatenate(
            [x1 * self.cos_f - x2 * self.sin_f, x1 * self.sin_f + x2 * self.cos_f], axis=-1)

    def call(self, x, training=None):
        B, T = ops.shape(x)[0], ops.shape(x)[1]

        # Queries
        q = ops.reshape(self.q_proj(x), (B, T, self.n_heads, self.head_dim))

        # Latent KV bottleneck
        kv  = ops.reshape(
            self.kv_up_proj(self.kv_down_proj(x)),
            (B, T, self.n_heads, self.head_dim * 2))
        k, v = ops.split(kv, 2, axis=-1)

        # Transpose: (B, H, T, D)
        q = ops.transpose(q, (0, 2, 1, 3))
        k = ops.transpose(k, (0, 2, 1, 3))
        v = ops.transpose(v, (0, 2, 1, 3))

        # QK-Norm before RoPE for stable attention-score scale
        q = self.q_norm(q)
        k = self.k_norm(k)

        # Apply RoPE to Q and K for temporal awareness
        q = self._apply_rope(q)
        k = self._apply_rope(k)

        # Causal scaled dot-product softmax attention
        scale  = 1.0 / ops.sqrt(ops.cast(self.head_dim, "float32"))
        scores = ops.matmul(q, ops.transpose(k, (0, 1, 3, 2))) * scale  # (B, H, T, T)
        scores = scores - self.causal_bias * 1e9
        attn   = ops.softmax(scores, axis=-1)
        attn   = self.attn_dropout(attn, training=training)

        out = ops.matmul(attn, v)  # (B, H, T, D)

        out = ops.transpose(out, (0, 2, 1, 3))
        return self.out_proj(ops.reshape(out, (B, T, self.d_model)))

    def get_config(self):
        config = super().get_config()
        config.update({
            "d_model": self.d_model,
            "n_heads": self.n_heads,
            "kv_lora_rank": self.kv_lora_rank,
            "dropout_rate": self.dropout_rate
        })
        return config


@keras.saving.register_keras_serializable(package="KAT")
class GatedMoE(layers.Layer):
    """
    V11.0: DeepSeekMoE-style Mixture-of-Experts — routed experts (top-4 of 256,
    per-token specialized) plus a small always-active shared-expert path that
    captures common patterns every token needs, so the routed experts don't
    have to keep relearning them. Memory-efficient dynamic dispatch for the
    routed side (gathers only the active K=4 expert weight matrices per token).
    """
    def __init__(self, d_model=128, n_experts=256, **kwargs):
        super().__init__(**kwargs)
        self.d_model   = d_model
        self.n_experts = n_experts

    def build(self, input_shape):
        self.gate = layers.Dense(self.n_experts, activation="softmax")
        self.expert_w = self.add_weight(
            shape=(self.n_experts, self.d_model, self.d_model),
            initializer="glorot_uniform", name="expert_weights")
        self.swiglu = SwiGLU()
        # Shared expert: always active for every token, no routing.
        self.shared_dense  = layers.Dense(self.d_model)
        self.shared_swiglu = SwiGLU()

    def call(self, x, context=None):
        route_input = context if context is not None else x
        if len(ops.shape(route_input)) == 2:
            route_input = ops.repeat(
                ops.expand_dims(route_input, axis=1), ops.shape(x)[1], axis=1)

        gate_scores = self.gate(route_input)

        # Top-4 routing for richer gradient flow
        top_k_vals, top_k_idx = ops.top_k(gate_scores, k=4)
        top_k_weights = top_k_vals / (ops.sum(top_k_vals, axis=-1, keepdims=True) + 1e-6)

        # Optimize memory footprint by gathering only active expert weights
        # active_weights shape: (B, T, K, D, D)
        active_weights = ops.take(self.expert_w, top_k_idx, axis=0)

        # Compute outputs only for the active K=4 experts instead of all E=256: (B, T, K, D)
        expert_outputs = ops.einsum("btd,btkdo->btko", x, active_weights)

        # Weighted average of active expert outputs before SwiGLU
        weighted_inputs = ops.sum(expert_outputs * ops.expand_dims(top_k_weights, axis=-1), axis=2)
        routed_out = self.swiglu(weighted_inputs)

        # Shared expert path — always active, adds common-pattern capacity
        # that doesn't have to compete for routing gradient.
        shared_out = self.shared_swiglu(self.shared_dense(x))
        weighted_avg = routed_out + shared_out

        # Convergence Consensus signal (routed experts only — measures routing agreement)
        diff_sq       = ops.square(expert_outputs - ops.expand_dims(routed_out, axis=2))
        weighted_var  = ops.sum(diff_sq * ops.expand_dims(top_k_weights, axis=-1), axis=2)
        consensus     = ops.exp(-ops.mean(weighted_var, axis=-1))

        # Entropy load balancing (prevents expert collapse)
        # Scaled down to -1e-4 to prevent regularizer accumulation from dominating main loss
        entropy = -ops.mean(ops.sum(gate_scores * ops.log(gate_scores + 1e-9), axis=-1))
        self.add_loss(-1e-4 * entropy)

        return weighted_avg, consensus

    def get_config(self):
        config = super().get_config()
        config.update({"d_model": self.d_model, "n_experts": self.n_experts})
        return config


@keras.saving.register_keras_serializable(package="KAT")
class HydraBlock(layers.Layer):
    """
    V10.7 HydraBlock: MLA + TurboQuant + SwiGLU + MoE + Dropout.
    Dropout(0.1) added after MoE output to prevent expert memorization.
    """
    def __init__(self, d_model=128, n_heads=8, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.d_model      = d_model
        self.n_heads      = n_heads
        self.dropout_rate = dropout_rate

    def build(self, input_shape):
        self.norm1   = RMSNorm()
        self.attn    = MLALayer(d_model=self.d_model, n_heads=self.n_heads)
        self.tq      = TurboQuant(d_model=self.d_model)
        self.swiglu  = SwiGLU()
        self.norm2   = RMSNorm()
        self.moe     = GatedMoE(d_model=self.d_model, n_experts=256)
        self.dropout = layers.Dropout(self.dropout_rate)

    def call(self, x, training=None, context=None):
        # Attention path with TurboQuant stabilization
        attn_out = self.tq(self.attn(self.norm1(x), training=training))
        x = x + self.swiglu(attn_out)

        # MoE path with dropout regularization
        moe_out, consensus = self.moe(self.norm2(x), context=context)
        x = x + self.dropout(moe_out, training=training)

        return x, consensus

    def get_config(self):
        config = super().get_config()
        config.update({
            "d_model": self.d_model,
            "n_heads": self.n_heads,
            "dropout_rate": self.dropout_rate
        })
        return config


# ── Loss & Metrics ────────────────────────────────────────────────────────────

@keras.saving.register_keras_serializable(package="KAT")
class SovereignLoss(keras.losses.Loss):
    """
    V10.7: Volatility-Weighted Directional Loss.
    - MSE on price trajectory.
    - Direction loss (sign match) weighted by local volatility.
    - High-volatility moves incur a LARGER penalty when predicted wrong.
    """
    def __init__(self, direction_weight=10.0, **kwargs):
        super().__init__(**kwargs)
        self.direction_weight = direction_weight

    def call(self, y_true, y_pred):
        p_true = y_true[:, :, 0]
        p_pred = y_pred[:, :, 0]
        v_true = y_true[:, :, 1]  # Volatility channel

        # Base MSE
        mse = ops.mean(ops.square(p_true - p_pred))

        # Direction loss
        p_entry  = p_true[:, 0:1]
        raw_true = p_true[:, 1:] - p_entry
        raw_pred = p_pred[:, 1:] - p_entry
        dir_loss = ops.square(ops.sign(raw_true) - ops.tanh(raw_pred * 10.0))

        # Volatility weighting: errors during high-vol periods penalized more
        vol_weight = ops.abs(v_true[:, 1:]) + 1.0
        weighted_dir = ops.mean(dir_loss * vol_weight)

        return mse + (self.direction_weight * weighted_dir)

    def get_config(self):
        config = super().get_config()
        config.update({"direction_weight": self.direction_weight})
        return config


@keras.saving.register_keras_serializable(package="KAT")
class SovereignReasoningLoss(keras.losses.Loss):
    """Sovereign Reasoning Loss: Categorical Crossentropy with Label Smoothing on integer targets."""
    def __init__(self, label_smoothing=0.1, **kwargs):
        super().__init__(**kwargs)
        self.label_smoothing = label_smoothing

    def call(self, y_true, y_pred):
        y_true = ops.cast(y_true, "int32")
        if len(ops.shape(y_true)) > 1 and ops.shape(y_true)[-1] == 1:
            y_true = ops.squeeze(y_true, axis=-1)
        y_true_one_hot = ops.one_hot(y_true, num_classes=4)
        return keras.losses.categorical_crossentropy(
            y_true_one_hot, y_pred, label_smoothing=self.label_smoothing
        )

    def get_config(self):
        config = super().get_config()
        config.update({"label_smoothing": self.label_smoothing})
        return config


@keras.saving.register_keras_serializable(package="KAT")
def certainty_loss(y_true, y_pred):
    """
    Train certainty head to predict profitable (1.0) vs unprofitable (0.0) setups.
    y_pred: (B, T) per-timestep consensus from HydraBlocks
    y_true: (B, 1) binary target — 1.0 if LONG/SHORT, 0.0 if FEE_TRAP/NOISE
    """
    pred_mean = ops.mean(y_pred, axis=-1, keepdims=True)  # (B, 1)
    return keras.losses.binary_crossentropy(y_true, pred_mean)


@keras.saving.register_keras_serializable(package="KAT")
def dummy_certainty_loss(y_true, y_pred):
    """Kept for backward-compat loading of old checkpoints. Delegates to certainty_loss."""
    return certainty_loss(y_true, y_pred)


@keras.saving.register_keras_serializable(package="KAT")
class CertaintyMetric(keras.metrics.Metric):
    def __init__(self, name="certainty", **kwargs):
        super().__init__(name=name, **kwargs)
        self.cert_sum = self.add_weight(name="cert_sum", initializer="zeros")
        self.count    = self.add_weight(name="count",    initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        # FIX: use mean (not sum) so result stays in 0-1 range
        # y_pred shape: (batch, 120) — prev code summed all 120*batch values
        # and divided only by batch → 120× inflation
        self.cert_sum.assign_add(ops.mean(y_pred))
        self.count.assign_add(ops.cast(1, "float32"))  # count batches

    def result(self):
        return self.cert_sum / (self.count + 1e-6)

    def get_config(self):
        return super().get_config()


@keras.saving.register_keras_serializable(package="KAT")
class SovereignAccuracy(keras.metrics.Metric):
    """Directional accuracy: did we call the 15m move correctly?"""
    def __init__(self, name="dir_acc", **kwargs):
        super().__init__(name=name, **kwargs)
        self.total = self.add_weight(name="total", initializer="zeros")
        self.count = self.add_weight(name="count", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        p_true   = y_true[:, :, 0]
        p_pred   = y_pred[:, :, 0]
        p_entry  = p_true[:, 0:1]
        correct  = ops.equal(
            ops.sign(p_true[:, 1:] - p_entry),
            ops.sign(p_pred[:, 1:] - p_entry))
        self.total.assign_add(ops.sum(ops.cast(correct, "float32")))
        self.count.assign_add(ops.cast(ops.size(correct), "float32"))

    def result(self):
        return self.total / (self.count + 1e-9)

    def get_config(self):
        return super().get_config()


# ── Model Builder ─────────────────────────────────────────────────────────────

def build_kraken(n_features=38, context_window=CONTEXT_WINDOW, forecast_steps=FORECAST_STEPS,
                 dropout_rate=0.1, noise_stddev=0.02, vocab_size=128):
    """
    Build Phase 3 Deep-Predator V11.0.

    Improvements over V10.6:
      - Gaussian Input Noise layer (noise_stddev=0.02) for augmentation
      - 8x HydraBlock with Dropout(0.1)
      - MLALayer with RoPE positional encoding
      - SovereignLoss with volatility weighting
      - Label smoothing on reasoning head
      - Cosine LR Decay 5e-6 → 5e-7 with gradient clipping
      - Genuine GPT-style next-candle-token prediction: a discrete return-bucket
        vocabulary embedded alongside the continuous features, with a per-position
        classification head trained the way real language models train — every
        position predicts its own next token, not just the last one.
    """
    inputs = layers.Input(shape=(context_window, n_features), name="market_input")
    token_inputs = layers.Input(shape=(context_window,), dtype="int32", name="token_input")

    # Input Noise Augmentation (only active during training)
    x = layers.GaussianNoise(noise_stddev)(inputs)

    # Embed to d_model — continuous features + discrete return-token embedding,
    # summed like GPT sums token and positional embeddings.
    token_embed = layers.Embedding(vocab_size, 128, name="token_embedding")(token_inputs)
    x = RMSNorm()(layers.Dense(128)(x) + token_embed)

    # 8x HydraBlock with dropout
    all_consensus = []
    for i in range(8):
        x, c = HydraBlock(d_model=128, n_heads=8,
                          dropout_rate=dropout_rate, name=f"hydra_{i}")(x)
        all_consensus.append(c)

    # Consensus aggregation — stack all 8 block signals then calibrate with a
    # trainable Dense so certainty actually spans [0,1] instead of sitting at ~0.99
    cert_stacked = layers.Concatenate(axis=-1)([
        layers.Reshape((context_window, 1))(c) for c in all_consensus
    ])  # (B, T, 8)
    cert_calibrated = layers.Dense(
        1, activation='sigmoid', name='certainty_calibrate')(cert_stacked)  # (B, T, 1)
    avg_consensus = layers.Reshape(
        (context_window,), name='certainty')(cert_calibrated)  # (B, T)

    # Output heads - preserve dynamic temporal sequence ordering using last-token extraction
    normed_x  = RMSNorm()(x)
    last_step = layers.Lambda(lambda t: t[:, -1, :])(normed_x)

    preds = layers.Reshape(
        (forecast_steps + 1, 3), name="prediction")(
        layers.Dense((forecast_steps + 1) * 3)(last_step))

    # Label smoothing(0.1) on reasoning to prevent overconfidence
    reasoning = layers.Dense(4, activation="softmax", name="reasoning")(last_step)

    # Genuine next-token head: applied at EVERY position (not just the last),
    # so every one of the 120 context positions is its own training example —
    # this is the actual causal-LM training signal GPT uses, predicting token
    # t+1 from positions 0..t via the causal attention mask already in place.
    next_token = layers.Dense(vocab_size, activation="softmax", name="next_token")(normed_x)

    model = keras.Model(
        [inputs, token_inputs], [preds, avg_consensus, reasoning, next_token],
        name="sovereign_kraken_v11_0")

    # Linear warmup (500 steps) into Cosine Decay LR: 5e-6 → 5e-7 over 10,000 steps
    lr_schedule = WarmupCosineDecay(
        initial_learning_rate=5e-6,
        decay_steps=10000,
        warmup_steps=500,
        alpha=0.1   # floor = 5e-7
    )

    model.compile(
        optimizer=keras.optimizers.AdamW(
            learning_rate=lr_schedule,
            weight_decay=0.01,
            clipnorm=0.5       # Tightened clipping for MoE stability
        ),
        loss={
            "prediction": SovereignLoss(direction_weight=3.0),
            "certainty":  certainty_loss,
            "reasoning":  SovereignReasoningLoss(label_smoothing=0.1),
            "next_token": keras.losses.SparseCategoricalCrossentropy(),
        },
        loss_weights={"prediction": 3.0, "certainty": 1.0, "reasoning": 1.0, "next_token": 2.0},
        metrics={
            "prediction": [SovereignAccuracy()],
            "certainty":  [CertaintyMetric()],
            "next_token": [keras.metrics.SparseCategoricalAccuracy(name="token_acc")],
        }
    )
    return model


def generate_future_tokens(model, x_context, tok_context, n_steps, bin_centers,
                            local_mean, local_std, t_close_idx):
    """
    Genuine autoregressive generation: predict one next-candle token, convert it
    back into an approximate price, append it to the window, and repeat — the
    actual GPT generation loop (predict -> append -> predict again), as opposed
    to the 'prediction' head's one-shot forecast of all future steps at once.

    x_context   : (context_window, n_features) most recent DLS-scaled feature window
    tok_context : (context_window,) most recent real return-token sequence
    bin_centers : (vocab_size,) representative raw return per token, from fit_return_vocab
    local_mean/local_std : the DLS stats used to scale x_context (from apply_dls),
                            reused to convert generated raw returns back into the
                            same Z-scored space the model expects as input.
    Returns: list of dicts with token id, predicted raw return, and predicted raw close.
    """
    x   = x_context.copy()
    tok = tok_context.copy()
    results = []

    raw_close = float(x[-1, t_close_idx] * local_std[t_close_idx] + local_mean[t_close_idx])

    for _ in range(n_steps):
        X_in   = x[np.newaxis].astype("float32")
        Tok_in = tok[np.newaxis].astype("int32")
        outputs = model([X_in, Tok_in], training=False)
        next_token_probs = outputs[3].numpy()[0, -1]   # last position's next-token distribution
        token_id = int(np.argmax(next_token_probs))
        predicted_return = float(bin_centers[token_id])

        raw_close = raw_close * (1.0 + predicted_return)
        results.append({"token": token_id, "return": predicted_return, "close": raw_close})

        # Reconstruct the synthetic next candle in the same DLS-scaled space,
        # carrying forward the other (non-price) features from the last real candle.
        new_row = x[-1].copy()
        new_row[t_close_idx] = (raw_close - local_mean[t_close_idx]) / (local_std[t_close_idx] + 1e-9)

        x   = np.concatenate([x[1:], new_row[np.newaxis]], axis=0)
        tok = np.concatenate([tok[1:], [token_id]], axis=0)

    return results
