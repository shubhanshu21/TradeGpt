"""
HYDRA SOVEREIGN KRAKEN (V12.5) - DEEP PREDATOR PHASE 3 ⚓🚀⚡
=================================================================
Architecture: causal softmax attention (QK-Norm + RoPE, latent KV bottleneck)
+ MoE-32 (top-4 routed + shared expert) + DLS + expanded SwiGLU + Dropout.
Equity swing edition: 60-day context window, single input, 4 outputs
(prediction/certainty/reasoning/next_candle - see build_kraken's docstring
for the single-step-not-generation design of the next_candle head), one
model trained per symbol.
V12.5 changes over V10.7 (numbering kept for history; all still apply here):
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
        CONTEXT_WINDOW = 60
        FORECAST_STEPS = 20

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
    V12.5: Latent-bottleneck attention with RoPE and real causal softmax attention.
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
    def __init__(self, d_model=128, n_heads=8, kv_lora_rank=32, dropout_rate=0.15, **kwargs):
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
    V12.5: DeepSeekMoE-style Mixture-of-Experts — routed experts (top-4 of 32,
    per-token specialized) plus a small always-active shared-expert path that
    captures common patterns every token needs, so the routed experts don't
    have to keep relearning them. Memory-efficient dynamic dispatch for the
    routed side (gathers only the active K=4 expert weight matrices per token).
    n_experts lowered from 256 — that count makes sense for models trained on
    billions of tokens (DeepSeek-scale), but with ~25K training windows most
    of 256 experts were only ever seeing a trickle of gradient signal each.
    Fewer experts means each one actually gets enough real examples to
    specialize on, instead of spreading the same data thin across 256 slots.
    """
    def __init__(self, d_model=128, n_experts=32, expert_dropout_rate=0.4, **kwargs):
        super().__init__(**kwargs)
        self.d_model   = d_model
        self.n_experts = n_experts
        # Sparse/routed capacity is more overfitting-prone than dense layers
        # (each expert only ever sees a fraction of tokens, so it gets less
        # real gradient signal to generalize from) - well-documented practice
        # for MoE architectures is a HIGHER dropout on the sparse/expert path
        # than on the surrounding dense layers, not the same uniform rate
        # everywhere. Applied only to the routed-expert output below, not the
        # always-active shared-expert path, which behaves like a normal dense
        # layer and is already covered by HydraBlock's own dropout.
        self.expert_dropout_rate = expert_dropout_rate

    def build(self, input_shape):
        self.gate = layers.Dense(self.n_experts, activation="softmax")
        self.expert_w = self.add_weight(
            shape=(self.n_experts, self.d_model, self.d_model),
            initializer="glorot_uniform", name="expert_weights")
        self.swiglu = SwiGLU()
        self.expert_dropout = layers.Dropout(self.expert_dropout_rate)
        # Shared expert: always active for every token, no routing.
        self.shared_dense  = layers.Dense(self.d_model)
        self.shared_swiglu = SwiGLU()

    def call(self, x, context=None, training=None):
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
        routed_out = self.expert_dropout(self.swiglu(weighted_inputs), training=training)

        # Shared expert path — always active, adds common-pattern capacity
        # that doesn't have to compete for routing gradient.
        shared_out = self.shared_swiglu(self.shared_dense(x))
        weighted_avg = routed_out + shared_out

        # Convergence Consensus signal (routed experts only — measures routing agreement)
        diff_sq       = ops.square(expert_outputs - ops.expand_dims(routed_out, axis=2))
        weighted_var  = ops.sum(diff_sq * ops.expand_dims(top_k_weights, axis=-1), axis=2)
        consensus     = ops.exp(-ops.mean(weighted_var, axis=-1))

        # Entropy load balancing (prevents expert collapse). The regularizer's
        # ceiling is log(n_experts) (max entropy = uniform routing), so a
        # coefficient tuned against log(256)~5.545 was implicitly ~60% too
        # strong once n_experts dropped to 32 (log(32)~3.466) — rescaled here
        # so the same proportional pressure applies regardless of expert count,
        # instead of silently drifting every time n_experts changes.
        entropy = -ops.mean(ops.sum(gate_scores * ops.log(gate_scores + 1e-9), axis=-1))
        max_entropy = ops.log(ops.cast(self.n_experts, "float32"))
        entropy_coef = -1e-4 * (max_entropy / ops.log(256.0))
        self.add_loss(entropy_coef * entropy)

        return weighted_avg, consensus

    def get_config(self):
        config = super().get_config()
        config.update({"d_model": self.d_model, "n_experts": self.n_experts,
                        "expert_dropout_rate": self.expert_dropout_rate})
        return config


@keras.saving.register_keras_serializable(package="KAT")
class HydraBlock(layers.Layer):
    """
    V10.7 HydraBlock: MLA + TurboQuant + SwiGLU + MoE + Dropout.
    Dropout applied after BOTH the SwiGLU branch and the MoE output -
    SwiGLU is the largest always-active capacity block (every token, every
    layer) and needs its own regularization, not just the sparse MoE path.
    """
    def __init__(self, d_model=128, n_heads=8, dropout_rate=0.15, **kwargs):
        super().__init__(**kwargs)
        self.d_model      = d_model
        self.n_heads      = n_heads
        self.dropout_rate = dropout_rate

    def build(self, input_shape):
        self.norm1   = RMSNorm()
        self.attn    = MLALayer(d_model=self.d_model, n_heads=self.n_heads, dropout_rate=self.dropout_rate)
        self.tq      = TurboQuant(d_model=self.d_model)
        self.swiglu  = SwiGLU()
        self.norm2   = RMSNorm()
        # Routed/sparse expert capacity gets a HIGHER dropout than the
        # block's own dense-layer rate (~1.5x, capped at 0.5) - standard
        # practice for MoE architectures, since each expert only ever sees a
        # fraction of tokens and is more prone to overfitting than the
        # always-active dense/shared paths around it.
        # n_experts lowered 32 -> 8 alongside the 8x -> 4x block count
        # reduction above - GatedMoE's entropy_coef auto-rescales from
        # n_experts (log(n_experts)/log(256)), so this stays correctly
        # calibrated without a separate change there.
        self.moe     = GatedMoE(d_model=self.d_model, n_experts=8,
                                 expert_dropout_rate=min(0.5, self.dropout_rate * 1.5))
        self.dropout = layers.Dropout(self.dropout_rate)

    def call(self, x, training=None, context=None):
        # Attention path with TurboQuant stabilization. SwiGLU's output was
        # previously added to the residual stream with NO dropout at all -
        # unlike the MoE branch below, it's always-active (every token,
        # every layer, 8 layers deep) and was the largest unregularized
        # capacity block in the model, a real contributor to fast
        # overfitting that weight_decay alone couldn't reach (found via
        # real training data: train_dir_acc racing past 70% while val
        # stayed flat ~52-53%, unchanged whether weight_decay was 0.05 or
        # 0.15 - the missing dropout here, not weight_decay, was the lever).
        attn_out = self.tq(self.attn(self.norm1(x), training=training))
        x = x + self.dropout(self.swiglu(attn_out), training=training)

        # MoE path with dropout regularization
        moe_out, consensus = self.moe(self.norm2(x), context=context, training=training)
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
    - MSE on price trajectory (close, the channel the direction term and
      every trading-decision site actually key on) + a lower-weighted MSE
      on the open/high/low channels (a real full-candle forecast, not
      informing the trade-direction gate itself).
    - Direction loss (sign match) weighted by local volatility.
    - High-volatility moves incur a LARGER penalty when predicted wrong.
    """
    def __init__(self, direction_weight=10.0, ohl_weight=0.5, **kwargs):
        super().__init__(**kwargs)
        self.direction_weight = direction_weight
        self.ohl_weight = ohl_weight

    def call(self, y_true, y_pred):
        p_true = y_true[:, :, 0]
        p_pred = y_pred[:, :, 0]
        v_true = y_true[:, :, 1]  # Volatility channel

        # Base MSE (close)
        mse = ops.mean(ops.square(p_true - p_pred))

        # Direction loss
        p_entry  = p_true[:, 0:1]
        raw_true = p_true[:, 1:] - p_entry
        raw_pred = p_pred[:, 1:] - p_entry
        dir_loss = ops.square(ops.sign(raw_true) - ops.tanh(raw_pred * 10.0))

        # Volatility weighting: errors during high-vol periods penalized more
        vol_weight = ops.abs(v_true[:, 1:]) + 1.0
        weighted_dir = ops.mean(dir_loss * vol_weight)

        # Open/High/Low MSE (channels 3,4,5) - a real full-candle forecast,
        # weighted lower than close since nothing downstream trades on it
        # directly, but it's a genuine trained target, not a dead output.
        ohl_mse = ops.mean(ops.square(y_true[:, :, 3:6] - y_pred[:, :, 3:6]))

        return mse + (self.direction_weight * weighted_dir) + (self.ohl_weight * ohl_mse)

    def get_config(self):
        config = super().get_config()
        config.update({"direction_weight": self.direction_weight, "ohl_weight": self.ohl_weight})
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
    Trains certainty (reasoning's own max-softmax confidence, see
    build_kraken) to be high specifically on real LONG/SHORT setups and low
    on FEE_TRAP/NOISE - not a separate learned head, so this loss's real
    effect is pushing gradient back into reasoning's own logits: sharpen
    the distribution (more confident) when the window is a genuine
    trade, flatten it (less confident) when it isn't.
    y_pred: (B,) reasoning's max-softmax value per window
    y_true: (B, 1) binary target — 1.0 if LONG/SHORT, 0.0 if FEE_TRAP/NOISE
    """
    pred = ops.reshape(y_pred, (-1, 1))  # (B, 1) to match y_true
    return keras.losses.binary_crossentropy(y_true, pred)


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
    """Directional accuracy: did we call the direction of the price move correctly?"""
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
                 dropout_rate=0.30, noise_stddev=0.05, vocab_size=32):
    """
    Equity swing edition — Deep Predator, single-input, four outputs.

    Single INPUT still (no dual market+token input like the old crypto
    version needed) - the next-candle head reads from the same shared
    backbone as prediction/certainty/reasoning, it doesn't need its own
    separate token-embedding input pathway. What it DOES have is a genuine
    next-token classification OUTPUT (see "next_candle" below) - real
    next-token prediction, GPT-style, just single-step (predict tomorrow's
    token once) rather than chained autoregressive generation (sample a
    token, feed it back in as input, repeat for N days) - that generation
    loop is where numeric time series compound errors badly (no grammar-like
    structure to self-correct, unlike text), so it's deliberately not here.

    dropout_rate/noise_stddev raised from the crypto defaults (0.15/0.02) to
    0.30/0.05 for per-symbol training - one stock alone has only ~4-6k
    overlapping (heavily autocorrelated) 60-day windows for a 7.3M-param
    model, a real overfitting risk (observed directly: HDFCBANK's val
    accuracy peaked at epoch 1 and drifted down for several epochs after).
    Neither of these touches the architecture's topology (block count,
    d_model, expert count) - just how hard the existing Dropout/GaussianNoise
    layers regularize.
    """
    inputs = layers.Input(shape=(context_window, n_features), name="market_input")

    # Input Noise Augmentation (only active during training)
    x = layers.GaussianNoise(noise_stddev)(inputs)
    x = RMSNorm()(layers.Dense(128)(x))

    # 4x HydraBlock with dropout - was 8x. Six independent, real fixes
    # (weight_decay, missing SwiGLU dropout, a feature leak, degenerate
    # early-history features, an LR schedule that froze too early, and a
    # ruled-out dropout-during-eval check) all left val_dir_acc capped in
    # the same ~53-55% band across BOTH pooled pretrain (30K windows) and
    # single-symbol fine-tune (~4K windows) - a pattern much more
    # consistent with the model being oversized for the actual data
    # (~7.3M params, 8 layers x 32 experts each) than with one remaining
    # hidden bug. Halved here as a direct test of that theory.
    for i in range(4):
        x, _ = HydraBlock(d_model=128, n_heads=8,
                          dropout_rate=dropout_rate, name=f"hydra_{i}")(x)

    # Output heads - preserve dynamic temporal sequence ordering using last-token extraction
    normed_x  = RMSNorm()(x)
    last_step = layers.Lambda(lambda t: t[:, -1, :])(normed_x)

    # 6 channels: [close, volatility, volume, open, high, low] - a real full
    # candle forecast (not just close), close stays channel 0 since
    # SovereignLoss's direction term and every inference site key on that
    # specific position.
    preds = layers.Reshape(
        (forecast_steps + 1, 6), name="prediction")(
        layers.Dense((forecast_steps + 1) * 6)(last_step))

    # Label smoothing(0.1) on reasoning to prevent overconfidence
    reasoning = layers.Dense(4, activation="softmax", name="reasoning")(last_step)

    # Certainty = the reasoning head's OWN max-softmax confidence (a
    # standard, well-established confidence measure - not a new trainable
    # head). Previously derived from MoE expert-routing agreement
    # (per-block consensus, stacked + calibrated) - measured directly on a
    # real 23-year NIFTYBEES history and found collapsed to a near-constant
    # ~0.33 (matching the training label's ~31% LONG/SHORT base rate almost
    # exactly - the textbook signature of a classifier that gave up
    # discriminating and just learned the marginal probability), zeroing
    # out every trade under ml_strategy.py's real gate despite the
    # reasoning/price/next-candle signals all looking healthy. MoE-routing
    # agreement measures the model's own internal stability, not "is this
    # a real trading opportunity" - reasoning's max-softmax is mathematically
    # bounded to vary with how sharply reasoning discriminates (can't
    # collapse the same way unless reasoning itself goes uniform, which
    # would show up as a different, already-visible problem), and it's the
    # semantically correct thing to call "certainty" for a gate that reads
    # reasoning's own class call.
    certainty = layers.Lambda(
        lambda r: ops.max(r, axis=-1), name="certainty")(reasoning)

    # GPT-style next-candle token head: real next-token classification over
    # a quantile-binned return vocabulary (see preprocess.py's
    # fit_return_vocab/tokenize_returns) - single-step prediction of
    # tomorrow's discretized return bucket, trained with next-token
    # cross-entropy same as a language model, from the same shared backbone.
    next_candle = layers.Dense(vocab_size, activation="softmax", name="next_candle")(last_step)

    model = keras.Model(
        inputs, [preds, certainty, reasoning, next_candle],
        name="sovereign_kraken_equity_v1")

    # Deliberately UNCOMPILED here. train.py's _compile_hydra() is the one real
    # compile step for both the pretrain and fine-tune phases (LR schedule,
    # class-weighted reasoning loss, loss_weights - all tuned per-phase there).
    # An earlier version of this function also called model.compile() with
    # its own, different hyperparameters (a fixed 1e-4/10,000-step schedule,
    # unweighted reasoning loss, next_candle loss_weight 1.0 vs train.py's
    # 0.5) - since train.py always immediately recompiles anyway, that block
    # was dead code that just documented a stale, inconsistent config anyone
    # reading this file could mistake for what's actually used. Removed
    # rather than fixed in place, so there's exactly one source of truth for
    # how this model is compiled. Inference-only callers (ml_strategy.py,
    # tests) don't need a compiled model either - only .fit()/.evaluate() do.
    return model
