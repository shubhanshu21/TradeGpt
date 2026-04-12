---
description: Sovereigh Kraken — Mission Operations & Safeguards
---

## ⚓ Mission Log: V5.1 "Sovereign-Persistence" Patch

This workflow documents the critical architectural lessons learned during the 300-epoch training mission of the Hydra V5.0 brain. Use these safeguards to prevent regressions during long-running CPU training.

### 1. The Persistent LR Rule
**Problem**: Keras `optimizer.iterations` resets to 0 on script restart, even if weights are loaded. This causes "Thermal Shock" by jumping back to high Initial LR (e.g., 5e-4) in the middle of a mission.
**Safeguard**: 
- Always calculate `initial_step = current_epoch * steps_per_epoch` on resume.
- Use `optimizer.iterations.assign(initial_step)` before starting `fit()`.
- Ensure `initial_epoch` is passed to `fit()` for log continuity.

### 2. The Alpha Priority (Loss Weighting)
**Problem**: The model often prioritizes Absolute Price (MSE) over Directional Accuracy (Alpha). This leads to low win rates even if Val Loss is low.
**Safeguard**: 
- Maintain `direction_weight >= 10.0` in `SovereignLoss`.
- Use **Label Smoothing** (0.1) on directions to prevent overconfidence on micro-noise.
- Use **Huber Loss** for raw prices to reduce influence of outlier wicks.

### 3. Hardware Stability (CPU Peak)
**Problem**: Training is CPU-bound on this host. CTX_WIN > 120 or Batch > 256 leads to activation-buffer bloat and system hangs.
**Safeguard**:
- Context Window: **120 candles** (2 hours).
- Batch Size: **128** (Saturates ~15GB RAM perfectly).
- **Abyss-Streamer**: Use `tf.data.from_generator` to ensure zero-copy RAM usage.

### 4. Backtest Protocol
**Problem**: Training loss is a weak proxy for trading profit.
**Requirement**:
- Run `src/backtest_checkup.py` every ~20 epochs.
- Look for **Hold Rate** improvements as a sign of "maturity."
- Expect "Buy-Only" alpha (Long Win Rate) to mature faster than "Short-Only" alpha.

---
*Documented by Antigravity — 2026-04-12*
