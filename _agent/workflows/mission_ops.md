---
description: Sovereigh Kraken — Mission Operations & Safeguards
---

## ⚓ Mission Log: V10.0 "Singularity" Standard

This workflow documents the critical architectural lessons learned during the construction of the 1,152-expert Singularity brain. These safeguards are mandatory to survive the 300-epoch mission.

### 1. The "Infinity Hammer" Rule (Stability)
**Problem**: BTC market wicks or exchange freezes can produce `inf` or `nan` values in indicators like VWAP or Funding Proxy. This poisens the 1,152-expert ensemble instantly on Step 1.
**Safeguard**: 
- `compute_indicators` must scrub all `inf` values using `df.replace([np.inf, -np.inf], 0)`.
- Use `.clip(-1e9, 1e9)` as a final defense before returning to the model.

### 2. The Unbreakable Scaler
**Problem**: Zero-variance columns (constant price/volume) cause "divide-by-zero" errors in standard scalers, leading to NaN loss.
**Safeguard**: 
- `KATScaler` must force `std = 1.0` if `std == 0` is detected in any feature column.

### 3. Hardware Stability (21GB Singularity Peak)
**Problem**: 96 Experts per layer create a massive neural graph. High batch sizes (>64) or massive buffers (>30k) will cause OOM freezes.
**Safeguard**:
- **Batch Size: 64** (Optimal for backprop on 4-core CPUs).
- **Shuffle Buffer: 20,000** (Provides randomization while staying under the 23.4GB RAM ceiling).
- **Target RAM: 21GB** (Stable peak).

### 4. The Precision Flow Rule
**Problem**: Large ensembles (MoE) are difficult to converge with high initial learning rates.
**Safeguard**:
- Use **2e-4** for initial LR.
- Always use `clipnorm=1.0` in the optimizer to prevent gradient spikes in the Gated Router.

---
*Updated to V10.0 Singularity by Antigravity — 2026-04-12*
