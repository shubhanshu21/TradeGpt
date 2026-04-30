# ⚓ Sovereign Kraken (KAT) Engine — V11.1 SOVEREIGN ALPHA 🏛️
> **Autonomous 15-Minute BTC Swing Trading | 256-Expert MoE | Fee-Aware Reasoning (FAR) | Sovereign Goal Paradigm**

<p align="center">
  <img src="media/banner.png" width="100%" alt="Kraken Banner">
</p>

---

## 🏛️ V11.1 "Sovereign Alpha — Phase 6"

V11.1 marks the evolution from simple prediction to **Sovereign Decision Making**. By integrating **Fee-Aware Reasoning (FAR)** directly into the model's neural weights, Iron Oracle no longer just predicts the market—it evaluates the **Business Case** for every trade.

### 🏛️ Sovereign Capabilities:
- 🛡️ **Fee-Aware Reasoning (FAR)**: Internal "Reasoning Head" distinguishes between **Sovereign Profit** and **Fee Traps**.
- 🎯 **Sovereign Goal Paradigm**: Model is trained to ignore "Noise" (small moves) and focus exclusively on signals with >2x Fee Advantage.
- ⚙️ **Dynamic Sovereign Config**: Centralized control over fees and "Greed Level" via `src/config/sovereign_config.py`.
- 🧠 **MLA + RoPE + MoE**: 256-Expert Mixture-of-Experts with **Rotary Positional Encoding** for 30-hour temporal context.

---

## 📐 Architecture Overview (V11.1)

```
Market Input (120 × 42 features)
       │
   [Dense → RMSNorm]       ← 42 → 128 embedding
       │
   ┌───┴──── × 8 ────────┐
   │    HydraBlock V11.1  │
   │  ┌─────────────────┐ │
   │  │  MLALayer+RoPE  │ │  ← time-aware latent attention
   │  │  GatedMoE-256   │ │  ← top-4 expert routing
   │  │  Dropout(0.1)   │ │  ← swing robustness
   │  └─────────────────┘ │
   └──────────────────────┘
       │
   Global Pooling → [Prediction] [Certainty] [Reasoning Head]
                                            (Fee-Aware Filter)
```

---

## 📦 Sovereign Reasoning Labels
The model is now trained to classify every market condition into four "Business States":

| Class | Label | Meaning |
|-------|-------|---------|
| **0** | **SOVEREIGN_LONG** | Predicted Profit > 2x Fees (High Conviction) |
| **1** | **SOVEREIGN_SHORT** | Predicted Profit > 2x Fees (High Conviction) |
| **2** | **FEE_TRAP** | Direction correct, but move too small to cover fees |
| **3** | **NOISE** | Choppy / Sideways market (Stay in Cash) |

---

## 🚀 Quick Start Commands

| Task | Command |
| :--- | :--- |
| 🔥 **Start Sovereign Training** | `nohup sudo /root/miniconda3/bin/python train.py --symbol BTCUSD --timeframe 15m --epochs 300 --batch 64 > logs/iron_oracle_v11.log 2>&1 &` |
| ◀️ **Restart Dashboard** | `sudo kill $(sudo lsof -t -i:5000) && sudo nohup /root/miniconda3/bin/uvicorn src.api.serve:app --host 0.0.0.0 --port 5000 > logs/dashboard.log 2>&1 &` |
| 📡 **Watch Live Status** | `tail -f logs/iron_oracle_v11.log` |
| 💰 **Run ROI Benchmark** | `sudo /root/miniconda3/bin/python scripts/calc_net_roi.py` |

---

## ⚙️ Sovereign Configuration (`src/config/sovereign_config.py`)
Modify these values to adapt to exchange changes instantly:
*   `CURRENT_FEE_PCT`: Update when exchange fees change (Default: 0.0006).
*   `SOVEREIGN_MULTIPLIER`: Your "Greed Level" (Default: 2.0x fee cover).

---

⚓ **Sovereign Kraken V11.1 "Iron Oracle" — Intelligence that understands Profit.**