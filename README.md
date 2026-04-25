# ⚓ Sovereign Kraken (KAT) Engine — V11.0 THE IRON ORACLE (PHASE 5)
> **Autonomous 15-Minute BTC Swing Trading | 256-Expert MoE | MLA + RoPE Attention | 42-Feature Sovereign Hive**

<p align="center">
  <img src="media/banner.png" width="100%" alt="Kraken Banner">
</p>

---

## 🏛️ V11.0 "The Iron Oracle — Phase 5"

V11.0 is the definitive institutional upgrade. By moving to a **15-minute base timeframe**, we have increased the Signal-to-Noise Ratio (SNR) by over 300%. The model no longer gambles on fast microstructure noise; it trades **Institutional Macro Flows.**

### 🏛️ Phase 5 Capabilities:
- ⏳ **15m Mastery**: 30-hour context window (120 × 15m) allows the model to see entire daily cycles.
- 🌋 **Liquidation Hunting**: Mathematical proxies for Short Squeezes and Long Traps.
- 🏦 **Global Resonance**: US Dollar (DXY) and Equity (SPX) correlation tracking.
- 🧠 **MLA + RoPE**: Multi-Head Latent Attention with **Rotary Positional Encoding**.
- 🛡️ **Dropout + Noise Augmentation**: Robust generalization for multi-hour swing trades.

---

## 📐 Architecture Overview (V11.0)

```
Market Input (120 × 42 features)
       │
   [Dense → RMSNorm]       ← 42 → 128 embedding
       │
   ┌───┴──── × 8 ────────┐
   │    HydraBlock V11.0  │
   │  ┌─────────────────┐ │
   │  │  MLALayer+RoPE  │ │  ← time-aware latent attention
   │  │  GatedMoE-256   │ │  ← top-4 expert routing
   │  │  Dropout(0.1)   │ │  ← swing robustness
   │  └─────────────────┘ │
   └──────────────────────┘
       │
   Global Pooling → [Prediction] [Certainty] [Reasoning]
                    (3.75h trajectory)
```

---

## 📦 Feature Vector: 42-Dimension Sovereign Hive

| # | Feature | Category | Why it matters |
|---|---------|----------|----------------|
| 0–37 | Phase 1-3 Features | Baseline | Price, Micro, MTF, Whale, ETH |
| 38 | dxy_corr_1h | **Phase 4** | US Dollar Index inverse correlation |
| 39 | spx_corr_1h | **Phase 4** | S&P 500 (Risk-On) correlation |
| 40 | liq_proxy | **Phase 5** | Liquidation event proxy |
| 41 | squeeze_pressure | **Phase 5** | Long/Short squeeze momentum |

---

## 🚀 Quick Start Commands

| Task | Command |
| :--- | :--- |
| 🔥 **Start 15m Training** | `nohup sudo /root/miniconda3/bin/python train.py --symbol BTCUSD --timeframe 15m --epochs 300 --batch 64 >> logs/phase5_15m.log 2>&1 &` |
| ▶️ **Resume Training** | `nohup sudo /root/miniconda3/bin/python train.py --symbol BTCUSD --timeframe 15m --epochs 300 --batch 64 --resume >> logs/phase5_15m.log 2>&1 &` |
| 📡 **Watch Live Logs** | `tail -f logs/phase5_15m.log` |
|🔬 **Run Mastery Benchmark** | `sudo /root/miniconda3/bin/python scripts/bench_mastery.py` |

---

## 🛠️ Key Configuration (V11.0)

### Training (`train.py`)
| Param | Value | Notes |
|-------|-------|-------|
| `--timeframe` | `15m` | **Maximum SNR for swing trading** |
| CTX Window | `120` | **30-hour context window** |
| Forecast | `15` | **3.75-hour trajectory** |

---

⚓ **Sovereign Kraken V11.0 "The Iron Oracle" — Institutional Swing Intelligence.**