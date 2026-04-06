# Sovereign Quant Suite (V4.2) 🏛️📈

Institutional-grade AI trading intelligence for **Delta Exchange**.  
Designed to analyze multi-scale price action, order-book imbalances, and market regimes using **Mixture-of-Experts (MoE)** Transformers and LLM-native Attention.

---

## 🏗️ Technical Architecture
The suite is powered by the **Sovereign Engine (HYDRA V4.2)**, optimized for high-performance execution on resource-constrained servers and specialized NVIDIA A40 clusters.

### 🧠 The Sovereign Stack (V4.2):
*   **RoPE (Rotary Positional Embeddings)**: Injecting relative temporal awareness, allowing the model to distinguish between Micro-noise and Macro-trends.
*   **MoE (Mixture of Experts)**: 8-32 specialized experts (Hardware-Adaptive) using **Regime-Aware Routing** to dominate Bull, Bear, and Chop regimes.
*   **MLA (Multi-Head Latent Attention)**: DeepSeek-V3 inspired attention that captures complex price dependencies while drastically reducing Memory/KV cache overhead.
*   **MTP-15 (Auxiliary Multi-Target Prediction)**: Generates a 15-minute future curve for **Price**, **Volatility**, and **Volume Urgency** simultaneously.
*   **Sovereign Loss (Directional)**: Penalizes "Directional Errors" 3x more heavily than price error, ensuring high-conviction trend following.

---

## 🚀 Operational Workflow

### **1. The "Grand Mastery" (Mission Ignition)**
Builds the foundational neural patterns from 45,000+ candles (Approx. 4 weeks of data).
```bash
# Optimized for 4-core CPU / 24GB RAM
python auto_run.py train --model hydra --timeframe 1m --epochs 300 --candles 45000
```

### **2. Sovereign Alpha Pilot (Autonomous Live)** 📡
Deploys the `hydra_best.keras` brain to live market execution with dynamic volatility-aware trade thresholds.
```bash
# Deploy with professional risk-management (SL/TP)
python auto_run.py trade --model hydra --timeframe 1m
```

---

## 📂 Project Structure

```text
auto_run.py              ← Master CLI Control (Train / Research / Trade)
models/                  ← Neural Checkpoints (Check for hydra_best.keras)
data/                    ← Persistent Market Cache (Parquet format)
src/
├── architectures/       ← The Brains (Hydra V4.2, MLA, MoE, RoPE)
├── delta_client.py      ← Exchange Bridge (API Authentication/Execution)
└── preprocess.py        ← Neural Data Pipeline (Memory-efficient sliding windows)
```

---

## 🛡️ Sovereign Attributes
*   **CPU Optimized**: Uses Matrix-Sharding and Tiled MatMul to achieve high-velocity training on standard CPUs.
*   **Memory Efficient**: Utilizes NumPy Stride Tricks (`sliding_window_view`) to reduce dataset memory footprint by 95%.
*   **Volatility-Aware**: Real-time adaptive thresholds prevent "Fee-Burning" during low-conviction noise.

---
**Status**: 🟢 MISSION V4.2 ACTIVE (RoPE Mastery)  
**Target**: BTC/USD Perpetuals  
**License**: MIT / Proprietary Sovereign Development  