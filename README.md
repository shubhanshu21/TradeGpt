# Sovereign Quant Suite (V3.7) 🏛️📈

Institutional-grade AI trading intelligence for **Delta Exchange**.  
Designed to analyze multi-scale price action, order-book imbalances, and market regimes using **Mixture-of-Experts (MoE)** Transformers.

---

## 🏗️ Technical Architecture
The suite is powered by the **Sovereign Engine (HYDRA V3.7)**, optimized for high-performance execution on resource-constrained 4-core servers.

### 🧠 The Sovereign Stack (V3.7):
*   **MoE (Mixture of Experts)**: 16 specialized experts per block (top-k gated) to dominate Bull, Bear, and Chop regimes independently.
*   **MLA (Multi-Head Latent Attention)**: DeepSeek-v3 inspired attention that captures macro-trends and micro-volatility with minimal CPU overhead.
*   **VSN (Variable Selection Network)**: Inherited from Google's TFT to dynamically filter out feature noise (e.g. ignoring low volume during high-conviction moves).
*   **MTP (Multi-Target Prediction)**: Generates a 5-step future price curve (1–5 minutes) rather than a single buy/sell guess.
*   **Fee-Aware Logic**: Penalizes "Directional Churn" to ensure the model only trades when profit outpaces exchange fees.

---

## 🚀 Operational Workflow

### **1. The "Grand Mastery" (Grand Training)**
Builds the foundational neural patterns from 60,000+ candles (Approx. 6 weeks of data).
```bash
# Optimized for 4-core CPU / 24GB RAM
python auto_run.py train --model hydra --timeframe 1m --epochs 300 --candles 60000
```

### **2. Live Trade Mode (Autonomous Pilot)** 📡
Deploys the `hydra_best.keras` brain to live market execution with a real-time dashboard.
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
├── architectures/       ← The Brains (Hydra, MLA, MoE, VSN)
├── delta_client.py      ← Exchange Bridge (API Authentication/Execution)
└── preprocess.py        ← Neural Data Pipeline (Memory-efficient sliding windows)
```

---

## 🛡️ Sovereign Attributes
*   **CPU Optimized**: Uses Matrix-Sharding and Tiled MatMul to achieve 5s/step velocity on standard 4-core CPUs.
*   **Memory Efficient**: Utilizes NumPy Stride Tricks (`sliding_window_view`) to reduce dataset memory footprint by 95%.
*   **Serialization Ready**: Full Keras 3 `get_config` implementation for stable checkpointing and continuous fine-tuning.

---
**Status**: 🟢 MISSION V3.7 ACTIVE (8-Block Pivot)  
**Target**: BTC/USD Perpetuals  
**License**: MIT / Proprietary Sovereign Development  