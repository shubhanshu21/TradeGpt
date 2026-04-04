# Neural Trading Engines — Sovereign Quant Suite

Sovereign-grade AI trading intelligence for **Delta Exchange India**.  
Designed to analyze multi-scale, multi-timeframe price action and execute autonomous trades with institutional-level risk management.

## 🏗️ Project Overview

```
auto_run.py              ← The "Remote Control" (Train / Predict / Trade)
│
├── src/
│   ├── architectures/   ← The "Brains" (Hydra, Causal, Titan, Alpha)
│   ├── delta_client.py  ← The "Exchange Link" (Connecting to India Testnet)
│   └── preprocess.py    ← The "Data Filter" (Cleaning raw market info)
│
├── train.py             ← The "Simulation Lab" (Training the AI)
└── live_trader.py       ← The "Autonomous Pilot" (Live trading & HUD)
```

## 🧠 The AI Lineup

| Engine | Description | Best For |
| :--- | :--- | :--- |
| **HYDRA (Sovereign)** | **V3.5 Elite Quant**. Uses **VSN**, **Dual-Scale Fusion**, and **Gated MoE balanced experts**. | **Institutional-Grade Decision Making** |
| **CAUSAL (V2)** | Predicts the full "path" of a price move using Causal Transformers. | **Trajectory forecasting** |
| **TITAN (V1)** | A heavy-重量, hybrid Bi-LSTM + Transformer model. | **Complex market analysis** |
| **ALPHA (V0)** | Our classic, lightweight baseline model. | **Simple price prediction baseline** |

## 🚀 The HYDRA "Sovereign" Advantage (V3.5)

The HYDRA Engine is a **16-Block architecture**, making it one of the most sophisticated time-series models available. It has been upgraded with **Google TFT** and **MoonshotAI** inspired refinements.

### 🛡️ **Institutional Specs:**
- **VSN (Variable Selection Network)**: Dynamically weights the most "truthful" features per minute (e.g. Price over noisy volume).
- **Dual-Scale Temporal Fusion**: Every block fuses **1m high-frequency bars** with **15m Macro-Trend** aggregates to avoid Bull/Bear traps.
- **Active Expert Balancing**: Mathematically forces MoE experts to specialize in different market regimes (Bull/Bear/Chop/Volatile).
- **Attention Residuals (AttnRes)**: Selective memory across the 16 Elite blocks.
- **RMSNorm**: Root Mean Square Layer Normalization for extreme numerical stability at high depth.
- **MTP (Multi-Token Prediction)**: Predicts the next 5 minutes simultaneously.

---

## 🦾 How to Train (Simulation)

Before trading, the AI needs to "go to school" on historical data.

### **1. Deep Train (Sovereign 300-Epoch Mission)**
Fetches 6+ weeks of data and performs deep neural optimization.
```bash
# Example: Train Sovereign HYDRA on 1-minute candles
python auto_run.py train --model hydra --timeframe 1m --epochs 300 --candles 60000
```

### **2. Live Trading (Autonomous Pilot)**
Launches the real-time Bloomberg-style dashboard and starts the autonomous loop.
```bash
# Example: Start the Sovereign pilot on a 5-minute timeframe
python auto_run.py trade --model hydra --timeframe 5m
```

### **3. Stability Rails (4-Core / 24GB Optimized)**
The system is pre-configured to limit CPU parallelism for server safety and use the `tf.data` streaming pipeline for extreme memory efficiency.

---

## 🛡️ License
MIT / Proprietary