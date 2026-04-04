# Neural Trading Engines — Autonomous Decision Suite

Professional-grade AI trading intelligence for **Delta Exchange India**.  
Designed to analyze multi-timeframe price action and execute autonomous trades with institutional-level risk management.

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
| **HYDRA (Elite)** | **16 Blocks Deep**. Uses **Attention Residuals** (Selective Memory) and **MoE**. | **"Best-of-the-Best" Strategic Trading** |
| **CAUSAL (V2)** | Predicts the full "path" of a price move using Causal Transformers. | **Trajectory forecasting** |
| **TITAN (V1)** | A heavy-duty, hybrid Bi-LSTM + Transformer model. | **Complex market analysis** |
| **ALPHA (V0)** | Our classic, lightweight baseline model. | **Simple price prediction** |

## 🚀 The HYDRA "Elite" Advantage

The HYDRA Engine is a **16-Block architecture**, making it one of the most sophisticated time-series models available.

### 🛡️ **Technical Specs:**
- **Dynamic Timeframes (New)**: Support for `1m`, `5m`, `15m`, `1h`, or any timeframe.
- **Attention Residuals (AttnRes)**: Instead of simple skip connections, each layer "selectively memory-checks" every previous layer.
- **RMSNorm**: Root Mean Square Layer Normalization for extreme numerical stability at high depth.
- **MTP (Multi-Token Prediction)**: Predicts the next 5 minutes simultaneously to ensure high-conviction forecasting.
- **MLA (Multi-Head Latent Attention)**: Highly efficient context compression for large lookback windows.

---

## 🦾 How to Train (Simulation)

Before trading, the AI needs to "go to school" on historical data.

### **1. Deep Train (Elite 300-Epoch Mission)**
Fetches 6 weeks of data and performs deep neural optimization.
```bash
# Example: Train HYDRA on 5-minute candles
python auto_run.py train --model hydra --timeframe 5m --epochs 300 --candles 60000
```

### **2. Live Trading (Autonomous Pilot)**
Launches the real-time Bloomberg-style dashboard and starts the autonomous loop.
```bash
# Example: Start the HYDRA pilot on a 15-minute timeframe
python auto_run.py trade --model hydra --timeframe 15m
```

### **3. Smart Caching**
The system automatically caches datasets to avoid re-fetching from the exchange. Caches are separated by **Symbol** and **Timeframe** to ensure zero data pollution.

---

## 🛡️ License
MIT / Proprietary