# Neural Trading Engines — Autonomous Decision Suite

Professional-grade AI trading intelligence for **Delta Exchange India**.  
Designed to analyze high-frequency price action and execute autonomous trades with institutional-level risk management.

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
| **HYDRA (V3)** | Our most advanced brain. Uses 4 specialized "experts" to handle different market regimes. | **High-precision trading** |
| **CAUSAL (V2)** | Predicts the full "path" of a price move, like GPT predicts a sentence. | **Trajectory forecasting** |
| **TITAN (V1)** | A heavy-duty, high-capacity hybrid model. | **Complex market analysis** |
| **ALPHA (V0)** | Our classic, lightweight baseline model. | **Simple price prediction** |

## 🚀 How to Trade (Live)

Launches the autonomous trading pilot on the Delta India Testnet. Includes a real-time dashboard.

```bash
# Start the HYDRA pilot on BTCUSD
python auto_run.py trade --model hydra
```

### 🛡️ **Built-in Safety (Automatic):**
- **Dynamic Risk**: Automatically widens or tightens your Stop-Loss and Take-Profit based on current market volatility (ATR).
- **Directional Check**: Only enters trades when it's highly confident in the direction, not just the price.
- **Heartbeat**: Re-calculates every 60 seconds to stay perfectly synced with the ticker.

## 🦾 How to Train (Simulation)

Before trading, the AI needs to "go to school" on historical data.

### **1. Deep Learning (41 days of history)**
```bash
python auto_run.py train --model hydra --epochs 300 --candles 60000
```

### **2. Quick Refresh (Latest 24 hours)**
```bash
python auto_run.py train --model hydra --epochs 10 --candles 2000 --finetune
```

---

## 🛡️ License
MIT / Proprietary