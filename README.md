# ⚓ Sovereign Kraken (KAT) Engine
## V10.3 "Singularity" Mission Center

Sovereign Kraken is an autonomous, institutional-grade neural trading engine optimized for 1-minute BTC/USD scalping. Built on a 1,152-expert Mixture-of-Experts (MoE) architecture with Real-Time Certainty Tracking.

---

### 📂 Engine Architecture (Modular V10.3)

```bash
/var/www/html/ML/kat/
├── src/
│   ├── core/           # 🧠 Hydra Neural Engine & Loss Logic
│   ├── data/           # 🌊 Preprocessing & Abyss-Streamers
│   ├── exchange/       # 📡 Delta API & Data Fetchers
│   ├── evaluation/     # 📊 Backtesting & Visual Analytics
│   └── trading/        # 🦾 Live Execution & Agent Pilot
├── scripts/            # 🛠️ Maintenance (Finetune, Quantize)
├── models/             # 💾 Model Weights & Scalers
├── logs/               # 📜 Mission Telemetry
└── train.py            # 🔥 Master Orchestrator
```

---

### 🕹️ Mission Control Commands

#### 1. Training (Ignite the Brain)
Starts the 300-epoch Singularity training cycle.
```bash
nohup python train.py --symbol BTCUSD --epochs 300 --batch 64 2>&1 | tee logs/omni_brain_300.log &
```

#### 2. Performance Evaluation (Tactical Audit)
Run a walk-forward profit simulation and generate a trend honesty chart.
```bash
python src/evaluation/backtest_checkup.py
python src/evaluation/visualize_backtest.py
```

#### 3. Maintenance (Daily Adaptation)
"Nudge" the model with the latest 24-hours of market data.
```bash
python scripts/daily_finetune.py
```

#### 4. Hardening (INT8 Deployment)
Compress the model for high-speed, microsecond live execution.
```bash
python scripts/quantize_model.py
```

---

### 📊 Engine Specifications (V10.3)
- **Neural Model**: Hydra V10.3 (Dual-Output: Price + Certainty)
- **Expert Council**: 1,152 Sparse MoE Experts (Top-4 Gating)
- **Prediction Horizon**: 15 Minutes (Multi-Target Forecasting)
- **Features**: 27 High-Resolution Indicators (CVD, Volatility, Liquidation Proxy)
- **Target Accuracy**: 55%+ Win-Rate for BTC Scalping
- **Protection**: Entry-Anchored Loss + Label Smoothing

---

### 🛡️ Safety & Safeguards
- **Abyss Streaming**: Lazy-loading data pipeline to prevent OOM errors on limited RAM.
- **Firewall Logic**: strict zero-leakage sliding window validation.
- **Certainty Filter**: Only trades when the cumulative expert consensus crosses the 110 threshold.

⚓ **Sovereign Kraken — Hunting the Alpha.**