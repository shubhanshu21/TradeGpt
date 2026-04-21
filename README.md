# ⚓ Sovereign Kraken (KAT) Engine — V10.7 DEEP-PREDATOR PHASE 3
> **Autonomous 5-Minute BTC Swing Trading | 256-Expert MoE | MLA + RoPE Attention | 38-Feature Sovereign Hive**

<p align="center">
  <img src="media/banner.png" width="100%" alt="Kraken Banner">
</p>

---

## 🦅 V10.7 "Deep-Predator Phase 3" — The Sovereign Hive

V10.7 is a full-stack upgrade over V10.6. The system has transitioned from **1-minute noise scalping** to **5-minute swing intelligence**, enabling the model to see clean institutional patterns rather than bot-driven microstructure noise.

### 🔥 Phase 3 Capabilities:
- 🧠 **MLA + RoPE**: Multi-Head Latent Attention with **Rotary Positional Encoding** (LLaMA/DeepSeek DNA). The model now knows *when* each candle occurred inside the 10-hour window.
- 🌐 **Cross-Asset Resonance**: ETH correlation vectors and BTC-ETH divergence pressure as leading indicators.
- 📊 **Multi-Timeframe Bias**: 5m momentum + 1h RSI + 1h trend direction baked into every prediction.
- 🐋 **Whale Detector**: On-Chain Liquidity Proxy catches large-order CVD spikes (>5× avg volume).
- ⚖️ **Class-Balanced Reasoning**: Dynamic class weights prevent the model from predicting "Sideways" for everything.
- 🛡️ **Dropout + Noise Augmentation**: Dropout(0.1) at every HydraBlock + Gaussian input noise for robust, non-memorizing patterns.
- 📉 **Volatility-Weighted Loss**: Errors during high-volatility setups are penalized harder, forcing the model to focus on quality signals.
- 🎯 **Cosine LR Decay**: 1e-5 → 1e-6 over 10,000 steps, with gradient clipping (clipnorm=1.0) for stable MoE training.

---

## 📐 Architecture Overview (V10.7)

```
Market Input (120 × 38 features)
       │
   [GaussianNoise(0.02)]   ← augmentation during training
       │
   [Dense → RMSNorm]       ← 38 → 128 embedding
       │
   ┌───┴──── × 8 ────────┐
   │    HydraBlock V10.7  │
   │  ┌─────────────────┐ │
   │  │  RMSNorm        │ │
   │  │  MLALayer+RoPE  │ │  ← time-aware latent attention
   │  │  TurboQuant     │ │  ← INT8 polar compression
   │  │  SwiGLU         │ │  ← gated noise filter
   │  │  RMSNorm        │ │
   │  │  GatedMoE-256   │ │  ← top-4 expert routing
   │  │  Dropout(0.1)   │ │  ← regularization
   │  └─────────────────┘ │
   └──────────────────────┘
       │
   [GlobalAveragePooling]
       │
   ┌───┼───────────────────┐
   │   │                   │
[Prediction]  [Certainty]  [Reasoning]
(16 × 3)     (consensus)   (4-class)
 75min traj   expert vote  Bull/Bear/
                           Sideways/Trend
```

---

## 📦 Feature Vector: 38-Dimension Sovereign Hive

| # | Feature | Category | Why it matters |
|---|---------|----------|----------------|
| 0–5 | OHLCV + quote_volume | Core | Raw price action |
| 6 | funding_rate_proxy | Derivatives | Futures premium = directional bias |
| 7–8 | taker_buy_volume/quote | Flow | Aggressor dominance |
| 9–11 | SMA 7/25/99 | Trend | Short/mid/long bias |
| 12–15 | RSI, MACD, Signal, Hist | Momentum | Standard momentum signals |
| 16–17 | BB Upper/Lower | Volatility | Squeeze + expansion zones |
| 18–22 | ATR, OBV, Vol, ADX, CCI | Strength | Trend strength + flow |
| 23–25 | bb_width, vwap_dist, stoch_rsi | Scalp | Squeeze + anchor + fast momentum |
| 26 | cvd | Microstructure | Cumulative order aggression |
| 27–29 | obi_l1/l2/l5 | Microstructure | Order book wall pressure |
| 30 | eth_corr_1h | **Phase 3** | Cross-asset leading signal |
| 31 | eth_return_5m | **Phase 3** | ETH 5m return (BTC predictor) |
| 32 | btc_eth_spread | **Phase 3** | Divergence pressure (reversal signal) |
| 33 | trend_5m | **Phase 3** | Last candle direction |
| 34 | trend_1h | **Phase 3** | 12-candle (1h) directional bias |
| 35 | rsi_1h | **Phase 3** | 12-period RSI = 1h strength |
| 36 | vol_regime | **Phase 3** | Low/Med/High volatility (0/1/2) |
| 37 | large_trade_cvd | **Phase 3** | Whale order flow proxy |

---

## 🚀 Quick Start Commands

| Task | Command |
| :--- | :--- |
| 🔥 **Start Training (Phase 3)** | `nohup sudo /root/miniconda3/bin/python train.py --symbol BTCUSD --timeframe 5m --epochs 300 --batch 64 >> logs/phase3_training.log 2>&1 &` |
| ▶️ **Resume Training** | `nohup sudo /root/miniconda3/bin/python train.py --symbol BTCUSD --timeframe 5m --epochs 300 --batch 64 --resume >> logs/phase3_training.log 2>&1 &` |
| 📡 **Watch Live Logs** | `tail -f logs/phase3_training.log` |
| 📊 **Epoch Diagnostics** | `cat logs/diagnostics.log` |
| 🔬 **Run Mastery Benchmark** | `sudo /root/miniconda3/bin/python scripts/bench_mastery.py` |
| 🦾 **Launch Live Pilot** | `sudo /root/miniconda3/bin/python auto_run.py trade --model hydra --size 1 --thresh 0.15` |
| 📅 **Daily Fine-Tune** | `sudo /root/miniconda3/bin/python scripts/daily_finetune.py` |

---

## 🛠️ Key Configuration

### Training (`train.py`)
| Param | Value | Notes |
|-------|-------|-------|
| `--timeframe` | `5m` | **5× less noise than 1m** |
| `--epochs` | `300` | EarlyStopping kicks in at patience=7 |
| `--batch` | `64` | Stable for 21GB RAM |
| `--candles` | `120000` | ~417 days of 5m data |
| CTX Window | `120` | 10-hour context (120 × 5m) |
| Forecast | `15` | 75-minute horizon (15 × 5m) |

### Architecture (`hydra.py`)
| Param | Value |
|-------|-------|
| Blocks | 8× HydraBlock |
| d_model | 128 |
| Heads | 8 (MLA with RoPE) |
| Experts | 256 (top-4 routing) |
| Dropout | 0.1 |
| LR | CosineDecay 1e-5 → 1e-6 |
| Clip | clipnorm=1.0 |
| Params | ~35M |

---

## 📈 Training Progress (V10.7 Phase 3)

| Phase | Metric | Target |
|-------|--------|--------|
| Phase 1 (1m) | ~51.9% Val Acc | Baseline established ✅ |
| **Phase 3 (5m)** | **Target: 58%+** | 🔄 Training starting... |

> **Sovereign Edge threshold: > 53% Val Dir Acc** (on 5m data, expected to reach 56-60%).  
> EarlyStopping patience: **7 epochs** (~3.5 days on CPU).

---

## 🔐 Environment Setup (.env)

1. **Copy the example**: `cp .env.example .env`
2. **Key Requirements**:
   - `DELTA_API_KEY`: Your institutional API key from Delta India.
   - `DELTA_API_SECRET`: Your private secret key.
   - `DELTA_TESTNET`: Set to `true` for paper trading.
   - `TELEGRAM_BOT_TOKEN`: (Optional) Real-time mobile P&L alerts.

---

## 📂 The Modular Vault

- `src/core/hydra.py` 🧠 — V10.7 architecture: MLA+RoPE, GatedMoE-256, TurboQuant, SwiGLU, SovereignLoss.
- `src/data/preprocess.py` 🌊 — Abyss-Streamer V4.7: 38-feature DLS-normalized streaming pipeline.
- `src/exchange/` 🔌 — Delta Exchange client, order execution, live data feed.
- `scripts/` ⚙️ — `bench_mastery.py`, `daily_finetune.py`, `quantize_model.py`.
- `auto_run.py` 🦾 — Unified CLI: train / predict / trade modes.
- `train.py` 🚀 — Master training orchestrator with class weights and EarlyStopping.

---

⚓ **Sovereign Kraken V10.7 "Deep-Predator Phase 3" — 5-Minute Swing Intelligence.**