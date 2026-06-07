# ⚓ Iron Oracle V12.4 — Sovereign Kraken Intelligence
> **Institutional-Grade Autonomous Neural Trading Station | BTC/USDT 15m | 256-Expert MoE Architecture | Fee-Aware Simulation Engine**


<p align="center">
  <img src="src/api/static/img/logo.png" width="180px" alt="Iron Oracle Logo">
  <br>
  <strong>SOVEREIGN NEURAL COMMAND CENTER</strong>
</p>

---

## 🧠 What Is This?

**Iron Oracle** is a self-contained, autonomous AI trading engine built for **BTC/USDT perpetual futures** on the **15-minute timeframe**. It trains a deep neural network on 11+ years of historical market data (400,000 candles) and uses it to generate high-conviction trade signals, complete with institutional-grade risk metrics and a live monitoring dashboard.

The system is designed for a **CPU server (24GB RAM)** and runs fully autonomously — data fetching, training, simulation, and dashboard serving are all handled automatically.

---

## 📁 Project Structure

```
kat/
├── train.py                        ← Main training orchestrator (run this)
├── auto_run.py                     ← Live trading runner (post-training)
├── src/
│   ├── config/
│   │   └── sovereign_config.py     ← Central config (fees, wallet, risk)
│   ├── core/
│   │   └── hydra.py                ← Neural model architecture (HYDRA V10.7)
│   ├── data/
│   │   └── preprocess.py           ← 45-feature data pipeline + streaming dataset
│   ├── api/
│   │   ├── serve.py                ← FastAPI dashboard server (port 5000)
│   │   ├── dashboard.html          ← Live monitoring UI
│   │   └── static/                 ← Local JS/CSS assets (offline-capable)
│   ├── exchange/
│   │   └── fetch_data.py           ← Market data fetcher (Delta Exchange)
│   ├── trading/
│   │   ├── live_trader.py          ← Live trade executor
│   │   └── risk.py                 ← Risk management engine
│   └── evaluation/
│       ├── flash_backtest.py       ← Quick 100-step backtest evaluation
│       ├── backtest_checkup.py     ← Full 1,000-candle walk-forward backtest
│       └── visualize_backtest.py   ← Denormalized close-price zoom graph generator
├── scripts/
│   ├── certainty_audit.py          ← Real-time certainty distribution timelines
│   ├── wfa_backtest.py             ← Walk-forward fine-tuning simulator
│   ├── test_today.py               ← Performance diagnostic for today's market slice
│   ├── ob_collector.py             ← Background orderbook L2 snapshot collector
│   └── daily_finetune.py           ← Daily fine-tuning adapter (cron target)
├── models/                         ← Saved checkpoints (.keras)
├── logs/
│   ├── iron_oracle_v11.log         ← Live training log
│   ├── recent_sim_trades.json      ← Simulation trade feed (dashboard)
│   └── latest_roi.json             ← ROI data per certainty tier
└── data/
    └── BTCUSD_15m_history_*.parquet ← Cached historical candles
```

---

## 🏗️ Neural Architecture (HYDRA V10.7)

```
Market Input (120 candles × 45 features)
        │
   [GaussianNoise(0.02)]    ← Input augmentation (training only)
        │
   [Dense → RMSNorm]        ← 45 → 128 embedding
        │
   ┌────┴──── × 8 ──────────┐
   │      HydraBlock V10.7   │
   │  ┌──────────────────┐   │
   │  │ MLALayer + RoPE  │   │  ← Temporal-Aware Latent Attention
   │  │ TurboQuant       │   │  ← INT8-sim quantization stabilizer
   │  │ GatedMoE-256     │   │  ← 256 experts, top-4 routing
   │  │ Dropout(0.1)     │   │  ← Prevents expert memorization
   │  └──────────────────┘   │
   └─────────────────────────┘
        │
   GlobalAveragePooling1D
        │
   ┌────┼────────────────┐
   │    │                │
[Prediction]  [Certainty]  [Reasoning]
 (16×3 traj)  (256 scores)  (4 classes)
```

### Key Components:
| Component | Description |
|---|---|
| **MLALayer + RoPE** | Multi-Head Latent Attention with Rotary Positional Embeddings — time-aware |
| **GatedMoE-256** | 256 expert sub-networks, top-4 routing, entropy load balancing |
| **TurboQuant** | INT8 simulation with orthogonal rotation — quantization-aware training |
| **SovereignLoss** | Volatility-weighted directional loss — penalizes errors on high-vol moves more |
| **RMSNorm** | Fast root mean square normalization (LLaMA-style) |
| **SwiGLU** | Gated activation function (Gemma/LLaMA DNA) |

---

## ⚡ V12.4 Patch Notes (Diagnostic & Logic Fixes)

1. **L5 Order Book Imbalance (OBI) Fix**: Resolved a `KeyError: 'obi_l5'` in `preprocess.py` by calculating the Level 5 Order Book Imbalance dynamically from Level 1-5 bid/ask volumes, restoring correct calculation of `squeeze_pressure`.
2. **Evaluation/Audit Certainty Alignment**: Updated `certainty_audit.py` to use a standard 0-100% scale and dynamic `CERT_THRESHOLD` check matching the live pilot.
3. **Logic Corrections in Diagnostics**: Fixed price trend delta calculations in `test_today.py` and `backtest_checkup.py` by subtracting anchor entry price `p_anchor`.
4. **Trajectory Mean Evaluation Alignment**: Corrected `backtest_checkup.py` to evaluate realized move using the mean of the 15-candle future trajectory rather than a single point-price change.
5. **Denormalization Fix in Visualizer**: Fixed the price denormalization formula in `visualize_backtest.py` to use `(pred_scaled_ret * std) + mean` instead of `entry_p + (pred_scaled_ret * std)`.
6. **Compounding Visualizer Bug Fix**: Corrected `forecast_visual` in `auto_run.py` to prevent cumulative compounding of `usd_deltas` on the plot.
7. **Walk-Forward Simulator**: Added `wfa_backtest.py` to iteratively test and fine-tune the model on daily chunks of unseen data, accurately simulating a live production environment. Deprecated `test_10days.py`.

---

## ⚡ V12.3 Patch Notes (Critical Architecture Fixes)

1. **Strict Dynamic Local Scaling (DLS)**: Replaced raw `std + 1e-8` normalization during live trading with a unified `apply_dls()` function synced precisely with the training pipeline. This enforces strict Standard Deviation flooring (`1e-3`) and Z-score bounding (`[-5.0, 5.0]`), permanently killing the "Neural Hallucination" bug that caused random trading during low volatility periods.
2. **Delta API Exponential Backoff Engine**: The Delta client (`delta_client.py`) is now hardened against `502 Bad Gateway` and timeout errors with intelligent retry backoffs. Background collectors and execution traders no longer crash during API outages.
3. **Optimized I/O Checkpointing**: `train.py` now saves its 400MB+ `.keras` models every 10 epochs instead of every 1 epoch. This completely eliminates CPU/Disk I/O stalling and accelerates the training pipeline massively.
4. **Order Book Data Collector**: The L2 Order Book snapshot collector now properly logs output to `logs/ob_collector.log` while perfectly executing its 15-minute sync cycles.

---

## 📊 45 Input Features

| Category | Features |
|---|---|
| **Time** | `hour_sin`, `hour_cos`, `day_of_week` |
| **Price OHLCV** | `open`, `high`, `low`, `close`, `volume`, `quote_volume` |
| **Microstructure** | `funding_rate_proxy`, `taker_buy_volume`, `taker_buy_quote_volume`, `cvd`, `obi_l1`, `obi_l2`, `obi_l5` |
| **Momentum** | `rsi`, `macd`, `macd_signal`, `macd_hist`, `adx`, `cci`, `stoch_rsi` |
| **Trend/MA** | `sma_7`, `sma_25`, `sma_99`, `bollinger_upper`, `bollinger_lower`, `bb_width`, `vwap_dist` |
| **Volatility** | `atr`, `volatility`, `vol_regime`, `squeeze_pressure`, `liq_proxy` |
| **Volume** | `obv`, `large_trade_cvd` |
| **Multi-Timeframe** | `trend_15m`, `trend_1h`, `rsi_1h` |
| **Cross-Asset** | `eth_corr_1h`, `eth_return_15m`, `btc_eth_spread`, `dxy_corr_1h`, `spx_corr_1h` |

---

## ⚙️ Training Configuration

| Parameter | Value | Reason |
|---|---|---|
| **Candles** | 400,000 (≈11 years) | Maximum historical depth for robust pattern learning |
| **Timeframe** | 15m | Best signal-to-noise ratio for swing trading |
| **Context Window** | 120 candles | 30-hour lookback — captures full market sessions |
| **Forecast Steps** | 15 candles | Predicts next 3h 45m of price trajectory |
| **Batch Size** | 32 (Sniper) | Maximum precision, catches micro-patterns |
| **Epochs** | 300 | Full deep maturation cycle |
| **Optimizer** | AdamW (lr=1e-5→1e-6, cosine decay) | Stable MoE training with weight decay |
| **EarlyStopping** | Patience=20 | Safe for CPU training (~28 days max wait) |

### 🎯 Batch Size Guide

| Batch | Persona | Profile | Use When |
|---|---|---|---|
| **32** | 🎯 **Sniper** | Peak precision, catches micro-patterns | You want maximum alpha per trade |
| **64** | ⚖️ **Strategist** | Institutional balance, industry standard | Default reliable choice |
| **128** | 🚀 **Tank** | Fast training, stable gradients | Rapid testing or large datasets |

---

## 🚀 Quick Start

### 1. Start Training (Fresh)
```bash
nohup sudo /root/miniconda3/bin/python -u train.py \
  --candles 400000 \
  --batch 32 \
  > logs/iron_oracle_v11.log 2>&1 &
```

### 2. Launch Dashboard
```bash
nohup sudo /root/miniconda3/bin/python -u src/api/serve.py \
  > logs/dashboard.log 2>&1 &
```

### 3. Monitor Training
```bash
tail -f logs/iron_oracle_v11.log
```

### 4. Open Dashboard
```
http://<your-server-ip>:5000
```

### 5. Resume Training (After Crash/Reboot)
```bash
nohup sudo /root/miniconda3/bin/python -u train.py \
  --candles 400000 \
  --batch 32 \
  --resume \
  > logs/iron_oracle_v11.log 2>&1 &
```

---

## 📡 Dashboard Features

| Panel | Description |
|---|---|
| **AI Brain Growth** | Current epoch / total epochs with progress bar |
| **Win Chance** | Validated directional accuracy (%) |
| **Confusion Level** | Validation loss (lower = better) |
| **Conviction Strength** | Average MoE expert certainty score |
| **Equity Chart** | Compounding portfolio growth from simulation trades |
| **AI Strikes Feed** | Live trade log with strategist identity, side, entry, P&L |
| **AI Memory History** | Per-epoch validation accuracy, loss, certainty table |
| **Hall of Fame** | Top 3 performing strategists by cumulative P&L |
| **Fee Burn Meter** | Total fees paid across all simulation trades |
| **Expert Dialogue** | Real-time neural council debate in plain English |

---

## 📈 Performance Benchmarking

After each training run, use these scripts:

```bash
# 1. Multi-Certainty Timeline Audit (Today's signal & consensus distribution)
sudo /root/miniconda3/bin/python scripts/certainty_audit.py

# 2. Walk-Forward Fine-Tuning Simulator (Live simulation with daily fine-tuning)
sudo /root/miniconda3/bin/python scripts/wfa_backtest.py

# 3. Walk-Forward Directional Backtest (1,035 evaluations of trend accuracy & Net P&L)
sudo /root/miniconda3/bin/python src/evaluation/backtest_checkup.py

# 4. Singularity Visualizer (Generates actual vs forecast plot at backtest_honesty.png)
sudo /root/miniconda3/bin/python src/evaluation/visualize_backtest.py
```

> **All scripts automatically use the latest epoch checkpoint** — no manual configuration needed.

### 📊 Reading the Benchmark Output

```
Certainty Threshold    Signals   Coverage   Accuracy
ALL (raw)              1000       100%        49-52%    ← All signals
>= 60%                  680        68%        61%+      ← Filtered (GOOD)
>= 80%                  130        13%        65%+      ← High conviction (DEPLOY)
```

- **Raw accuracy of 52%+** = the model has a genuine edge
- **At 80% certainty, accuracy 60%+** = safe to deploy in live trading
- **Profit Factor > 2.0** = professional grade strategy

---

## ⚙️ Configuration (`src/config/sovereign_config.py`)

```python
FEE_RATE          = 0.0012   # 0.12% per trade (Delta Exchange India: taker + GST)
INITIAL_WALLET_USD = 200.0   # Starting simulation capital
POSITION_SIZE_PCT  = 1.0     # 100% of wallet per trade (full port)
PROFIT_GOAL_PCT    = 0.1     # 10% profit target for dashboard UI
RISK_MULTIPLIER    = 100.0   # Drawdown penalty for station health display
CONTEXT_WINDOW     = 120     # Candles of historical context
FORECAST_STEPS     = 15      # Steps to predict ahead
```

---

## 🛡️ Trade Signal Logic

| Signal | Condition | Meaning |
|---|---|---|
| **LONG** | `prediction_delta > 0` and `abs(delta) >= 0.001` | AI expects price to rise |
| **SHORT** | `prediction_delta < 0` and `abs(delta) >= 0.001` | AI expects price to fall |
| **HOLD** | `abs(prediction_delta) < 0.001` | Move too small — deadzone filter active |

The **Deadzone Filter** (0.1% minimum predicted move) eliminates noise-driven trades that would be eaten by fees. This is the primary selectivity mechanism.

### 🛡️ Server-Side Bracket Sync (Sovereign Risk Sync)
To protect active trade profits and guarantee absolute operational safety from script or network failures:
* **Direct Server Enforcement**: The system dynamically synchronizes all stop-loss and take-profit brackets directly with **Delta Exchange India servers** so they display on the exchange UI and trigger even if the local Python server crashes.
* **Query-Preserve-Delete-Recreate Lifecycle**: Delta Exchange throws a `bracket_order_exists` (400) error if you try to overwrite an active bracket. The exchange client resolves this by querying active exchange orders, extracting and preserving your original Take Profit targets, cancelling the old brackets using correct query-less DELETE signatures, and posting the updated Trailing SL and original TP.
* **Risk Triggers**:
  * **Breakeven Lock (`BREAKEVEN_TRIGGER_PCT = 0.8%`)**: Moves Stop Loss directly to your entry price on the exchange UI to lock in a zero-risk trade.
  * **Trailing Stop Loss (`TRAILING_STOP_PCT = 0.5%`)**: Continually trails your stop loss `0.5%` behind the highest peak price reached (updates rate-limited to movements of $\ge \$50$ USD to protect against API blocks).


---

## 🔧 Known Architecture Decisions

| Decision | Rationale |
|---|---|
| CPU-only training | 24GB RAM server, no GPU available |
| Streaming dataset | Prevents OOM — no full materialization |
| Checkpoint pruner (keep 3) | Prevents disk fill during 300-epoch runs |
| Local JS/CSS assets | Offline-capable — no CDN dependency |
| Deterministic strategists | MD5-seeded names — same trade = same expert always |
| `--resume` flag | Parses epoch number from checkpoint filename for accurate resumption |

---

## 📋 Changelog

| **V12.4** | Preprocessor KeyError: 'obi_l5' fix, certainty audit alignment, test_today.py/backtest_checkup.py delta logic correction, visualization denormalization and price compounding fixes. |
| **V12.3** | Strict Dynamic Local Scaling (DLS) synchronization, Delta API exponential backoff, checkpoint IO optimization, L2 Order Book collection logging. |
| **V12.2** | Dynamic ATR-based MIN_SWING (clamped at $100 floor for fee protection), exponential backoff on exchange 502s, optimized training checkpoints (save every 10 epochs to fix I/O stall). |
| **V12.1** | Delta Exchange Server-Side Bracket Sync (DELETE query-less signature resolution, query-preserve-delete-recreate lifecycle, real-time UI SL/TP locking) |
| **V12.0** | Deterministic strategist names, fee-aware P&L, 3D branding, localised CDN, dashboard fixes |

| **V11.7** | Strategist identity system, expert dialogue engine |
| **V11.2** | Neural de-scaling patch, compounding wallet logic, Sharpe/Profit Factor metrics |
| **V11.0** | MissionControl callback, real-time dashboard updates every 50 batches |
| **V10.7** | RoPE positional encoding, Dropout(0.1), top-4 MoE routing, SovereignLoss volatility weighting |

---

## ⚠️ Important Notes

1. **Training is slow on CPU** — Each epoch takes ~13.5 hours at Batch 32. This is normal and by design (precision over speed).
2. **Do not interrupt training** — The model saves a checkpoint every 10 epochs. If interrupted, use `--resume` to continue from the last checkpoint.
3. **Dashboard port** — The server runs on **port 5000** by default.
4. **Data cache** — Historical candles are cached in `data/BTCUSD_15m_history_400000.parquet`. Delete this file to force a fresh fetch.
5. **Simulation vs Live** — All trade metrics shown on the dashboard are **simulated**. Use `auto_run.py` for live trading only after thorough backtesting.

---

⚓ **Iron Oracle V12.4 — Sovereign Intelligence, Institutional Performance.**