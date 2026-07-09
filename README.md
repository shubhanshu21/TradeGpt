# ⚓ Iron Oracle V12.5 — Sovereign Kraken Intelligence
> **Institutional-Grade Autonomous Neural Trading Station | BTC/USDT 15m/1h | 256-Expert MoE + GPT-Style Next-Token Prediction | Fee-Aware Simulation Engine**


<p align="center">
  <strong>SOVEREIGN NEURAL COMMAND CENTER</strong>
</p>

---

## 🧠 What Is This?

**Iron Oracle** is a self-contained, autonomous AI trading engine built for **BTC/USDT perpetual futures**, with multi-timeframe support (15-minute and 1-hour, selectable via `--timeframe`). It trains a deep neural network — combining a GPT-style next-candle-token predictor with a fee-aware trade-viability classifier — on real historical market data and uses it to generate high-conviction trade signals, complete with institutional-grade risk metrics and a multi-path prediction dashboard.

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
│   │   └── hydra.py                ← Neural model architecture (HYDRA V12.5)
│   ├── data/
│   │   └── preprocess.py           ← 45-feature data pipeline + return-token vocabulary + streaming dataset
│   ├── api/
│   │   └── prediction_viewer/      ← Flask dashboard (port 5000): multi-path prediction chart w/ real uncertainty
│   ├── exchange/
│   │   ├── fetch_data.py           ← Market data fetcher (Delta Exchange)
│   │   └── delta_client.py         ← Delta Exchange REST/order API client
│   ├── trading/
│   │   ├── live_trader.py          ← Live trade executor
│   │   └── risk.py                 ← Risk management engine
│   └── evaluation/
│       ├── flash_backtest.py       ← Quick 100-step backtest evaluation
│       ├── backtest_checkup.py     ← Full 1,000-candle walk-forward backtest
│       ├── backtest_live_logic.py  ← Replays live_trader.py's exact gate logic against history
│       └── visualize_backtest.py   ← Denormalized close-price zoom graph generator
├── scripts/
│   ├── certainty_audit.py          ← Real-time certainty distribution timelines
│   ├── wfa_backtest.py             ← Walk-forward fine-tuning simulator
│   ├── test_today.py               ← Performance diagnostic for today's market slice
│   ├── ob_collector.py             ← Background orderbook L2 snapshot collector
│   ├── bridge_history.py           ← Delta+Binance history bridge (see note below — has a known truncation bug)
│   └── daily_finetune.py           ← Daily fine-tuning adapter (cron target)
├── models/                         ← Created by training; empty until the first run completes an epoch
│   ├── hydra_best.keras            ← Best checkpoint (by val_prediction_dir_acc), appears after epoch 1
│   └── return_vocab.pkl            ← Next-candle-token vocabulary (bin edges/centers), needed for inference
├── logs/
│   ├── iron_oracle_v11.log         ← Live training log
│   └── edge_tracker.csv            ← Per-epoch statistical significance verdict (Wilson 95% CI on dir_acc)
└── data/
    └── BTCUSD_{15m,1h}_history_master.parquet ← Cached historical candles, per timeframe
```

---

## 🏗️ Neural Architecture (HYDRA V12.5)

```
Market Input (CTX candles × 45 features)  +  Return-Token Input (CTX candle-return buckets)
        │                                            │
   [GaussianNoise(0.02)]                    [Embedding(vocab_size, 128)]
        │                                            │
   [Dense → 128] ───────────────────── + ────────────┘
        │
   [RMSNorm]
        │
   ┌────┴──── × 8 ──────────────┐
   │      HydraBlock V12.5       │
   │  ┌────────────────────┐    │
   │  │ QK-Norm + RoPE     │    │  ← Stabilized, time-aware attention
   │  │ Causal Softmax Attn│    │  ← Real GPT/DeepSeek-style attention (not linear approx)
   │  │ TurboQuant         │    │  ← INT8-sim quantization stabilizer
   │  │ GatedMoE-256       │    │  ← 256 routed experts (top-4) + shared expert path
   │  │ SwiGLU (2x expand) │    │  ← Gated FFN, real hidden-dim expansion
   │  │ Dropout(0.1)       │    │  ← Prevents expert memorization
   │  └────────────────────┘    │
   └─────────────────────────────┘
        │
   RMSNorm (full sequence)
        │
   ┌────┼──────────┬──────────────┐
   │    │          │              │
[Prediction] [Certainty] [Reasoning] [Next-Token] (every position — true causal LM head)
 (16×3 traj)  (256 scores) (4 classes)  (CTX × vocab_size softmax)
```

Inference also supports genuine autoregressive generation (`generate_future_tokens`/
`generate_with_confidence` in `hydra.py`): sample a next-candle token, feed it back in,
predict the next — with temperature/top-k/top-p sampling and multi-path agreement as a
real confidence signal (not just a separately-trained certainty head).

### Key Components:
| Component | Description |
|---|---|
| **Causal Softmax Attention + QK-Norm** | Real GPT/DeepSeek-style attention (replaced an earlier linear-attention approximation) |
| **Return-Token Vocabulary** | 128-bucket quantile-binned candle-to-candle returns — genuine discrete next-token prediction, not just continuous regression |
| **GatedMoE-256** | 256 routed expert sub-networks (top-4) + an always-active shared-expert path (DeepSeekMoE-style) |
| **TurboQuant** | INT8 simulation with orthogonal rotation — quantization-aware training |
| **SovereignLoss** | Volatility-weighted directional loss — penalizes errors on high-vol moves more |
| **RMSNorm** | Fast root mean square normalization (LLaMA-style) |
| **SwiGLU (2x expansion)** | Gated activation with real hidden-dim expansion (Gemma/LLaMA DNA) |
| **WarmupCosineDecay** | Linear LR warmup before cosine decay — MoE training stability |

---

## ⚡ V12.5 Patch Notes (GPT-Style Rewrite + Multi-Timeframe)

1. **Real causal softmax attention** — replaced the earlier ELU+1 linear-attention approximation with standard scaled dot-product attention (what GPT/DeepSeek actually use); added QK-Norm for stability.
2. **Genuine next-candle-token prediction** — 128-bucket quantile return vocabulary, token embeddings, and a true per-position causal-LM head (every one of the CTX context positions is its own training example, not just the last one).
3. **Autoregressive generation** — `generate_future_tokens`/`generate_with_confidence` in `hydra.py`: predict one token, feed it back in, predict the next, with temperature/top-k/top-p sampling. Multi-path sample agreement now gives a real confidence signal instead of relying solely on a disconnected certainty head.
4. **Shared-expert MoE** (DeepSeekMoE-style) and **2x SwiGLU expansion** — closed two real capacity gaps versus the GPT/DeepSeek lineage this architecture is modeled on.
5. **Rebalanced loss weights** (`prediction:3, certainty:1, reasoning:1`, was `1:10:5`) and **`WarmupCosineDecay`** LR schedule — direction prediction now actually dominates the training signal instead of being drowned out.
6. **`EdgeTracker` callback** — per-epoch Wilson 95% confidence interval on `val_prediction_dir_acc`, logged to `logs/edge_tracker.csv`, so "is there a real edge" is answered statistically instead of by eyeballing a noisy number.
7. **Path-aware trade labeling** — `preprocess.py` now checks that price actually reaches take-profit *before* stop-loss along the realized path, not just whether the final excursion ratio looks favorable in hindsight.
8. **Multi-timeframe support** — `--timeframe`/`--context_window`/`--forecast_steps` CLI overrides; feature engineering (`trend_1h`, `rsi_1h`, `trend_4h`, funding-rate proxy windows) is now timeframe-aware instead of hardcoding "4 candles = 1 hour," which was only true for 15m data.
9. **Direction-agreement gate in `live_trader.py`** — trades now only fire when the reasoning head and the price-trajectory head agree on direction.
10. **Removed unused `serve.py` dashboard**; replaced with `src/api/prediction_viewer/` (Flask, port 5000) — shows real multi-path predictions with sampled-path uncertainty against actual outcomes.

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
| **Candles** | 120,000 (≈3.4 years) | Historical depth actually available in the current data cache |
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

### 1. Start Training (Fresh) — 15-minute (default)
```bash
nohup sudo /root/miniconda3/bin/python -u train.py \
  --candles 120000 \
  --batch 32 \
  > logs/iron_oracle_v11.log 2>&1 &
```

### 1b. Start Training on a Different Timeframe (e.g. 1-hour)
Requires a pre-built `data/BTCUSD_1h_history_master.parquet` (resample from the 15m
cache — see `data/` note below). `--context_window`/`--forecast_steps` let you keep
the same real-world lookback/forecast time spans across timeframes instead of
accidentally changing both the timeframe *and* the horizon at once:
```bash
nohup sudo /root/miniconda3/bin/python -u train.py \
  --timeframe 1h --candles 30000 \
  --context_window 30 --forecast_steps 4 \
  --batch 32 \
  > logs/iron_oracle_v11.log 2>&1 &
```

### 2. Launch Prediction Viewer Dashboard
```bash
nohup sudo /root/miniconda3/bin/python src/api/prediction_viewer/app.py &
```

### 3. Monitor Training
```bash
tail -f logs/iron_oracle_v11.log
tail -f logs/edge_tracker.csv   # per-epoch statistical significance verdict
```

### 4. Open Dashboard
```
http://<your-server-ip>:5000
```

### 5. Resume Training (After Crash/Reboot)
```bash
nohup sudo /root/miniconda3/bin/python -u train.py \
  --candles 120000 \
  --batch 32 \
  --resume \
  > logs/iron_oracle_v11.log 2>&1 &
```

---

## 📡 Dashboard Features (Prediction Viewer, `src/api/prediction_viewer/`)

The old `serve.py` dashboard (equity chart, strikes feed, hall of fame, etc.) has been
removed — it was unused. The current dashboard is a focused prediction-inspection tool:

| Feature | Description |
|---|---|
| **Historical Candlestick** | Real Delta Exchange history leading into the prediction point |
| **Multi-Path Prediction** | Several independently-sampled future paths (temperature/top-p sampling), shown as thin lines, plus a bold mean prediction |
| **Actual Outcome Overlay** | Real candlesticks for the holdout period, if available, for direct visual comparison |
| **Confidence Readout** | Majority direction + % agreement across sampled paths — genuine confidence derived from the prediction itself, not a separate disconnected score |
| **Time Window Slider / Temperature / Top-p / Sample Count** | Adjustable inference parameters, applied live |

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

| **V12.5** | Real causal softmax attention (replaced linear-attention approx), QK-Norm, shared-expert MoE, 2x SwiGLU expansion, LR warmup, genuine GPT-style next-candle-token prediction w/ autoregressive generation, EdgeTracker statistical significance monitoring, multi-timeframe support (15m/1h), removed unused serve.py dashboard (replaced with prediction_viewer, port 5000). |
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
3. **Dashboard port** — `src/api/prediction_viewer/app.py` (multi-path candle prediction with real sampled-path uncertainty) runs on **port 5000**. This replaced the old, unused `serve.py` dashboard, which has been removed. Requires a trained checkpoint at `models/hydra_best.keras` and `models/return_vocab.pkl` to load real predictions.
4. **Data cache** — Historical candles are cached in `data/BTCUSD_15m_history_master.parquet` (currently 120,000 candles, ~3.4 years). Delete this file to force a fresh fetch.
5. **Simulation vs Live** — All trade metrics shown on the dashboard are **simulated**. Use `auto_run.py` for live trading only after thorough backtesting.

---

⚓ **Iron Oracle V12.5 — Sovereign Intelligence, Institutional Performance.**