# ⚓ Iron Oracle V12.6 — Sovereign Kraken Intelligence
> **Institutional-Grade Autonomous Neural Trading Station | BTC/USDT 1h | 32-Expert MoE + GPT-Style Next-Token Prediction | Fee-Aware Simulation Engine**


<p align="center">
  <strong>SOVEREIGN NEURAL COMMAND CENTER</strong>
</p>

---

## 🧠 What Is This?

**Iron Oracle** is a self-contained, autonomous AI trading engine built for **BTC/USDT perpetual futures** on Delta Exchange India, currently trained exclusively on the **1-hour** timeframe (multi-timeframe support exists via `--timeframe`, but 1h is the actively maintained/tested configuration). It trains a deep neural network — combining a GPT-style next-candle-token predictor with a fee-aware trade-viability classifier — on real historical market data (including genuine Binance order-book depth, not synthetic fill) and uses it to generate high-conviction trade signals, complete with institutional-grade risk metrics and a multi-path prediction dashboard.

The system is designed for a **CPU server** and runs fully autonomously — data fetching, training, simulation, and dashboard serving are all handled automatically.

---

## 📁 Project Structure

```
kat/
├── train.py                        ← Main training orchestrator (run this)
├── auto_run.py                     ← CLI wrapper: train / trade modes
├── pyproject.toml                  ← Project metadata + curated real dependency list
├── tests/                          ← pytest suite (preprocessing + model shape/sanity tests)
├── src/
│   ├── config/
│   │   └── sovereign_config.py     ← Central config (fees, wallet, risk, architecture defaults)
│   ├── core/
│   │   ├── hydra.py                ← Neural model architecture (HYDRA V12.5)
│   │   └── inference.py            ← Shared model-loading/inference helper (real trained shape from checkpoint)
│   ├── data/
│   │   └── preprocess.py           ← 45-feature data pipeline + return-token vocabulary + streaming dataset
│   ├── api/
│   │   └── prediction_viewer/      ← Flask dashboard (port 5000): multi-path prediction chart w/ real uncertainty
│   ├── exchange/
│   │   ├── fetch_data.py           ← Market data fetcher (Delta Exchange)
│   │   └── delta_client.py         ← Delta Exchange REST/order API client
│   ├── trading/
│   │   └── live_trader.py          ← Live trade executor
│   └── evaluation/
│       ├── backtest_checkup.py     ← Walk-forward backtest w/ naive-baseline comparison
│       ├── backtest_live_logic.py  ← Replays live_trader.py's exact gate logic against history
│       ├── wfa_backtest.py         ← Walk-forward fine-tuning simulator, regime-segmented + Wilson CI
│       ├── certainty_audit.py      ← Real-time certainty distribution timelines
│       ├── check_cert_threshold.py ← Data-driven CERT_THRESHOLD calibration
│       └── visualize_backtest.py   ← Denormalized close-price zoom graph generator
├── scripts/
│   ├── ob_collector.py             ← Background L2 snapshot collector — permanent compact real-data log
│   ├── fetch_book_depth_history.py ← Real historical order-book backfill (Binance bookDepth, free)
│   ├── merge_book_depth_into_master.py ← Merges real order-book data into the training parquet
│   ├── build_timeframe_master.py   ← Builds/refreshes a candle master parquet for a given timeframe
│   ├── bridge_history.py           ← Delta+Binance history bridge (has a known truncation bug — see Notes)
│   ├── daily_finetune.py           ← Daily fine-tuning adapter (cron target, currently disabled)
│   ├── fetch_l2_history.py         ← Superseded by fetch_book_depth_history.py — kept for reference, unused
│   └── test_trade.py               ← Testnet order-placement smoke test
├── models/                         ← Created by training; empty until the first run completes an epoch
│   ├── hydra_best.keras            ← Best checkpoint (by val_prediction_dir_acc), appears after epoch 1
│   └── return_vocab.pkl            ← Next-candle-token vocabulary + real trained shape, needed for inference
├── logs/
│   ├── hydra_train_<timestamp>.log ← Live training log (new file per run)
│   └── edge_tracker.csv            ← Per-epoch statistical significance verdict (Wilson 95% CI on dir_acc)
└── data/
    ├── BTCUSD_1h_history_master.parquet    ← Cached historical candles (primary, actively used)
    ├── BTCUSD_1h_orderbook_depth.csv       ← Real Binance order-book depth, merged into the master parquet
    └── orderbook_l5_history.csv            ← Permanent live-collected real L5 snapshots (ob_collector.py)
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
   │  │ GatedMoE-32        │    │  ← 32 routed experts (top-4) + shared expert path
   │  │ SwiGLU (2x expand) │    │  ← Gated FFN, real hidden-dim expansion
   │  │ Dropout(0.15)      │    │  ← Prevents expert memorization
   │  └────────────────────┘    │
   └─────────────────────────────┘
        │
   RMSNorm (full sequence)
        │
   ┌────┼──────────┬──────────────┐
   │    │          │              │
[Prediction] [Certainty] [Reasoning] [Next-Token] (every position — true causal LM head)
 (5×3 traj)  (CTX scores) (4 classes)  (CTX × vocab_size softmax)
```

Inference also supports genuine autoregressive generation (`generate_future_tokens`/
`generate_with_confidence` in `hydra.py`): sample a next-candle token, feed it back in,
predict the next — with temperature/top-k/top-p sampling and multi-path agreement as a
real confidence signal (not just a separately-trained certainty head).

### Key Components:
| Component | Description |
|---|---|
| **Causal Softmax Attention + QK-Norm** | Real GPT/DeepSeek-style attention (replaced an earlier linear-attention approximation) |
| **Return-Token Vocabulary** | 32-bucket quantile-binned candle-to-candle returns — genuine discrete next-token prediction, sized so the task is actually learnable (128 buckets left the auxiliary task barely above random chance) |
| **GatedMoE-32** | 32 routed expert sub-networks (top-4) + an always-active shared-expert path (DeepSeekMoE-style) — sized for ~25K training windows, not billions-of-tokens scale |
| **TurboQuant** | INT8 simulation with orthogonal rotation — quantization-aware training |
| **SovereignLoss** | Volatility-weighted directional loss — penalizes errors on high-vol moves more |
| **RMSNorm** | Fast root mean square normalization (LLaMA-style) |
| **SwiGLU (2x expansion)** | Gated activation with real hidden-dim expansion (Gemma/LLaMA DNA) |
| **WarmupCosineDecay** | Linear LR warmup before cosine decay (peak 1e-4) — MoE training stability |

---

## ⚡ V12.6 Patch Notes (Real Data + Training-Dynamics Fixes)

1. **Real order-book data** — replaced synthetically-fabricated `bid_vol1-5`/`ask_vol1-5` history with genuine Binance `bookDepth` archive data (free, ~600MB, 2023-01-01 to present), merged into the training parquet. The order-book-imbalance feature (`obi_l5`/`squeeze_pressure`) was trained on fabricated noise the entire time before this fix. `ob_collector.py` also fixed to permanently log real L5 snapshots going forward instead of auto-purging them after 48 hours.
2. **Learning rate raised 20x** (`5e-6 → 1e-4`) — the old value was 20-100x below typical practice for training a transformer this size from scratch, producing noise-dominated, barely-moving validation accuracy across many epochs.
3. **Regularization increased** (dropout `0.1 → 0.15`, weight decay `0.01 → 0.05`) — the LR fix surfaced a real train/validation gap (practice-score climbing to 70%+ while real-score stayed flat); this closed most of that gap.
4. **Return-token vocabulary shrunk 128 → 32**, and **MoE experts shrunk 256 → 32** — both were sized for a much larger dataset than the ~25K training windows actually available; concentrating capacity into fewer buckets/experts gives each one enough real examples to learn something instead of spreading gradient signal thin. MoE entropy load-balancing coefficient rescaled to match (it's proportional to `log(n_experts)`, so the old coefficient was ~60% too strong for the new expert count).
5. **Fixed a severe live-trading bug** — `live_trader.py` still called the model with the old single-input signature and had stale 15m/120-candle settings from before this architecture's dual-input rewrite; would have crashed on the first real inference call. Migrated to `core/inference.py`'s shared helper (same fix already applied to every evaluation script).
6. **Fixed a fee double-counting bug** across `wfa_backtest.py`, `backtest_checkup.py`, and `backtest_live_logic.py` — `FEE_RATE` already represents the full round-trip cost (0.12% ≈ real Delta India taker fee + 18% GST, doubled for entry+exit), but five call sites doubled it again, making every backtest assume trades needed to clear 2x the real fee cost.
7. **Backtest rigor**: `wfa_backtest.py` now reports a naive-baseline (persistence) comparison and a volatility-regime breakdown alongside a Wilson 95% CI on the win rate, instead of one raw aggregate win-rate number.
8. **Fixed a duplicate feature**: `trend_15m` was mathematically identical to `trend_1h` once the base candle became 1 hour (both computed `close.pct_change(1)`); replaced with a genuinely distinct `trend_1d`, completing a real short/medium/long momentum ladder.
9. **`daily_finetune.py` and `auto_run.py` cleaned up** — removed dead architecture modes and hardcoded stale-timeframe fallbacks that would have silently launched a mismatched run if ever triggered.
10. **Removed dead code**: `src/trading/risk.py` (never imported), `src/evaluation/flash_backtest.py` (redundant), several one-off `scratch/` debug scripts, and a stale `requirements.txt` (was a raw dump of an unrelated shared environment, not this project's real dependencies — `pyproject.toml` now has the curated, verified list).

---

## ⚡ V12.5 Patch Notes (GPT-Style Rewrite + Multi-Timeframe)

1. **Real causal softmax attention** — replaced the earlier ELU+1 linear-attention approximation with standard scaled dot-product attention (what GPT/DeepSeek actually use); added QK-Norm for stability.
2. **Genuine next-candle-token prediction** — quantile return vocabulary, token embeddings, and a true per-position causal-LM head (every one of the CTX context positions is its own training example, not just the last one).
3. **Autoregressive generation** — `generate_future_tokens`/`generate_with_confidence` in `hydra.py`: predict one token, feed it back in, predict the next, with temperature/top-k/top-p sampling. Multi-path sample agreement now gives a real confidence signal instead of relying solely on a disconnected certainty head.
4. **Shared-expert MoE** (DeepSeekMoE-style) and **2x SwiGLU expansion** — closed two real capacity gaps versus the GPT/DeepSeek lineage this architecture is modeled on.
5. **Rebalanced loss weights** (`prediction:6, certainty:1, reasoning:1, next_token:2`) and **`WarmupCosineDecay`** LR schedule — direction prediction now actually dominates the training signal instead of being drowned out.
6. **`EdgeTracker` callback** — per-epoch Wilson 95% confidence interval on `val_prediction_dir_acc`, logged to `logs/edge_tracker.csv`, so "is there a real edge" is answered statistically instead of by eyeballing a noisy number.
7. **Path-aware trade labeling** — `preprocess.py` now checks that price actually reaches take-profit *before* stop-loss along the realized path, not just whether the final excursion ratio looks favorable in hindsight.
8. **Multi-timeframe support** — `--timeframe`/`--context_window`/`--forecast_steps` CLI overrides; feature engineering (`trend_1h`, `rsi_1h`, `trend_4h`, funding-rate proxy windows) is timeframe-aware instead of hardcoding "4 candles = 1 hour."
9. **Direction-agreement gate in `live_trader.py`** — trades now only fire when the reasoning head and the price-trajectory head agree on direction.
10. **Removed unused `serve.py` dashboard**; replaced with `src/api/prediction_viewer/` (Flask, port 5000) — shows real multi-path predictions with sampled-path uncertainty against actual outcomes.

<details>
<summary>Older patch notes (V12.4 and earlier)</summary>

### V12.4 — Diagnostic & Logic Fixes
1. **L5 Order Book Imbalance (OBI) Fix**: Resolved a `KeyError: 'obi_l5'` in `preprocess.py`.
2. **Evaluation/Audit Certainty Alignment**: Standard 0-100% scale, dynamic `CERT_THRESHOLD` check.
3. **Logic Corrections in Diagnostics**: Fixed price trend delta calculations by subtracting anchor entry price.
4. **Trajectory Mean Evaluation Alignment**: Evaluate realized move using the mean of the future trajectory.
5. **Denormalization Fix in Visualizer**: `(pred_scaled_ret * std) + mean` instead of `entry_p + (pred_scaled_ret * std)`.
6. **Compounding Visualizer Bug Fix**: Prevented cumulative compounding of `usd_deltas` on the plot.
7. **Walk-Forward Simulator**: Added `wfa_backtest.py`.

### V12.3 — Critical Architecture Fixes
1. **Strict Dynamic Local Scaling (DLS)** synced precisely with the training pipeline.
2. **Delta API Exponential Backoff** for `502`/timeout errors.
3. **Optimized I/O Checkpointing** — save every 10 epochs instead of every 1.
4. **Order Book Data Collector** logging fix.

</details>

---

## 📊 45 Input Features

| Category | Features |
|---|---|
| **Time** | `hour_sin`, `hour_cos`, `day_of_week` |
| **Price OHLCV** | `open`, `high`, `low`, `close`, `volume`, `quote_volume` |
| **Microstructure** | `funding_rate_proxy`, `taker_buy_volume`, `taker_buy_quote_volume`, `cvd`, `taker_ratio`, `net_taker_flow`, `taker_pressure_20` |
| **Momentum** | `rsi`, `macd`, `macd_signal`, `macd_hist`, `macd_hist_slope`, `adx`, `cci`, `stoch_rsi` |
| **Trend/MA** | `sma_7`, `sma_25`, `sma_99`, `bollinger_upper`, `bollinger_lower`, `bb_width`, `vwap_dist` |
| **Volatility** | `atr`, `atr_ratio`, `volatility`, `vol_regime`, `squeeze_pressure` (real order-book imbalance × momentum), `liq_proxy` |
| **Volume** | `obv`, `large_trade_cvd`, `volume_ratio`, `hl_position_20` |
| **Multi-Timeframe** | `trend_1h`, `trend_4h`, `trend_1d`, `rsi_1h` |

> **Note:** an earlier version of this file's docstring referenced planned "Cross-Asset (ETH)" features (`eth_corr_1h`, `dxy_corr_1h`, etc.) — these were never actually implemented and have been removed from this table to match reality. Real cross-asset correlation remains a legitimate, unbuilt future improvement.

---

## ⚙️ Training Configuration

| Parameter | Value | Reason |
|---|---|---|
| **Candles** | 31,430 (≈3.6 years) | Full historical depth available in the current 1h data cache |
| **Timeframe** | 1h | Only timeframe with statistically significant edge found so far (15m showed none) |
| **Context Window** | 30 candles | 30-hour lookback |
| **Forecast Steps** | 4 candles | Predicts next 4 hours of price trajectory |
| **Batch Size** | 32 | Peak precision, catches micro-patterns |
| **Epochs** | 300 (EarlyStopping patience=20) | Full deep maturation cycle, auto-stops and restores best weights if 20 rounds pass with no improvement |
| **Optimizer** | AdamW (lr=1e-4→1e-5, cosine decay, weight_decay=0.05) | See V12.6 patch notes for why these values changed from earlier releases |
| **Epoch time (CPU, no GPU)** | ~1h40m at Batch 32 | Varies with host load |

---

## 🚀 Quick Start

### 1. Start Training (Fresh)
```bash
nohup sudo /root/miniconda3/bin/python -u train.py \
  --timeframe 1h --candles 31430 \
  --context_window 30 --forecast_steps 4 \
  --batch 32 \
  > logs/hydra_train_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

### 2. Launch Prediction Viewer Dashboard
```bash
nohup sudo /root/miniconda3/bin/python src/api/prediction_viewer/app.py &
```

### 3. Monitor Training
```bash
tail -f logs/hydra_train_*.log         # most recent run's live log
tail -f logs/edge_tracker.csv          # per-epoch statistical significance verdict
```

### 4. Open Dashboard
```
http://<your-server-ip>:5000
```

### 5. Resume Training (After Crash/Reboot)
```bash
nohup sudo /root/miniconda3/bin/python -u train.py \
  --timeframe 1h --candles 31430 \
  --context_window 30 --forecast_steps 4 \
  --batch 32 --resume \
  > logs/hydra_train_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

---

## 📡 Dashboard Features (Prediction Viewer, `src/api/prediction_viewer/`)

| Feature | Description |
|---|---|
| **Historical Candlestick** | Real Delta Exchange history leading into the prediction point |
| **Multi-Path Prediction** | Several independently-sampled future paths (temperature/top-p sampling), shown as thin lines, plus a bold mean prediction |
| **Actual Outcome Overlay** | Real candlesticks for the holdout period, if available, for direct visual comparison |
| **Confidence Readout** | Majority direction + % agreement across sampled paths — genuine confidence derived from the prediction itself, not a separate disconnected score |
| **Time Window Slider / Temperature / Top-p / Sample Count** | Adjustable inference parameters, applied live |

---

## 📈 Performance Benchmarking

After each training run, use these scripts (all read the real trained shape from `models/return_vocab.pkl` — no manual configuration needed):

```bash
# 1. Certainty distribution audit (today's signal & consensus)
sudo /root/miniconda3/bin/python src/evaluation/certainty_audit.py

# 2. Data-driven CERT_THRESHOLD calibration
sudo /root/miniconda3/bin/python src/evaluation/check_cert_threshold.py

# 3. Walk-forward fine-tuning simulator — naive-baseline comparison + regime breakdown + Wilson CI
sudo /root/miniconda3/bin/python src/evaluation/wfa_backtest.py

# 4. Walk-forward directional backtest w/ naive-baseline comparison
sudo /root/miniconda3/bin/python src/evaluation/backtest_checkup.py

# 5. Backtest w/ live_trader.py's exact gate logic replayed against history
sudo /root/miniconda3/bin/python src/evaluation/backtest_live_logic.py

# 6. Visualizer (actual vs. forecast plot at backtest_honesty.png)
sudo /root/miniconda3/bin/python src/evaluation/visualize_backtest.py
```

### 📊 Reading the Benchmark Output — an honest calibration note

Published, peer-reviewed research on short-horizon crypto direction prediction has not
demonstrated robust, fee-surviving superiority over a naive baseline — the most-cited
studies land around 51-52% direction accuracy. Treat that as the realistic ceiling, not
60%+. What actually matters more than the headline accuracy number is: (a) whether the
Wilson 95% CI stays above 50% consistently, not just occasionally, and (b) whether the
certainty-gated subset of trades beats the naive baseline on the *same* trades it chose
— that's what `wfa_backtest.py`'s naive-baseline comparison is for.

---

## ⚙️ Configuration (`src/config/sovereign_config.py`)

```python
FEE_RATE           = 0.0012  # 0.12% round-trip (Delta India: 0.05% taker + 18% GST, x2 for entry+exit)
INITIAL_WALLET_USD = 200.0   # Starting simulation capital
POSITION_SIZE_PCT  = 0.25    # Fraction of available wallet USD used as margin per trade
LEVERAGE           = 10      # Order leverage
CONTEXT_WINDOW     = 30      # Candles of historical context (train.py default when --context_window unset)
FORECAST_STEPS     = 4       # Steps to predict ahead (train.py default when --forecast_steps unset)
CERT_THRESHOLD     = 0.85    # Not yet recalibrated against a mature checkpoint — see check_cert_threshold.py
```

---

## 🛡️ Trade Signal Logic (`live_trader.py`)

Five sequential gates, all must pass before a trade fires:

| Gate | Condition |
|---|---|
| **1. Certainty** | Model's certainty output must exceed `CERT_THRESHOLD` |
| **2. Reasoning (Fee Awareness)** | Reasoning head must classify as LONG or SHORT, not FEE_TRAP/NOISE |
| **2b. Direction Agreement** | The reasoning head and the price-trajectory head must agree on direction |
| **3. Swing Size (Fee Protection)** | Predicted move must exceed a dynamic ATR-based minimum, or fees would eat the profit |
| **4. Cooldown** | No new entry within `forecast_steps` candles of the last trade |
| **5. ADX Regime Filter** | Market must show real trend strength (ADX ≥ 20), not choppy/ranging |

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
| CPU-only training | No GPU available on the current host |
| Streaming dataset | Prevents OOM — no full materialization |
| Checkpoint pruner (keep 3) | Prevents disk fill during 300-epoch runs |
| Local JS/CSS assets | Offline-capable — no CDN dependency |
| `--resume` flag | Parses epoch number from checkpoint filename for accurate resumption |
| Manual `sys.path.insert` imports, not an installed package | The top-level `data/` directory (raw parquet cache) and `src/data/` (the preprocessing package) share a name — an installed package's default import resolution would let the cache directory shadow the real package. The explicit path insert is the correct fix here, not technical debt. |
| Last 10% of training windows held out entirely | Reserved so `wfa_backtest.py`-style evaluation never tests on data the model trained on — currently coordinated only by convention (not enforced in code), a known minor gap |

---

## 📋 Changelog

| **V12.6** | Real Binance order-book data (replacing fabricated fill), LR 5e-6→1e-4, dropout 0.1→0.15, weight_decay 0.01→0.05, vocab 128→32, experts 256→32 w/ rescaled entropy coefficient, fixed a severe live_trader.py dual-input crash bug, fixed a fee double-counting bug across 3 evaluation scripts, backtest naive-baseline + regime-breakdown reporting, fixed a duplicate trend_15m/trend_1h feature, cleaned up daily_finetune.py/auto_run.py stale hardcoding, added tests/ + pyproject.toml, removed dead code. |
| **V12.5** | Real causal softmax attention (replaced linear-attention approx), QK-Norm, shared-expert MoE, 2x SwiGLU expansion, LR warmup, genuine GPT-style next-candle-token prediction w/ autoregressive generation, EdgeTracker statistical significance monitoring, multi-timeframe support, removed unused serve.py dashboard (replaced with prediction_viewer, port 5000). |
| **V12.4** | Preprocessor KeyError: 'obi_l5' fix, certainty audit alignment, backtest delta logic correction, visualization denormalization and price compounding fixes. |
| **V12.3** | Strict Dynamic Local Scaling (DLS) synchronization, Delta API exponential backoff, checkpoint IO optimization, L2 Order Book collection logging. |
| **V12.2** | Dynamic ATR-based MIN_SWING (clamped at $100 floor for fee protection), exponential backoff on exchange 502s, optimized training checkpoints. |
| **V12.1** | Delta Exchange Server-Side Bracket Sync (DELETE query-less signature resolution, query-preserve-delete-recreate lifecycle, real-time UI SL/TP locking) |
| **V12.0** | Deterministic strategist names, fee-aware P&L, localised CDN, dashboard fixes |
| **V11.7** | Strategist identity system, expert dialogue engine |
| **V11.2** | Neural de-scaling patch, compounding wallet logic, Sharpe/Profit Factor metrics |
| **V11.0** | MissionControl callback, real-time dashboard updates |
| **V10.7** | RoPE positional encoding, Dropout(0.1), top-4 MoE routing, SovereignLoss volatility weighting |

---

## ⚠️ Important Notes

1. **Training is slow on CPU** — Each epoch takes ~1h40m at Batch 32, 1h timeframe. The model saves a checkpoint whenever validation accuracy improves; if interrupted, use `--resume` to continue from the last checkpoint.
2. **Dashboard port** — `src/api/prediction_viewer/app.py` runs on **port 5000**. Requires a trained checkpoint at `models/hydra_best.keras` and `models/return_vocab.pkl` to load real predictions.
3. **Data cache** — Historical candles are cached in `data/BTCUSD_1h_history_master.parquet` (currently 31,430 candles, ~3.6 years), with real order-book depth merged in for ~97% of that range. Delete this file to force a fresh fetch (you'll also want to re-run `merge_book_depth_into_master.py` afterward to restore the real order-book data).
4. **Simulation vs Live** — All trade metrics shown on the dashboard are **simulated**. Use `auto_run.py trade` for live trading only after thorough backtesting, and note `CERT_THRESHOLD` has not yet been recalibrated against a mature checkpoint.
5. **`scripts/bridge_history.py`** has a known truncation bug (`combined.tail(120000)` caps history at 120K rows) — currently unfixed by deliberate choice, not an oversight.

---

⚓ **Iron Oracle V12.6 — Sovereign Intelligence, Institutional Performance.**
