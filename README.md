# ⚓ Sovereign Kraken — Equity Swing Trading (Neural Network Only)

> **Self-contained Indian equity (NSE) swing trading system — HYDRA transformer + MoE, GPT-lineage architecture, real Zerodha delivery costs, daily candles, multi-day holds.**

---

## 🧠 What Is This?

A neural network (transformer + Mixture-of-Experts, the same architectural family as GPT/DeepSeek) trained to trade Indian equities on a swing-trading basis: it looks at 60 trading days of a stock's history and predicts whether a position entered tomorrow would be profitable within a 20-trading-day hold, using a 5% stop-loss / 10% target matching real, validated swing-trading parameters.

**This system was converted from an earlier crypto (BTC perpetual futures) version** — the model architecture, training discipline, and evaluation rigor carried over; the market, data, features, and execution infrastructure did not (equities have no order book microstructure, no funding rate, no 24/7 trading — none of that applies here). It was also merged with a separate, already-validated equity trading pipeline (broker integration, realistic cost model, backtest/paper/live engines) so this project is fully self-contained — no dependency on any other project.

**There are no rule-based technical-indicator strategies in this system.** An earlier version of the underlying pipeline had 3 validated rule-based strategies (Bollinger breakout, inside-bar breakout, Supertrend) — these were deliberately removed. The only trading logic here is the trained neural network.

---

## 📁 Project Structure

```
kat/
├── train.py                        ← Training orchestrator (run this to train)
├── auto_run.py                     ← Unified CLI: train / backtest / paper / live
├── pyproject.toml                  ← Project metadata + dependency list
├── tests/                          ← pytest suite
├── config/
│   └── settings.yaml               ← Broker/backtest/paper/live config (source of truth
│                                      for universe, swing params, costs — sovereign_config.py
│                                      reads from here, not a separate hardcoded copy)
├── .env                            ← Broker credentials (gitignored)
├── cache/
│   ├── historical/                 ← Cached daily candles per symbol (real Upstox data)
│   └── kaggle/                     ← Optional free CSV-based backtest data (see below)
├── models/                         ← Created by training
│   ├── hydra_best.keras            ← Best checkpoint (by val_prediction_dir_acc)
│   └── model_shape.pkl             ← Exact trained shape (context_window/forecast_steps/
│                                      n_features/symbols) — read by any inference code
├── logs/                           ← Training logs, edge_tracker.csv, paper/live trade logs
├── reports/swing/                  ← Backtest output (trade logs, strategy_comparison.csv)
├── scripts/
│   ├── equity_login_zerodha.py     ← Daily Zerodha token refresh (Kite Connect)
│   └── equity_login_upstox.py      ← Daily Upstox token refresh
└── src/
    ├── config/
    │   └── sovereign_config.py     ← ML training config (context window, fee approximation
    │                                  for labels, universe/costs read from config/settings.yaml)
    ├── core/
    │   └── hydra.py                ← Neural model architecture (HYDRA, single-input)
    ├── data/
    │   ├── preprocess.py           ← 29-feature equity pipeline, pooled multi-symbol
    │   │                              dataset builder, real swing-outcome labeling
    │   ├── fetch_historical.py     ← Cached daily-candle downloader (via configured broker)
    │   └── import_kaggle_data.py   ← Free CSV-based backtest data path (no broker needed)
    ├── exchange/
    │   ├── brokers/                ← Broker abstraction: base.py + zerodha/upstox/dhan/csv
    │   ├── time_utils.py           ← IST-aware "now"
    │   └── resample.py             ← OHLCV resampling helper
    ├── backtest/
    │   ├── delivery_costs.py       ← Real Zerodha delivery (CNC) cost model
    │   └── metrics.py              ← Sharpe, drawdown, win rate, profit factor
    ├── strategies/swing/
    │   ├── base.py                 ← Strategy interface (generate_signals contract)
    │   └── ml_strategy.py          ← Wraps the trained model to implement that interface
    ├── swing_backtest/             ← Backtest engine + CLI runner
    ├── swing_paper_trading/        ← Virtual portfolio + once-daily live-data engine
    ├── live_trading_swing/         ← Real order executor + risk manager (gated, real money)
    └── api/swing_dashboard/        ← FastAPI backend + static frontend dashboard (port 9000)
```

---

## 🏗️ Neural Architecture (HYDRA)

```
Market Input (60 trading days × 29 features)
        │
   [GaussianNoise(0.02)]
        │
   [Dense → 128]
        │
   [RMSNorm]
        │
   ┌────┴──── × 8 ──────────────┐
   │      HydraBlock             │
   │  ┌────────────────────┐    │
   │  │ QK-Norm + RoPE     │    │  ← Stabilized, time-aware attention
   │  │ Causal Softmax Attn│    │  ← Real GPT/DeepSeek-style attention
   │  │ TurboQuant         │    │  ← INT8-sim quantization stabilizer
   │  │ GatedMoE-32        │    │  ← 32 routed experts (top-4) + shared expert path
   │  │ SwiGLU (2x expand) │    │  ← Gated FFN, real hidden-dim expansion
   │  │ Dropout(0.15)      │    │  ← Prevents expert memorization
   │  └────────────────────┘    │
   └─────────────────────────────┘
        │
   RMSNorm (full sequence)
        │
   ┌────┼──────────┐
   │    │          │
[Prediction] [Certainty] [Reasoning]
 (21×3 traj)  (60 scores) (4 classes: LONG/SHORT/FEE_TRAP/NOISE)
```

**No GPT-style next-token head** — the crypto version of this project had one (predicting the next candle's return-bucket as an auxiliary task); it was dropped for this conversion. That task showed marginal value even on a larger single-asset dataset, and this project's pooled multi-symbol dataset (~33K windows across 14 usable symbols) is smaller — not worth taxing model capacity on a task that wasn't clearly paying for itself. This is also why the model takes a **single input** (just the feature window), not the dual market+token input the crypto version used.

### Key Components
| Component | Description |
|---|---|
| **Causal Softmax Attention + QK-Norm** | Real GPT/DeepSeek-style attention |
| **GatedMoE-32** | 32 routed expert sub-networks (top-4) + always-active shared-expert path (DeepSeekMoE-style) — sized for this dataset's scale, not billions-of-tokens scale |
| **TurboQuant** | INT8 simulation with orthogonal rotation |
| **SovereignLoss** | Volatility-weighted directional loss |
| **RMSNorm** | Fast root mean square normalization (LLaMA-style) |
| **SwiGLU (2x expansion)** | Gated activation, real hidden-dim expansion |
| **WarmupCosineDecay** | Linear LR warmup (peak 1e-4) before cosine decay |

---

## 📊 29 Input Features

All ratio-based or bounded — no raw price level fed directly without local normalization — since training pools data across 14 stocks with very different absolute price levels (₹400 ITC vs. ₹2,400 RELIANCE). Dynamic Local Scaling (`apply_dls`) additionally z-scores every window against its own local mean/std before it reaches the model.

| Category | Features |
|---|---|
| **Time** | `day_of_week`, `month_sin`, `month_cos` |
| **Price (raw, DLS-scaled)** | `open`, `high`, `low`, `close`, `volume` |
| **Returns** | `ret_1d`, `ret_5d`, `ret_20d` |
| **Trend / MA distance** | `sma20_dist`, `sma50_dist`, `sma200_dist`, `ema20_dist` |
| **Momentum** | `rsi`, `macd`, `macd_signal`, `macd_hist`, `adx` |
| **Volatility** | `atr_pct`, `bb_width`, `bb_position`, `volatility_20d` |
| **Volume** | `volume_ratio` |
| **Range position** | `hl_position_20`, `hl_position_252` (52-week) |
| **Trend structure** | `supertrend_dir`, `donchian_position` |

No order-book/microstructure features (funding rate, taker flow, order-book imbalance) — none of that exists for NSE cash equity; this is a genuinely different feature set from the crypto version, not a relabeled copy.

---

## 🎯 How Labels Are Built

For every historical window, the training pipeline asks: *if a position were entered the next trading day, would it hit its 10% target before its 5% stop, within 20 trading days?* — walking the real day-by-day high/low path, the **exact same check** the backtest/paper/live engines use for real trades. Four possible outcomes:

| Label | Meaning |
|---|---|
| **0 — SOVEREIGN_LONG** | Long would have cleanly won |
| **1 — SOVEREIGN_SHORT** | Short would have cleanly won |
| **2 — FEE_TRAP** | Both directions "won" (choppy/ambiguous), or a real move happened but never cleanly resolved |
| **3 — NOISE** | Nothing meaningful happened |

The model only fires a trade when it says LONG or SHORT *and* is confident enough (`CERT_THRESHOLD`) *and* its separate price-trajectory prediction agrees on direction — FEE_TRAP and NOISE both mean sit out.

---

## ⚙️ Training Configuration

| Parameter | Value | Reason |
|---|---|---|
| **Universe** | 14 usable symbols (Nifty50 slice — see `config/settings.yaml`) | TMCV excluded automatically (only 35 rows since its 2024 ticker rename) |
| **Data** | ~2,475 daily candles/symbol, 2016–2025, pooled | ~33K total training windows |
| **Context Window** | 60 trading days (~3 months) | |
| **Forecast Steps** | 20 trading days | Matches `max_holding_days` |
| **Batch Size** | 32 | |
| **Epochs** | 300 (EarlyStopping patience=20) | Auto-stops and restores best weights if 20 rounds pass with no improvement |
| **Optimizer** | AdamW (lr=1e-4→1e-5 cosine decay, weight_decay=0.05, clipnorm=1.0) | |
| **Train/val split** | Chronological **by date**, across all pooled symbols | No symbol's future data can leak into another symbol's training window |
| **Epoch time (CPU, no GPU)** | ~3.5 hours | Larger pooled dataset + 20-step forecast horizon than the crypto version |

---

## 🚀 Quick Start

### 1. Setup
```bash
cd /var/www/html/ML/kat
pip3 install -r pyproject.toml   # or: pip install -e .
```
Pick a broker in `config/settings.yaml` (`broker.name: zerodha|upstox|dhan|csv`) and fill in that broker's section of `.env`. Zerodha/Upstox tokens expire **daily** — run the matching login script each trading morning before paper/live trading:
```bash
python3 scripts/equity_login_zerodha.py
python3 scripts/equity_login_upstox.py
```
No broker account yet? Set `broker.name: csv` and use free Kaggle Nifty50 data instead — see `src/data/import_kaggle_data.py`'s docstring.

### 2. Train
```bash
python auto_run.py train --epochs 300
# or directly:
python train.py --epochs 300 --batch 32
```
Saves `models/hydra_best.keras` + `models/model_shape.pkl`. Monitor with:
```bash
tail -f logs/hydra_train_*.log
tail -f logs/edge_tracker.csv   # per-epoch Wilson 95% CI significance verdict
```

### 3. Backtest
```bash
python auto_run.py backtest
python auto_run.py backtest --refresh   # force re-download historical data
```
Real Zerodha delivery costs, no lookahead (entries fill at next day's open), portfolio-level position limits. Requires a trained checkpoint — fails with a clear message if `models/hydra_best.keras` doesn't exist yet.

### 4. Paper Trade
```bash
python auto_run.py paper
```
Virtual orders only, real broker prices. Checks once per trading day (~9:20am IST). Needs a real broker (not `csv`) and a fresh daily login.

### 5. Live Trade — REAL MONEY
```bash
python auto_run.py live
```
Gated: `config/settings.yaml` → `live_trading.enabled` must be manually `true`, plus a typed confirmation phrase at startup, plus (by default) per-order y/n confirmation. **Run paper trading for weeks first.**

### 6. Dashboard
```bash
uvicorn api.swing_dashboard.main:app --host 0.0.0.0 --port 9000
```
Open `http://<server-ip>:9000`. Backtest and paper trading are fully controllable from here (safe). Live trading is **read-only status** here by design — start it from a terminal you're actively watching, not a one-click web button.

---

## 🛡️ Safety Design

- **Live trading confirmation gate**: `enabled: true` in config + typed confirmation phrase + per-order confirmation (all defaults on).
- **Risk manager kill switch**: cumulative realized loss ≥ `max_total_loss_pct` of initial capital halts all new entries.
- **Broker-side stop-loss (GTT/OCO)**: attempted after every entry so the stop/target lives on the broker's own servers, not just this process's memory — important since a swing position stays open for days to weeks. Currently only implemented for Zerodha; Upstox/Dhan fall back to this process polling prices once a day.
- **No lookahead**: a signal on day *i* only uses information available up to and including day *i*; the backtest engine fills at day *i+1*'s open.

---

## ⚠️ Honest Status (as of this conversion)

**Verified via real execution:**
- Data pipeline, model, and training loop (real training runs cleanly)
- Backtest engine + real cost model (produces correct, real results)
- API dashboard (actually starts, serves real responses)
- Broker connection code (constructs correct requests; error handling verified against a real 401)

**Not yet verified:**
- `ml_swing` strategy against an actual trained checkpoint (training has not yet completed a full epoch)
- Paper trading's real-time loop (needs a fresh, non-expired broker token)
- Live trading (deliberately not casually tested — code was carefully read, not executed, given real money is involved)

---

⚓ **Sovereign Kraken — Equity Swing, Neural Network Only.**
