# ⚓ Sovereign Kraken — Equity Swing Trading (Neural Network Only)

> **Self-contained Indian equity (NSE) swing trading system — HYDRA transformer + MoE, GPT-lineage architecture, real Zerodha delivery costs, daily candles, multi-day holds.**

---

## 🧠 What Is This?

A neural network (transformer + Mixture-of-Experts, the same architectural family as GPT/DeepSeek) trained to trade Indian equities on a swing-trading basis: it looks at 60 trading days of a stock's history and predicts whether a position entered tomorrow would be profitable within a 20-trading-day hold, using a 5% stop-loss / 10% target matching real, validated swing-trading parameters.

This is a fully self-contained system — data fetching, broker integration, a realistic cost model, and the backtest/paper/live engines all live in this one project, with no dependency on anything external.

**There are no rule-based technical-indicator strategies in this system.** The only trading logic here is the trained neural network — no Bollinger breakout, no inside-bar breakout, no Supertrend crossover rules.

**Every symbol gets its own independently-trained model** (`models/<SYMBOL>/hydra_best.keras`), not one model pooled/shared across stocks — different stocks have different volatility regimes and price behavior, and a per-stock model can specialize on that instead of averaging it away.

---

## 🔄 Pipeline

```
config/settings.yaml (universe, training.symbols, swing params, costs)
        │
        ▼
┌───────────────────┐        ┌──────────────────────────────┐
│  Broker adapter    │──────▶│  cache/historical/*.csv        │
│  (zerodha/upstox/  │ fetch  │  one CSV per symbol + NIFTY    │
│   dhan/csv)        │        │  (benchmark index, see below)  │
└───────────────────┘        └───────────────┬───────────────┘
                                              │ load_universe_from_cache()
                                              ▼
                              ┌───────────────────────────────┐
                              │  preprocess.py                 │
                              │  compute_indicators()          │
                              │  → 30 features/day, incl.      │
                              │    rel_strength_20d vs NIFTY   │
                              │  apply_dls() per 60-day window │
                              └───────────────┬───────────────┘
                                              │
                    ┌─────────────────────────┼─────────────────────────┐
                    ▼                         ▼                         ▼
            train.py loops one symbol at a time — one HYDRA model each
                    │                         │                         │
                    ▼                         ▼                         ▼
        models/HDFCBANK/           models/RELIANCE/           models/<...>/
        hydra_best.keras           hydra_best.keras           hydra_best.keras
        model_shape.pkl            model_shape.pkl            model_shape.pkl
                    │                         │                         │
                    └─────────────────────────┼─────────────────────────┘
                                              ▼
                          strategies/swing/ml_strategy.py
                          MLSwingStrategy — lazy-loads the RIGHT
                          per-symbol model when generate_signals(df, symbol=...)
                          is called; batched inference over all windows
                                              │
                    ┌─────────────────────────┼─────────────────────────┐
                    ▼                         ▼                         ▼
          swing_backtest/            swing_paper_trading/      live_trading_swing/
          real Zerodha CNC costs,    virtual orders, real      real orders, kill-switch +
          no-lookahead simulation    broker prices, once/day   confirmation gate, once/day
                    │                         │                         │
                    └─────────────────────────┼─────────────────────────┘
                                              ▼
                          api/swing_dashboard/  (FastAPI + static frontend, port 9000)
                          backtest/paper fully controllable here; live is read-only status
```

A symbol without a trained checkpoint yet is skipped (with a one-time warning), not a crash — backtest/paper/live all tolerate a partially-trained universe.

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
├── models/                         ← Created by training - one subdirectory PER SYMBOL
│   └── <SYMBOL>/
│       ├── hydra_best.keras        ← That symbol's best checkpoint (by val_prediction_dir_acc)
│       └── model_shape.pkl         ← Exact trained shape (context_window/forecast_steps/
│                                      n_features/symbol) — read by MLSwingStrategy at load time
├── logs/                           ← train.log, edge_tracker_<SYMBOL>.csv (one per symbol),
│                                      paper/live trade logs
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
    │   ├── preprocess.py           ← 30-feature equity pipeline, per-symbol dataset builder,
    │   │                              real swing-outcome labeling, Nifty 50 relative strength
    │   ├── fetch_historical.py     ← Cached daily-candle downloader (via configured broker),
    │   │                              also fetches NIFTY as a benchmark reference
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
Market Input (60 trading days × 30 features)  ← ONE model per symbol, trained only on that symbol's own history
        │
   [GaussianNoise(0.05)]
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
   │  │ Dropout(0.30)      │    │  ← Raised from 0.15 - per-symbol data is much smaller
   │  └────────────────────┘    │     than a pooled set, so overfitting risk is higher
   └─────────────────────────────┘
        │
   RMSNorm (full sequence)
        │
   ┌────┼──────────┐
   │    │          │
[Prediction] [Certainty] [Reasoning]
 (21×3 traj)  (60 scores) (4 classes: LONG/SHORT/FEE_TRAP/NOISE)
```

**No GPT-style next-token head.** An auxiliary next-candle-token prediction task was evaluated and dropped — it showed only marginal value, and one symbol's own daily history (~4-6K windows) isn't enough to justify taxing model capacity on a task that wasn't clearly paying for itself. This is also why the model takes a **single input** (just the feature window), not a dual market+token input.

**Why per-symbol, not pooled across the universe:** each stock has ~4,000–6,500 real historical windows on its own — genuinely limited for a 7.3M-parameter model, and pooling multiple symbols was the original design specifically to compensate for that. It was switched to one-model-per-symbol on the reasoning that different stocks have different volatility regimes and behavior a shared model would average away. The tradeoff is real: less data per model means a higher overfitting risk, which is why dropout/noise/weight-decay were all raised and `EarlyStopping` uses a generous patience (25) with `restore_best_weights=True` — the final checkpoint is always whichever epoch had the best validation score, regardless of how long training continued past it.

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

## 📊 30 Input Features

All ratio-based or bounded — no raw price level fed directly without local normalization — since each stock's own price level varies wildly over its history (and, if ever pooled again, across stocks too — ₹400 ITC vs. ₹2,400 RELIANCE). Dynamic Local Scaling (`apply_dls`) additionally z-scores every window against its own local mean/std before it reaches the model.

| Category | Features |
|---|---|
| **Time** | `day_of_week`, `month_sin`, `month_cos` |
| **Price (raw, DLS-scaled)** | `open`, `high`, `low`, `close`, `volume` |
| **Returns** | `ret_1d`, `ret_5d`, `ret_20d` |
| **Trend / MA distance** | `sma20_dist`, `sma50_dist`, `sma200_dist`, `ema20_dist` |
| **Momentum** | `rsi`, `macd`, `macd_signal`, `macd_hist`, `adx` |
| **Volatility** | `atr_pct`, `bb_width`, `bb_position`, `volatility_20d` |
| **Volume** | `volume_ratio` |
| **Range position** | `hl_position_20` (20-day), `hl_position_252` (52-week) |
| **Trend structure** | `supertrend_dir`, `donchian_position` (10-day short-term breakout — deliberately a *different* window than `hl_position_20`'s 20-day one, since using the same window would make it a literal duplicate feature) |
| **Market context** | `rel_strength_20d` — this stock's 20-day return minus the Nifty 50 index's 20-day return over the same period, i.e. is it actually beating the market or just moving with it. Every other feature only looks at the stock in isolation; this is the one signal that gives the model market context. |

No order-book/microstructure features (funding rate, taker flow, order-book imbalance) — none of that exists for NSE cash equity, so the feature set is built entirely from real daily OHLCV and derived technical/trend indicators.

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
| **Tradeable universe** | 38 symbols, sector-balanced (banking, IT, FMCG, auto, pharma, energy, metals, cement, infra, telecom — see `config/settings.yaml` → `universe.symbols`) | Backtest/paper/live pull data for all of these; a pooled model trained on 6 banks and nothing else would just learn "bank patterns" |
| **Training subset** | 10 symbols by default (`config/settings.yaml` → `training.symbols`), one independent model each | Training all 38 sequentially on CPU is slow; trim/widen this list freely — every symbol not in it is simply skipped (with a warning) by backtest/paper/live until it's trained |
| **Data per symbol** | ~4,000–6,500 real daily candles (earliest available per symbol, not a fixed cutoff — e.g. RELIANCE back to 2000, most others to 2003) | ~4,400–6,400 usable training windows per symbol, depending on history length |
| **Context Window** | 60 trading days (~3 months) | |
| **Forecast Steps** | 20 trading days | Matches `max_holding_days` |
| **Batch Size** | 32 | |
| **Epochs** | 300 (EarlyStopping patience=25, `restore_best_weights=True`) | Patience only controls how long training keeps looking for a better epoch before stopping — it can never produce a worse final model than the actual best epoch seen, so it's kept generous rather than tight |
| **Regularization** | Dropout 0.30, GaussianNoise 0.05, AdamW weight_decay=0.15 | Raised from an initial 0.15/0.02/0.05 — per-symbol data is limited, so overfitting risk is real (observed directly: an early run's val accuracy peaked at epoch 1 and drifted down every epoch after, before these were raised) |
| **Optimizer** | AdamW (lr=1e-4→1e-5 cosine decay, clipnorm=1.0) | |
| **Train/val split** | Chronological **by date**, per symbol | No lookahead — a window's end-date always precedes every validation window's end-date |
| **Epoch time (CPU, no GPU)** | ~30 min for a ~4,500-window symbol at ~14s/step; scales with that symbol's own history length | Per-symbol training is inherently faster per model than the old pooled approach was overall, since each run only processes one symbol's data |

---

## 🚀 Quick Start

### 1. Setup
```bash
cd /var/www/html/ML/kat
pip install -e .
```
Note: stable `tensorflow` has no wheel for Python ≥3.14 yet — on 3.14+, `pyproject.toml` falls back to `tf-nightly` automatically (verified working). On Python 3.11–3.13, the stable `tensorflow` package is used.
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
# train only specific symbols (overrides config/settings.yaml -> training.symbols):
python train.py --symbols RELIANCE HDFCBANK INFY
```
Loops through the configured symbols one at a time, training and saving an independent model per symbol: `models/<SYMBOL>/hydra_best.keras` + `models/<SYMBOL>/model_shape.pkl`. Monitor with:
```bash
tail -f logs/train.log
tail -f logs/edge_tracker_<SYMBOL>.csv   # per-epoch Wilson 95% CI significance verdict, one file per symbol
```

### 3. Backtest
```bash
python auto_run.py backtest
python auto_run.py backtest --refresh   # force re-download historical data
```
Real Zerodha delivery costs, no lookahead (entries fill at next day's open), portfolio-level position limits. Any symbol without a trained checkpoint yet (`models/<SYMBOL>/hydra_best.keras`) is skipped with a warning, not a crash — the backtest still runs against whichever symbols do have one.

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
- **kiteconnect/tensorflow import order**: importing `tensorflow` before `kiteconnect` in the same process segfaults (verified — a native-library conflict, not a Python error). `api/swing_dashboard/main.py` forces `kiteconnect` to import first, since that one long-lived process can hit both an ML backtest and a Zerodha paper/live session across its lifetime. Relevant if you add new entry points that touch both.

---

## ⚠️ Honest Status

**Verified via real execution:**
- Data pipeline, model, and per-symbol training loop (real training runs cleanly end-to-end)
- A real trained checkpoint (`models/HDFCBANK/hydra_best.keras`) exists and was validated: epoch 1 reached `val_dir_acc=53.70%`, 95% CI `[52.77%, 54.64%]` — statistically significant (lower bound > 50%), per the Wilson-interval `EdgeTracker`
- `ml_swing` strategy running real batched inference against that checkpoint, feeding into the real backtest engine with real Zerodha delivery costs — produces correct, real results (0 trades in the tested window, which is expected: certainty threshold 0.85 is strict and that model was only a few epochs into training)
- API dashboard (actually starts, serves real responses)
- Broker connection code (constructs correct requests; error handling verified against a real 401; NIFTY 50 index history fetched and cached successfully)

**Not yet verified:**
- A checkpoint trained to actual convergence / EarlyStopping completion for any symbol (all runs so far were interrupted partway through by further fixes — regularization, feature corrections — each of which required restarting from epoch 0)
- Paper trading's real-time loop (needs a fresh, non-expired broker token)
- Live trading (deliberately not casually tested — code was carefully read, not executed, given real money is involved)
- Whether per-symbol models actually find a real, tradeable edge once fully trained — 53.70% at epoch 1 is a real but weak signal, not a conclusion

---

⚓ **Sovereign Kraken — Equity Swing, Neural Network Only.**
