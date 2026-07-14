# ⚓ Sovereign Kraken — Equity Swing Trading (Neural Network Only)

> **Self-contained Indian equity (NSE) swing trading system — HYDRA transformer + MoE, GPT-lineage architecture, real Zerodha delivery costs, daily candles, multi-day holds.**

---

## 🧠 What Is This?

A neural network (transformer + Mixture-of-Experts, the same architectural family as GPT/DeepSeek) trained to trade Indian equities on a swing-trading basis: it looks at 60 trading days of a stock's history and predicts whether a position entered tomorrow would be profitable within a 20-trading-day hold, using a 5% stop-loss / 10% target matching real, validated swing-trading parameters.

This is a fully self-contained system — data fetching, broker integration, a realistic cost model, and the backtest/paper/live engines all live in this one project, with no dependency on anything external.

**There are no rule-based technical-indicator strategies in this system.** The only trading logic here is the trained neural network — no Bollinger breakout, no inside-bar breakout, no Supertrend crossover rules.

**Training is two-phase — pooled, then specialized.** Phase 1 pretrains one shared model on all universe symbols pooled together (broad general pattern recognition, the same idea as a trader who's watched hundreds of stocks). Phase 2 fine-tunes a separate copy of that pretrained model per symbol (`models/<SYMBOL>/hydra_best.keras`), specializing on that stock's own behavior instead of learning everything from scratch on just its own ~5,000 windows. This exists specifically because per-symbol-from-scratch training showed real overfitting (val accuracy peaking at epoch 1, drifting down after) — transfer learning (pretrain-then-finetune) is a published, evidence-backed technique for exactly this stock-prediction data-scarcity problem, not a guess.

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
                              │  → 95 features/day: 33 tech/   │
                              │    trend/relative-strength/vol │
                              │    (incl. OBV, VWAP-dist, VIX) │
                              │    + 62 candlestick patterns   │
                              │  apply_dls() per 60-day window │
                              └───────────────┬───────────────┘
                                              │
                                              ▼
                       PHASE 1: pretrain_base_model() — ALL universe
                       symbols pooled → models/_pretrained_base/
                                              │
                    ┌─────────────────────────┼─────────────────────────┐
                    ▼                         ▼                         ▼
       PHASE 2: train_one_symbol() fine-tunes FROM the pretrained base,
                one specialized copy per configured symbol
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
    │   ├── preprocess.py           ← 92-feature equity pipeline (30 technical/trend +
    │   │                              62 candlestick patterns), dataset builder, real
    │   │                              swing-outcome labeling, GPT-style next-candle
    │   │                              vocabulary (fit_return_vocab/tokenize_returns)
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
Market Input (60 trading days × 92 features)
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
   ┌────┬──────────┬──────────────┐
   │    │          │              │
[Prediction] [Certainty] [Reasoning] [Next-Candle]
 (21×6 traj)  (60 scores) (4 classes)  (32-token softmax)
```

**Four output heads, all reading the same shared backbone:**
| Head | Shape | What it does |
|---|---|---|
| **`prediction`** | (21, 6) | Forecasts today + the next 20 days' close, volatility, volume, open, high, low — a real full-candle trajectory, not just close |
| **`certainty`** | (60,) | Per-day confidence, sigmoid-calibrated — gates whether a signal fires at all (≥0.85 required) |
| **`reasoning`** | (4,) | The actual trade decision: LONG / SHORT / FEE_TRAP / NOISE (see labels below) |
| **`next_candle`** | (32,) | GPT-style next-token classification — predicts a discretized bucket for tomorrow's return, trained with real next-token cross-entropy over a quantile-binned vocabulary (`fit_return_vocab`/`tokenize_returns` in `preprocess.py`). **Single-step only** — it does not chain predictions into a multi-day generated path the way GPT chains words into a sentence; that would compound errors badly on numeric data with no grammar-like self-correction. Its predicted direction (decoded via `bin_centers`) is used as a third independent confirmation signal in `ml_strategy.py`, alongside `reasoning` and the `prediction` trajectory's direction — a trade only fires if all three agree. |

**Why pretrain-then-finetune, not from-scratch-per-symbol:** each stock has ~4,000–6,500 real historical windows on its own — genuinely limited for a 7.3M-parameter model (observed directly: an early from-scratch run's val accuracy peaked at epoch 1 and drifted down every epoch after, the textbook small-data overfitting signature). Phase 1 pretrains one model on all universe symbols pooled (~200K windows) for broad pattern recognition; Phase 2 fine-tunes a specialized copy per symbol from those pretrained weights at a lower learning rate, instead of learning everything from that symbol's own data alone. This is a published, evidence-backed technique for exactly this problem (see e.g. Zeng & Zhao 2019, *Short-Term Stock Price Movement Prediction using Transfer Learning*), not a novel guess. Regularization (dropout/noise/weight-decay all raised) and a generous `EarlyStopping` patience (25, `restore_best_weights=True` — a higher patience can only cost time, never quality) further guard against the overfitting risk either phase still carries.

### Key Components
| Component | Description |
|---|---|
| **Causal Softmax Attention + QK-Norm** | Real GPT/DeepSeek-style attention |
| **GatedMoE-32** | 32 routed expert sub-networks (top-4) + always-active shared-expert path (DeepSeekMoE-style) — sized for this dataset's scale, not billions-of-tokens scale |
| **TurboQuant** | INT8 simulation with orthogonal rotation |
| **SovereignLoss** | Volatility-weighted directional loss on close + a lower-weighted MSE term on open/high/low (real full-candle training, not just close) |
| **RMSNorm** | Fast root mean square normalization (LLaMA-style) |
| **SwiGLU (2x expansion)** | Gated activation, real hidden-dim expansion |
| **WarmupCosineDecay** | Linear LR warmup before cosine decay (1e-4 pretrain/from-scratch, 3e-5 fine-tuning from a pretrained base) |

---

## 📊 95 Input Features

All ratio-based or bounded — no raw price level fed directly without local normalization — since each stock's own price level varies wildly over its history (and, if ever pooled again, across stocks too — ₹400 ITC vs. ₹2,400 RELIANCE). Dynamic Local Scaling (`apply_dls`) additionally z-scores every window against its own local mean/std before it reaches the model.

| Category | Features |
|---|---|
| **Time** | `day_of_week`, `month_sin`, `month_cos` |
| **Price (raw, DLS-scaled)** | `open`, `high`, `low`, `close`, `volume` |
| **Returns** | `ret_1d`, `ret_5d`, `ret_20d` |
| **Trend / MA distance** | `sma20_dist`, `sma50_dist`, `sma200_dist`, `ema20_dist` |
| **Momentum** | `rsi`, `macd`, `macd_signal`, `macd_hist`, `adx` |
| **Volatility** | `atr_pct`, `bb_width`, `bb_position`, `volatility_20d` |
| **Volume** | `volume_ratio`, `obv_dist` — On-Balance Volume's distance from its own 20-day moving average (is accumulation/distribution volume confirming or diverging from price?). Raw OBV is an unbounded running total, so like the trend features above we use its distance from its own recent normal, not the raw level. |
| **Range position** | `hl_position_20` (20-day), `hl_position_252` (52-week) |
| **Trend structure** | `supertrend_dir`, `donchian_position` (10-day short-term breakout — deliberately a *different* window than `hl_position_20`'s 20-day one, since using the same window would make it a literal duplicate feature) |
| **Price vs. volume-weighted average** | `vwap20_dist` — distance from a 20-day rolling volume-weighted average of the daily typical price `(high+low+close)/3`. **Not** intraday VWAP (that needs intraday tick/bar data we don't have — daily OHLCV only) — an honestly-named daily-bar analog. |
| **Market context** | `rel_strength_20d` — this stock's 20-day return minus the Nifty 50 index's 20-day return over the same period, i.e. is it actually beating the market or just moving with it. `vix_zscore_60d` — India VIX (NSE's options-derived "fear gauge") z-scored against its own trailing 60-day mean/std, so the model knows whether market-wide fear is unusually high or low right now, not tied to any one stock's own price/volume. These, plus the candlestick patterns below, are the only features that don't reduce to "some rolling-window trend/momentum/volatility summary of this one stock" — genuinely different information categories. |
| **Candlestick patterns** (62) | All of `pandas-ta-classic`'s candlestick pattern set (Doji, Hammer, Engulfing, Morning/Evening Star, Three White Soldiers, Head-and-shoulders-adjacent reversal/continuation shapes, etc.) — pure candle-SHAPE information no other feature here captures at all. Each is TA-Lib's standard signed scale (-1 bearish, 0 none, +1 bullish, scaled from the usual ±100), so one column carries both the long and short reading of a pattern. See `preprocess.py`'s `CANDLESTICK_PATTERN_NAMES` for the full list. |

No order-book/microstructure features (funding rate, taker flow, order-book imbalance) — none of that exists for NSE cash equity, so the feature set is built entirely from real daily OHLCV, India VIX, and derived technical/trend/candlestick-pattern indicators.

Cross-asset features (USD/INR, crude oil) were deliberately left out — on NSE those only exist as futures/options contracts (`MCX_FO`/`BCD_FO`/`NCD_FO` segments via the Upstox instrument master), which need proper contract-rollover handling to turn into a clean continuous daily series. Bolting that on without doing it properly would be a fragile, silently-wrong feature, so it's skipped rather than rushed.

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
| **Tradeable universe** | 38 symbols, sector-balanced (banking, IT, FMCG, auto, pharma, energy, metals, cement, infra, telecom — see `config/settings.yaml` → `universe.symbols`) | Backtest/paper/live pull data for all of these |
| **Phase 1: pretrain pool** | ALL 38 universe symbols pooled (~200K windows) | Broad general pattern recognition before any per-symbol specialization - see the architecture section above for why |
| **Phase 2: fine-tune subset** | 10 symbols by default (`config/settings.yaml` → `training.symbols`), one specialized copy each | Every symbol not in this list is simply skipped (with a warning) by backtest/paper/live until it's trained |
| **Data per symbol** | ~4,000–6,500 real daily candles (earliest available per symbol, not a fixed cutoff — e.g. RELIANCE back to 2000, most others to 2003) | ~4,400–6,400 usable training windows per symbol, depending on history length |
| **Context Window** | 60 trading days (~3 months) | |
| **Forecast Steps** | 20 trading days | Matches `max_holding_days` |
| **Batch Size** | 32 | |
| **Epochs** | Pretrain: 300 ceiling, patience=25. Fine-tune: 300 ceiling, patience=25. Both `restore_best_weights=True` | Patience only controls how long training keeps looking for a better epoch before stopping — it can never produce a worse final model than the actual best epoch seen, so it's kept generous rather than tight |
| **Regularization** | Dropout 0.30, GaussianNoise 0.05, AdamW weight_decay=0.15 (fine-tune) / 0.05 (pretrain, larger pooled dataset) | Raised from an initial 0.15/0.02/0.05 — per-symbol data is limited, so overfitting risk is real (observed directly: an early from-scratch run's val accuracy peaked at epoch 1 and drifted down every epoch after) |
| **Learning rate** | 1e-4 (pretrain / from-scratch), 3e-5 (fine-tuning from a pretrained base) | Lower LR for fine-tuning is standard transfer-learning practice - adjusts what the base already learned instead of overwriting it |
| **Train/val split** | Chronological **by date**, within each phase's own pool | No lookahead — a window's end-date always precedes every validation window's end-date |
| **Epoch time (CPU, no GPU)** | Pretrain (38 symbols pooled): ~19h/epoch. Fine-tune (1 symbol): ~30min-1h/epoch depending on that symbol's history length | Measured directly on this hardware; both phases lean on `EarlyStopping` + generous patience rather than a tight epoch budget, since time isn't the constraint here |

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
# skip phase 1 entirely - fine-tune each symbol from random init instead:
python train.py --skip_pretrain
# redo phase 1 even if models/_pretrained_base/hydra_pretrained.keras already exists:
python train.py --force_pretrain
```
Phase 1 pretrains one shared model on all universe symbols pooled (`models/_pretrained_base/hydra_pretrained.keras`) - skipped automatically on future runs once it exists, unless `--force_pretrain`. Phase 2 then loops through the configured symbols one at a time, fine-tuning a specialized copy of that pretrained model per symbol: `models/<SYMBOL>/hydra_best.keras` + `models/<SYMBOL>/model_shape.pkl`. Monitor with:
```bash
tail -f logs/train.log
tail -f logs/edge_tracker__pretrained_base.csv   # phase 1's own per-epoch significance verdict
tail -f logs/edge_tracker_<SYMBOL>.csv           # per-epoch Wilson 95% CI significance verdict, one file per fine-tuned symbol
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
- Data pipeline, model, and the full two-phase pretrain→fine-tune→inference chain — a complete smoke test (tiny data, 1 epoch each) ran phase 1, phase 2 (weight-loading from the pretrained base, no shape mismatch), and real batched inference through `ml_strategy.py` end-to-end with zero errors, including correctly decoding the next-candle head's token back into a direction and using it as a real gating signal
- An earlier, now-superseded architecture (pre-candlestick-features, pre-OHLC-prediction, pre-next-candle-head) did produce one real trained checkpoint that reached `val_dir_acc=53.70%` at epoch 1, 95% CI `[52.77%, 54.64%]` — statistically significant per the Wilson-interval `EdgeTracker`. That confirms the pipeline mechanics work; it does not carry forward as a result for the current architecture, which has since changed materially (92 features vs. 30, 4 output heads vs. 3, pretrain-then-finetune vs. from-scratch)
- API dashboard (actually starts, serves real responses, live training-progress chart verified against real and synthetic data)
- Broker connection code (constructs correct requests; error handling verified against a real 401; NIFTY 50 index history fetched and cached successfully)
- `pandas-ta-classic`'s 62 candlestick patterns verified directly against real HDFCBANK data — all compute cleanly, no NaN, genuine nonzero pattern detections (not a silent no-op)

**Not yet verified:**
- A full real training run (not a smoke test) completing under the current architecture — 92 features, 4 output heads, pretrain-then-finetune — is in progress as of this writing; no epoch has finished yet at real (non-toy) data scale
- Whether the pretrain-then-finetune approach actually beats from-scratch training on real per-symbol results (the honest test is comparing a fine-tuned symbol's peak accuracy/stability against the superseded from-scratch baseline above, once both exist under the same feature set)
- Whether the candlestick patterns and next-candle head measurably help, versus just adding parameters - the model has 3 independent gates now (reasoning, prediction-trajectory, next-candle direction) instead of 2, which should make it fire *fewer, more conservative* signals; whether that's a net improvement is an open question until real backtest trades exist
- Paper trading's real-time loop (needs a fresh, non-expired broker token)
- Live trading (deliberately not casually tested — code was carefully read, not executed, given real money is involved)
- Whether per-symbol models actually find a real, tradeable edge once fully trained — 53.70% at epoch 1 is a real but weak signal, not a conclusion

---

⚓ **Sovereign Kraken — Equity Swing, Neural Network Only.**
