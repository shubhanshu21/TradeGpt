"""
SOVEREIGN KRAKEN — Backtest Engine (V5.0) ⚓📊
===============================================
Runs a walk-forward directional backtest on the current best model.
Tests: directional accuracy, simulated P&L, fee-aware win rate.

Usage:
    python src/backtest_checkup.py
"""

import sys, os
os.environ["PYTHONUNBUFFERED"] = "1"
import numpy as np
import pandas as pd
import keras
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

from core.hydra import build_kraken, SovereignLoss
from data.preprocess import KATScaler, build_feature_cols, compute_indicators
from exchange.fetch_data  import fetch_live_kat_data

# ── CONFIG ────────────────────────────────────────────────────────────────────
SYMBOL       = "BTCUSD"
TIMEFRAME    = "15m"      # Match training timeframe
CTX_WIN      = 120        # Must match training
N_CANDLES    = 1000       # Test window
THRESHOLD    = 0.15       # Signal threshold
from config.sovereign_config import FEE_RATE
FEE_PCT      = FEE_RATE   # Synced from sovereign_config (0.0012)
TRADE_SIZE   = 1          # 1 contract (for P&L simulation)

# ── Smart Checkpoint Selection ────────────────────────────────────────────────
MODELS_DIR  = ROOT / "models"
checkpoints = sorted(MODELS_DIR.glob("hydra_checkpoint_E*.keras"), reverse=True)
MODEL_PATH  = checkpoints[0] if checkpoints else MODELS_DIR / "hydra_best.keras"
if not MODEL_PATH.exists():
    print(f"❌ No model at {MODEL_PATH} — train first."); sys.exit(1)

print(f"📦 Loading model: {MODEL_PATH.name}")
print(f"🏗️  Re-building Kraken V12.0...")
features  = build_feature_cols()
n_feats   = len(features)
model = build_kraken(n_features=n_feats, context_window=CTX_WIN, forecast_steps=15)
print(f"🧠 Loading Weights from: {MODEL_PATH.name} | Features: {n_feats}")
model.load_weights(str(MODEL_PATH))
print(f"✅ Brain loaded")

print(f"\n⚓ SOVEREIGN BACKTEST ENGINE V5.0 — {SYMBOL}")
print("="*60)

# DLS — no global scaler needed (matches training pipeline)
def _dls(window): return (window - window.mean(0)) / (window.std(0) + 1e-8)

df = fetch_live_kat_data(symbol=SYMBOL, n_candles=N_CANDLES + CTX_WIN + 50, timeframe=TIMEFRAME)
print(f"   Got {len(df):,} candles")
df_feat = compute_indicators(df)
data    = df_feat[features].values.astype("float32")

print(f"\n🔄 Running walk-forward backtest ({N_CANDLES:,} steps)...")

results = []
close_col = features.index("close")

for i in range(CTX_WIN, len(data) - 15):
    X_in = _dls(data[i - CTX_WIN : i]).reshape(1, CTX_WIN, n_feats).astype("float32")
    
    # Dual Output: [0] = Prediction (1, 16, 3), [1] = Certainty (1, 120)
    out       = model(X_in, training=False)
    pred      = out[0].numpy()[0]   # Predictions
    cert      = out[1].numpy()[0]   # Certainty mean
    
    mean_move = float(np.mean(pred[1:, 0])) # Future returns
    mean_cert = float(np.mean(cert))        # Expert consensus score
    
    # V10.3 Strategy: Only trade when consensus > threshold
    # Note: cert is scaled by 120, so 110+ means high agreement
    if mean_cert < 110:
        signal = "HOLD" # Experts are confused
    elif mean_move > THRESHOLD:
        signal = "LONG"
    elif mean_move < -THRESHOLD:
        signal = "SHORT"
    else:
        signal = "HOLD"

    # Actual future 1-min close vs current close (in scaled space)
    actual_now  = float(scaled[i,     close_col])
    actual_next = float(scaled[i + 1, close_col])
    actual_dir  = np.sign(actual_next - actual_now)

    results.append({
        "i":          i,
        "signal":     signal,
        "mean_move":  mean_move,
        "actual_dir": actual_dir,
    })

    if i % 500 == 0:
        print(f"   Step {i - CTX_WIN:,}/{N_CANDLES:,}...", end="\r")

print(f"\n   ✅ {len(results):,} steps evaluated")

# ── Analysis ──────────────────────────────────────────────────────────────────
df_r = pd.DataFrame(results)

# Trades only (exclude HOLDs)
trades = df_r[df_r["signal"] != "HOLD"].copy()
n_trades  = len(trades)
n_signals = len(df_r)
hold_pct  = (n_signals - n_trades) / n_signals * 100

# Directional accuracy on trades
def pred_dir(row):
    return 1.0 if row["signal"] == "LONG" else -1.0

trades["pred_dir"] = trades.apply(pred_dir, axis=1)
trades["correct"]  = (trades["pred_dir"] == trades["actual_dir"])

win_rate  = trades["correct"].mean() * 100
long_wr   = trades[trades["signal"] == "LONG"]["correct"].mean() * 100
short_wr  = trades[trades["signal"] == "SHORT"]["correct"].mean() * 100

# Simulated P&L (1 pip = 1 scaled unit → directional binary result)
# Win = +1 unit, Lose = -1 unit, minus 2× fee per round trip
fee_per_trade = FEE_PCT * 2   # entry + exit
trades["pnl"] = trades.apply(
    lambda r: (1 - fee_per_trade) if r["correct"] else (-1 - fee_per_trade), axis=1
)
total_pnl    = trades["pnl"].sum()
cum_pnl      = trades["pnl"].cumsum()
max_drawdown = (cum_pnl - cum_pnl.cummax()).min()
sharpe_proxy = trades["pnl"].mean() / (trades["pnl"].std() + 1e-9)

# ── Report ────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("📊 SOVEREIGN BACKTEST REPORT V5.0")
print("="*60)
print(f"  Symbol     : {SYMBOL} {TIMEFRAME}")
print(f"  Model      : {MODEL_FILE}")
print(f"  Window     : {N_CANDLES:,} candles (~{N_CANDLES//1440:.1f} days)")
print(f"  Threshold  : ±{THRESHOLD} (Z-score)")
print("-"*60)
print(f"  Signals    : {n_trades:,} trades  |  {hold_pct:.1f}% HOLD")
print(f"  Win Rate   : {win_rate:.1f}%  (Long: {long_wr:.1f}%  Short: {short_wr:.1f}%)")
print(f"  Net P&L    : {total_pnl:+.2f} units  ({total_pnl/n_trades*100:+.1f}% per trade)")
print(f"  Max Drawdown: {max_drawdown:.2f} units")
print(f"  Sharpe Proxy: {sharpe_proxy:.3f}")
print("-"*60)

# Verdict
if win_rate > 55:
    verdict = f"✅ PROFITABLE ALPHA — {win_rate:.1f}% win rate"
elif win_rate > 51:
    verdict = f"⚠️  WEAK ALPHA — {win_rate:.1f}% (marginal edge, needs more training)"
else:
    verdict = f"🛑 NO EDGE — {win_rate:.1f}% (below fee breakeven, keep training)"

print(f"\n  {verdict}")
print("="*60)
print(f"\n  Fee breakeven: >50.5% win rate")
print(f"  Profitable:    >53.0% win rate")
print(f"  Strong edge:   >57.0% win rate")
print("="*60 + "\n")
