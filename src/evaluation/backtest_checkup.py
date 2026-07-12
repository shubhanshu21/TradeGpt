"""
SOVEREIGN KRAKEN — Backtest Engine (V6.0) ⚓📊
===============================================
Runs a walk-forward directional backtest on the current best model.
Tests: directional accuracy, simulated P&L, fee-aware win rate.

Usage:
    python src/evaluation/backtest_checkup.py
"""

import sys, os
os.environ["PYTHONUNBUFFERED"] = "1"
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

from core.inference import load_trained_model, prepare_window, run_inference
from config.sovereign_config import FEE_RATE

# ── CONFIG ────────────────────────────────────────────────────────────────────
N_CANDLES    = 1000       # Test window
THRESHOLD    = 0.15       # Signal threshold (Z-score)
FEE_PCT      = FEE_RATE   # Synced from sovereign_config (0.0012)

MODELS_DIR = ROOT / "models"
model, vocab = load_trained_model(MODELS_DIR)
ctx_win  = vocab["context_window"]
forecast = vocab["forecast_steps"]
timeframe = vocab["timeframe"]
print(f"✅ Model loaded — timeframe={timeframe}, context_window={ctx_win}, forecast_steps={forecast}")

print(f"\n⚓ SOVEREIGN BACKTEST ENGINE V6.0 — BTCUSD {timeframe}")
print("=" * 60)

df = pd.read_parquet(ROOT / "data" / f"BTCUSD_{timeframe}_history_master.parquet")
df = df.reset_index().rename(columns={"index": "timestamps", "timestamp": "timestamps"})
print(f"   Got {len(df):,} candles")

close_col_vals = df["close"].values
n_test = min(N_CANDLES, len(df) - ctx_win - forecast - 150 - 1)
start_idx = len(df) - n_test - forecast

print(f"\n🔄 Running walk-forward backtest ({n_test:,} steps)...")

results = []
for step, i in enumerate(range(start_idx, start_idx + n_test)):
    x_scaled, tok_ids, local_mean, local_std, t_close = prepare_window(df, i, vocab)
    pred, cert, reasoning, _ = run_inference(model, x_scaled, tok_ids)

    p_anchor = pred[0, 0]
    mean_move = float(np.mean(pred[1:, 0] - p_anchor))  # Future returns relative to anchor (Z-score space)
    mean_cert = float(np.mean(cert))
    reasoning_class = int(np.argmax(reasoning))

    # Strategy: only trade when consensus is high AND the reasoning head agrees
    # with the price-trajectory direction (closes the gap found earlier where
    # live_trader.py could fire against its own reasoning gate).
    reasoning_dir = 1 if reasoning_class == 0 else (-1 if reasoning_class == 1 else 0)
    price_dir = 1 if mean_move > 0 else (-1 if mean_move < 0 else 0)

    if mean_cert < 0.85:
        signal = "HOLD"
    elif reasoning_class not in (0, 1) or reasoning_dir != price_dir:
        signal = "HOLD"
    elif mean_move > THRESHOLD:
        signal = "LONG"
    elif mean_move < -THRESHOLD:
        signal = "SHORT"
    else:
        signal = "HOLD"

    actual_now = float(close_col_vals[i - 1])
    actual_next = close_col_vals[i:i + forecast]
    actual_mean_future = float(np.mean(actual_next))
    actual_dir = np.sign(actual_mean_future - actual_now)

    results.append({"i": i, "signal": signal, "mean_move": mean_move, "actual_dir": actual_dir})

    if step % 100 == 0:
        print(f"   Step {step:,}/{n_test:,}...", end="\r")

print(f"\n   ✅ {len(results):,} steps evaluated")

# ── Analysis ──────────────────────────────────────────────────────────────────
df_r = pd.DataFrame(results)
trades = df_r[df_r["signal"] != "HOLD"].copy()
n_trades = len(trades)
n_signals = len(df_r)
hold_pct = (n_signals - n_trades) / n_signals * 100 if n_signals else 0

if n_trades == 0:
    print("\n⚠️  No trades fired under current gates in this window — nothing to report.")
    sys.exit(0)

trades["pred_dir"] = trades["signal"].map({"LONG": 1.0, "SHORT": -1.0})
trades["correct"] = (trades["pred_dir"] == trades["actual_dir"])

win_rate = trades["correct"].mean() * 100
long_wr = trades[trades["signal"] == "LONG"]["correct"].mean() * 100 if (trades["signal"] == "LONG").any() else 0
short_wr = trades[trades["signal"] == "SHORT"]["correct"].mean() * 100 if (trades["signal"] == "SHORT").any() else 0

fee_per_trade = FEE_PCT  # FEE_RATE already represents the full round trip (entry+exit), not one side
trades["pnl"] = trades["correct"].map({True: 1 - fee_per_trade, False: -1 - fee_per_trade})
total_pnl = trades["pnl"].sum()
cum_pnl = trades["pnl"].cumsum()
max_drawdown = (cum_pnl - cum_pnl.cummax()).min()
sharpe_proxy = trades["pnl"].mean() / (trades["pnl"].std() + 1e-9)

print("\n" + "=" * 60)
print("📊 SOVEREIGN BACKTEST REPORT V6.0")
print("=" * 60)
print(f"  Symbol/TF  : BTCUSD {timeframe}")
print(f"  Window     : {n_test:,} candles")
print(f"  Threshold  : ±{THRESHOLD} (Z-score) + reasoning/direction agreement gate")
print("-" * 60)
print(f"  Signals    : {n_trades:,} trades  |  {hold_pct:.1f}% HOLD")
print(f"  Win Rate   : {win_rate:.1f}%  (Long: {long_wr:.1f}%  Short: {short_wr:.1f}%)")
print(f"  Net P&L    : {total_pnl:+.2f} units  ({total_pnl/n_trades*100:+.1f}% per trade)")
print(f"  Max Drawdown: {max_drawdown:.2f} units")
print(f"  Sharpe Proxy: {sharpe_proxy:.3f}")
print("-" * 60)

if win_rate > 55:
    verdict = f"✅ PROFITABLE ALPHA — {win_rate:.1f}% win rate"
elif win_rate > 51:
    verdict = f"⚠️  WEAK ALPHA — {win_rate:.1f}% (marginal edge, needs more training)"
else:
    verdict = f"🛑 NO EDGE — {win_rate:.1f}% (below fee breakeven, keep training)"

print(f"\n  {verdict}")
print("=" * 60)
print(f"\n  Fee breakeven: >50.5% win rate")
print(f"  Profitable:    >53.0% win rate")
print(f"  Strong edge:   >57.0% win rate")
print("=" * 60 + "\n")
