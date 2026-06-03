"""
SOVEREIGN KRAKEN — Live-Logic Simulator & Backtester (V6.0) ⚓📊
============================================================
Evaluates historical model performance using the exact same filters, gates,
and position management rules found in src/trading/live_trader.py.
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

from core.hydra import build_kraken
from data.preprocess import build_feature_cols, compute_indicators, apply_dls
from exchange.fetch_data  import fetch_live_kat_data
from config.sovereign_config import (FEE_RATE, LEVERAGE, POSITION_SIZE_PCT, MAX_TRADES_PER_DAY,
                                     BREAKEVEN_TRIGGER_PCT, TRAILING_STOP_PCT, CERT_THRESHOLD, LABELS)

# ── SIMULATION CONFIG ─────────────────────────────────────────────────────────
SYMBOL         = "BTCUSD"
TIMEFRAME      = "15m"      
CTX_WIN        = 120        
N_CANDLES      = 1000       # Test window size
THRESHOLD      = 0.08       # Matches live_trader.py THRESHOLD (increased base conviction)
MIN_SWING_FLOOR = 100.0     # Clamps
MIN_SWING_CEIL  = 100.0     
COOLDOWN_BARS  = 1          # live_trader cooldown

# ── Load Model ────────────────────────────────────────────────────────
MODELS_DIR  = ROOT / "models"
checkpoints = sorted(MODELS_DIR.glob("hydra_checkpoint_E*.keras"), reverse=True)
MODEL_PATH  = checkpoints[0] if checkpoints else MODELS_DIR / "hydra_best.keras"
if not MODEL_PATH.exists():
    print(f"❌ No model at {MODEL_PATH} — train first."); sys.exit(1)

print(f"📦 Loading model: {MODEL_PATH.name}")
print(f"🏗️  Re-building Iron Oracle...")
features  = build_feature_cols()
n_feats   = len(features)
model = build_kraken(n_features=n_feats, context_window=CTX_WIN, forecast_steps=15)
model.load_weights(str(MODEL_PATH))
print(f"✅ Model loaded successfully")

print(f"\n⚓ RUNNING LIVE-LOGIC SIMULATOR — {SYMBOL}")
print("="*80)

# Fetch data: needs extra historical context for preprocessing and future outlook
df = fetch_live_kat_data(symbol=SYMBOL, n_candles=N_CANDLES + CTX_WIN + 50, timeframe=TIMEFRAME)
print(f"   Got {len(df):,} candles")
df_feat = compute_indicators(df)
data    = df_feat[features].values.astype("float32")
close_col = features.index("close")
high_col  = features.index("high")
low_col   = features.index("low")

# Cache pandas columns for ATR calculations
df_close = df_feat['close'].values
df_high  = df_feat['high'].values
df_low   = df_feat['low'].values

def get_dynamic_min_swing_sim(idx) -> float:
    """Helper to compute dynamic ATR swing floor at historical index `idx`"""
    try:
        high = df_high[idx-14:idx]
        low  = df_low[idx-14:idx]
        close_prev = df_close[idx-15:idx-1]
        tr = np.maximum.reduce([
            high - low,
            np.abs(high - close_prev),
            np.abs(low  - close_prev)
        ])
        atr = float(np.mean(tr))
        return float(np.clip(atr * 2.0, MIN_SWING_FLOOR, MIN_SWING_CEIL))
    except Exception:
        return MIN_SWING_FLOOR

# Position states
# None or {"side": "LONG"/"SHORT", "entry_price": float, "peak_price": float, "breakeven": bool, "sl": float, "tp": float}
active_position = None
last_trade_idx = -9999
trades_log = []
daily_trade_counts = [] # list of indices where trades occurred to enforce 4 trades max per 96 bars (24h)

print(f"🔄 Simulating walk-forward steps with live_trader sniper gates...")

for i in range(CTX_WIN, len(data) - 15):
    current_price = data[i, close_col]
    current_high  = data[i, high_col]
    current_low   = data[i, low_col]
    
    # ── 1. ACTIVE POSITION MANAGEMENT ──
    if active_position is not None:
        pos = active_position
        is_long = (pos["side"] == "LONG")
        
        # Check Exits via High/Low of the current bar
        if is_long:
            # Check Stop Loss
            if current_low <= pos["sl"]:
                # Stopped out
                pnl_pct = (pos["sl"] - pos["entry_price"]) / pos["entry_price"] - FEE_RATE * 2
                trades_log.append({
                    "type": "SL", "side": "LONG", "entry": pos["entry_price"], "exit": pos["sl"], 
                    "pnl_pct": pnl_pct, "entry_step": pos["entry_step"], "exit_step": i
                })
                active_position = None
                continue
            # Check Take Profit
            elif current_high >= pos["tp"]:
                # Profit target hit
                pnl_pct = (pos["tp"] - pos["entry_price"]) / pos["entry_price"] - FEE_RATE * 2
                trades_log.append({
                    "type": "TP", "side": "LONG", "entry": pos["entry_price"], "exit": pos["tp"], 
                    "pnl_pct": pnl_pct, "entry_step": pos["entry_step"], "exit_step": i
                })
                active_position = None
                continue
            
            # Update position state & trailing stops
            if current_high > pos["peak_price"]:
                pos["peak_price"] = current_high
            
            profit_pct = (current_price - pos["entry_price"]) / pos["entry_price"] * 100
            
            # Breakeven Activation
            if profit_pct >= BREAKEVEN_TRIGGER_PCT and not pos["breakeven"]:
                pos["breakeven"] = True
                pos["sl"] = pos["entry_price"]
                
            # Trailing Stop Update
            if profit_pct >= 0.5:
                target_ts = pos["peak_price"] * (1 - TRAILING_STOP_PCT/100)
                if target_ts > pos["sl"]:
                    pos["sl"] = target_ts
                    pos["tp"] = pos["peak_price"] * (1 + 2.8/100)
                    
        else: # SHORT position
            # Check Stop Loss
            if current_high >= pos["sl"]:
                # Stopped out
                pnl_pct = (pos["entry_price"] - pos["sl"]) / pos["entry_price"] - FEE_RATE * 2
                trades_log.append({
                    "type": "SL", "side": "SHORT", "entry": pos["entry_price"], "exit": pos["sl"], 
                    "pnl_pct": pnl_pct, "entry_step": pos["entry_step"], "exit_step": i
                })
                active_position = None
                continue
            # Check Take Profit
            elif current_low <= pos["tp"]:
                # Profit target hit
                pnl_pct = (pos["entry_price"] - pos["tp"]) / pos["entry_price"] - FEE_RATE * 2
                trades_log.append({
                    "type": "TP", "side": "SHORT", "entry": pos["entry_price"], "exit": pos["tp"], 
                    "pnl_pct": pnl_pct, "entry_step": pos["entry_step"], "exit_step": i
                })
                active_position = None
                continue
                
            # Update position state & trailing stops
            if current_low < pos["peak_price"]:
                pos["peak_price"] = current_low
            
            profit_pct = (pos["entry_price"] - current_price) / pos["entry_price"] * 100
            
            # Breakeven Activation
            if profit_pct >= BREAKEVEN_TRIGGER_PCT and not pos["breakeven"]:
                pos["breakeven"] = True
                pos["sl"] = pos["entry_price"]
                
            # Trailing Stop Update
            if profit_pct >= 0.5:
                target_ts = pos["peak_price"] * (1 + TRAILING_STOP_PCT/100)
                if target_ts < pos["sl"]:
                    pos["sl"] = target_ts
                    pos["tp"] = pos["peak_price"] * (1 - 2.8/100)

    # If we are in an active position, we skip checking new entries (live_trader monitors existing)
    if active_position is not None:
        continue

    # ── 2. NEW SIGNAL ENTRY CHECKS ──
    
    # Daily circuit breaker: max 4 trades per 24 hours (96 bars of 15m)
    recent_trades = [t_idx for t_idx in daily_trade_counts if i - t_idx < 96]
    if len(recent_trades) >= MAX_TRADES_PER_DAY:
        continue
        
    # Cooldown check
    if (i - last_trade_idx) < COOLDOWN_BARS:
        continue

    # Run inference
    X_in = apply_dls(data[i - CTX_WIN : i])[0].reshape(1, CTX_WIN, n_feats)
    out = model(X_in, training=False)
    pred         = out[0].numpy()[0]
    certainty_2d = out[1].numpy()[0]
    reasoning    = int(np.argmax(out[2].numpy()[0]))
    
    pred_future  = pred[1:]
    p_anchor     = pred[0, 0]
    p_curve      = pred_future[:, 0]
    v_curve      = pred_future[:, 1]
    p_change     = p_curve - p_anchor
    mean_price   = np.mean(p_change)
    mean_vol     = np.mean(v_curve)
    
    cert_norm = float(np.mean(certainty_2d))
    
    # Local standard deviation forclose price
    l_std = np.std(data[i - CTX_WIN : i, close_col])
    est_swing = abs(mean_price) * l_std
    
    dyn_min_swing = get_dynamic_min_swing_sim(i)
    
    # Gate 1: Certainty
    if cert_norm < CERT_THRESHOLD:
        continue
        
    # Gate 2: Reasoning (must be SOVEREIGN_LONG or SOVEREIGN_SHORT)
    if reasoning not in [0, 1]:
        continue
        
    # Gate 3: Fee Protection (expected swing size)
    if est_swing < dyn_min_swing:
        continue

    # Gate 4: Signal threshold check
    dynamic_thresh = THRESHOLD * (1.0 + max(0.0, mean_vol))
    
    # Place trade!
    base_sl = 1.2
    base_tp = 2.8
    vol_multiplier = 1.0 + min(0.5, abs(mean_vol))
    dyn_sl_pct = base_sl * vol_multiplier
    dyn_tp_pct = base_tp * vol_multiplier
    
    if mean_price > dynamic_thresh:
        # Enter LONG
        sl_price = current_price * (1 - dyn_sl_pct/100)
        tp_price = current_price * (1 + dyn_tp_pct/100)
        active_position = {
            "side": "LONG", "entry_price": current_price, "peak_price": current_price,
            "breakeven": False, "sl": sl_price, "tp": tp_price, "entry_step": i
        }
        last_trade_idx = i
        daily_trade_counts.append(i)
        
    elif mean_price < -dynamic_thresh:
        # Enter SHORT
        sl_price = current_price * (1 + dyn_sl_pct/100)
        tp_price = current_price * (1 - dyn_tp_pct/100)
        active_position = {
            "side": "SHORT", "entry_price": current_price, "peak_price": current_price,
            "breakeven": False, "sl": sl_price, "tp": tp_price, "entry_step": i
        }
        last_trade_idx = i
        daily_trade_counts.append(i)

# If still in position at the end, close it at current closing price
if active_position is not None:
    pos = active_position
    last_idx = len(data) - 16
    exit_p = data[last_idx, close_col]
    if pos["side"] == "LONG":
        pnl_pct = (exit_p - pos["entry_price"]) / pos["entry_price"] - FEE_RATE * 2
    else:
        pnl_pct = (pos["entry_price"] - exit_p) / pos["entry_price"] - FEE_RATE * 2
    trades_log.append({
        "type": "FORCE_CLOSE", "side": pos["side"], "entry": pos["entry_price"], "exit": exit_p, 
        "pnl_pct": pnl_pct, "entry_step": pos["entry_step"], "exit_step": last_idx
    })

# ── SIMULATION REPORT ─────────────────────────────────────────────────────────
df_trades = pd.DataFrame(trades_log)

print("\n" + "="*80)
print("📊 SOVEREIGN LIVE-LOGIC SIMULATED BACKTEST REPORT (V6.0)")
print("="*80)
print(f"  Symbol     : {SYMBOL} {TIMEFRAME}")
print(f"  Model      : {MODEL_PATH.name}")
print(f"  Window     : {N_CANDLES:,} candles (~{N_CANDLES/96:.1f} days)")
print(f"  Rules      : Exact live_trader.py Sniper Gates + SL/TP Bracket Dynamics")
print("-"*80)

if len(df_trades) == 0:
    print("  No trades executed under current sniper logic constraints.")
else:
    win_trades = df_trades[df_trades["pnl_pct"] > 0]
    win_rate = len(win_trades) / len(df_trades) * 100
    long_trades = df_trades[df_trades["side"] == "LONG"]
    short_trades = df_trades[df_trades["side"] == "SHORT"]
    long_wr = (len(long_trades[long_trades["pnl_pct"] > 0]) / len(long_trades) * 100) if len(long_trades) > 0 else 0
    short_wr = (len(short_trades[short_trades["pnl_pct"] > 0]) / len(short_trades) * 100) if len(short_trades) > 0 else 0
    
    total_net_pnl = df_trades["pnl_pct"].sum() * 100  # in terms of %
    cum_pnl = df_trades["pnl_pct"].cumsum() * 100
    max_dd = (cum_pnl - cum_pnl.cummax()).min()
    if pd.isna(max_dd) or max_dd > 0: max_dd = 0.0
    sharpe = df_trades["pnl_pct"].mean() / (df_trades["pnl_pct"].std() + 1e-9)
    
    tp_hits = len(df_trades[df_trades["type"] == "TP"])
    sl_hits = len(df_trades[df_trades["type"] == "SL"])
    fc_hits = len(df_trades[df_trades["type"] == "FORCE_CLOSE"])

    print(f"  Total Trades : {len(df_trades)} trades (Avg: {len(df_trades)/(N_CANDLES/96):.1f} trades/day)")
    print(f"  Win Rate     : {win_rate:.1f}%  (Long WR: {long_wr:.1f}% | Short WR: {short_wr:.1f}%)")
    print(f"  Net P&L (%)  : {total_net_pnl:+.2f}% cumulative ({total_net_pnl/len(df_trades):+.2f}% average per trade)")
    print(f"  Max Drawdown : {max_dd:.2f}%")
    print(f"  Sharpe Ratio : {sharpe * np.sqrt(365 * 96 / len(df_trades) if len(df_trades) > 0 else 1):.3f}")
    print(f"  Exit Types   : TP: {tp_hits} | SL: {sl_hits} | Force Close: {fc_hits}")
print("="*80 + "\n")
