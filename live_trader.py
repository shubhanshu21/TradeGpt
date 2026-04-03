#!/usr/bin/env python3
"""
KAT Live Sandbox Executor — Production Grade
===========================================
- Cycles: 60s
- Infrastructure: HYDRA Engine (MoE + MLA + MTP)
- Risk Management: Dynamic ATR-based Stop-Loss/Take-Profit
"""

import os, sys, time
import numpy as np
import tensorflow as tf
import keras
from pathlib import Path

# Fix paths
ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "src/architectures"
SAVED_MODELS = ROOT / "models"

sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(MODELS_DIR))

from delta_client import DeltaClient
from fetch_data    import fetch_live_kat_data
from preprocess    import build_dataset, KATScaler

# ── CONFIG ───────────────────────────────────────────────────────────────────
SYMBOL     = "BTCUSD"
SIZE       = 1    
THRESHOLD  = 0.05 
INTERVAL   = 60   
MODEL_NAME = "hydra"

# ── PERFORMANCE TRACKING ─────────────────────────────────────────────────────
START_EQUITY = None
HWM          = 0.0

# ── COLORS ───────────────────────────────────────────────────────────────────
C_RESET = "\033[0m"
C_BOLD  = "\033[1m"
C_GREEN = "\033[32m"
C_RED   = "\033[31m"
C_CYAN  = "\033[36m"
C_YELLOW = "\033[33m"
C_BLUE   = "\033[34m"

def calculate_atr(df, window=14):
    """Calculates Average True Range for dynamic risk management."""
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    return true_range.rolling(window=window).mean().iloc[-1]

def run_trader():
    print("="*60)
    print(f"  HYDRA Engine — SANDBOX EXECUTOR [ {SYMBOL} ]")
    print("="*60)

    # 1. Initialize Client & Model
    client = DeltaClient(testnet=True)
    
    scaler_path = SAVED_MODELS / "scaler_base.pkl"
    if not scaler_path.exists():
        print("! Error: Scaler not found. Train the model first.")
        return
    scaler = KATScaler.load(str(scaler_path))

    model_path = SAVED_MODELS / f"{MODEL_NAME}_final.keras"
    if not model_path.exists():
        print(f"! Error: Model not found at {model_path}.")
        return
    
    # Ensure custom modules are in path
    if   "alpha" in MODEL_NAME: import alpha
    elif "titan" in MODEL_NAME: import titan
    elif "causal" in MODEL_NAME: import causal
    elif "hydra" in MODEL_NAME: import hydra

    print(f"Loading {model_path}...")
    model = keras.models.load_model(str(model_path))

    global START_EQUITY, HWM
    print(f"\n✓ Engine started. Polling every {INTERVAL}s...")
    
    import pandas as pd
    
    while True:
        try:
            # 2. Position Check
            positions = client.get_positions()
            active_p = next((p for p in positions if p.get("product_symbol") == SYMBOL), None)
            pos_upnl = float(active_p.get("unrealized_pnl", 0)) if active_p else 0
            pos_color = C_GREEN if pos_upnl >= 0 else C_RED

            # 3. Performance
            balances = client._get("/v2/wallet/balances", auth=True).get("result", [])
            usd_bal = next((float(b["balance"]) for b in balances if b["asset_symbol"] == "USD"), 0)
            btc_bal = next((float(b["balance"]) for b in balances if b["asset_symbol"] == "BTC"), 0)
            
            ticker = client._get("/v2/tickers/BTCUSD", auth=False).get("result", {})
            btc_price = float(ticker.get("mark_price", 0))
            
            total_equity = usd_bal + (btc_bal * btc_price)
            if START_EQUITY is None: START_EQUITY = total_equity
            if total_equity > HWM: HWM = total_equity
            
            pnl_pct = ((total_equity - START_EQUITY) / START_EQUITY) * 100 if START_EQUITY else 0
            pnl_color = C_GREEN if pnl_pct >= 0 else C_RED
            
            # Dashboard
            print("\n" + C_CYAN + "═"*60 + C_RESET)
            print(f"  {C_BOLD}HYDRA DASHBOARD{C_RESET} | {C_YELLOW}{SYMBOL}{C_RESET} | EQUITY: {C_BOLD}${total_equity:,.2f}{C_RESET}")
            print(f"  PnL: {pnl_color}{pnl_pct:+.3f}%{C_RESET} | POS PnL: {pos_color}${pos_upnl:.4f}{C_RESET}")
            print(C_CYAN + "═"*60 + C_RESET)

            # 4. Fetch Fresh Data
            df = fetch_live_kat_data(symbol=SYMBOL, n_candles=500)
            
            # Dynamic Risk: ATR
            atr = calculate_atr(df)
            current_price = df['close'].iloc[-1]
            atr_pct = (atr / current_price) * 100
            
            sl_pct = max(1.0, min(3.0, atr_pct * 1.5)) # Dynamic SL: 1.5x ATR
            tp_pct = max(2.0, min(6.0, atr_pct * 3.0)) # Dynamic TP: 3.0x ATR
            
            print(f"   Volatility (ATR): {atr:.2f} ({atr_pct:.2f}%) | Brackets: SL={sl_pct:.2f}% TP={tp_pct:.2f}%")

            # 5. Predict
            ctx = 360 if "hydra" in MODEL_NAME else 150
            ds = build_dataset(df, context_window=ctx, forecast_steps=1, scaler=scaler)
            seed = ds["X_test"][-1:] 
            
            if "causal" in MODEL_NAME or "hydra" in MODEL_NAME:
                traj = model.generate(seed[0], steps=5, scaler=scaler)
            else:
                p_raw = model.predict(seed, verbose=0)[0]
                traj = [scaler.inverse_y(np.array([p_raw[-1] if hasattr(p_raw, "len") else p_raw]))[0]]
            
            last_p = current_price
            avg_p  = np.mean(traj)
            pct_move = ((avg_p - last_p) / last_p) * 100
            
            print(f"   Last: ${last_p:,.2f} | Forecast: ${avg_p:,.2f} ({pct_move:+.3f}%)")

            # 6. Signals
            p_size = float(active_p["size"]) if active_p else 0
            
            if pct_move > THRESHOLD:
                if p_size == 0:
                    print(f"   {C_GREEN}🚀 SIGNAL: LONG (Dynamic Brackets){C_RESET}")
                    client.place_order(SYMBOL, SIZE, "buy", sl_pct=sl_pct, tp_pct=tp_pct)
                elif active_p.get("side") == "sell":
                    client.place_order(SYMBOL, abs(p_size), "buy") # Close
                    client.place_order(SYMBOL, SIZE, "buy", sl_pct=sl_pct, tp_pct=tp_pct)
            
            elif pct_move < -THRESHOLD:
                if p_size == 0:
                    print(f"   {C_RED}🩸 SIGNAL: SHORT (Dynamic Brackets){C_RESET}")
                    client.place_order(SYMBOL, SIZE, "sell", sl_pct=sl_pct, tp_pct=tp_pct)
                elif active_p.get("side") == "buy":
                    client.place_order(SYMBOL, abs(p_size), "sell") # Close
                    client.place_order(SYMBOL, SIZE, "sell", sl_pct=sl_pct, tp_pct=tp_pct)
            
            elif p_size != 0 and abs(pct_move) < (THRESHOLD / 2):
                print(f"   {C_CYAN}💤 EXIT: Signal Faded{C_RESET}")
                close_side = "sell" if active_p.get("side") == "buy" else "buy"
                client.place_order(SYMBOL, abs(p_size), close_side)

        except Exception as e:
            print(f"   ⚠️ Loop Error: {e}")
        
        time.sleep(INTERVAL)

if __name__ == "__main__":
    (ROOT / "logs" / "plots").mkdir(parents=True, exist_ok=True)
    run_trader()
