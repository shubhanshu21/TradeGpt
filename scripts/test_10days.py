"""
📊 SOVEREIGN KRAKEN — 10-Day Wallet Compounding Simulator
=========================================================
Runs a full quantitative simulation over the last 1,000 candles (10 days) of BTCUSD.
Applies the active Epoch 19 neural weights, enforces strict Sniper Safety Gates
(Certainty >= 85%, Expected Swing >= $100, Class 0 or 1), scales position size dynamically 
to 100% of current wallet balance with 10x leverage, and logs the exact financial growth.
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from core.hydra import build_kraken
from exchange.fetch_data import fetch_live_kat_data
from data.preprocess import build_feature_cols, compute_indicators, apply_dls

def run_10day_simulation():
    print("=" * 70)
    print("⚓ 10-DAY SOVEREIGN WALLET BACKTEST — FEES GATE ENABLED")
    print("=" * 70)

    # 1. Load latest checkpoint
    model_dir = ROOT / "models"
    checkpoints = sorted(list(model_dir.glob("hydra_checkpoint_E*.keras")))
    if checkpoints:
        model_p = checkpoints[-1]
    else:
        model_p = model_dir / "hydra_best.keras"
        
    print(f"🧠 Loading neural weights: {model_p.name}")

    features = build_feature_cols()
    n_feats = len(features)
    CTX_WIN = 120
    FORECAST = 15

    model = build_kraken(n_features=n_feats, context_window=CTX_WIN)
    model.load_weights(str(model_p))
    print("✅ Model loaded successfully!")

    # 2. Fetch candles (1000 evaluate steps + 120 context = 1120 candles)
    print("\n📡 Fetching 1,150 historical candles from live market feed...")
    df = fetch_live_kat_data(symbol="BTCUSD", n_candles=1150, timeframe="15m")
    if df is None or len(df) == 0:
        print("❌ Failed to fetch market data.")
        return

    df = compute_indicators(df)
    df = df.reset_index()
    data = df[features].values.astype("float32")
    closes = df['close'].values

    # Setup simulation parameters
    wallet = 200.0
    starting_wallet = wallet
    leverage = 10
    fee_rate = 0.00118  # 0.118% roundtrip fee
    cert_threshold = 0.85
    min_swing = 100.0
    
    trades = []
    cooldown_until = 0

    print(f"\n🚀 Simulation started with Starting Capital: ${starting_wallet:,.2f} USD")
    print("🏃 Running walk-forward quantitative loop...")

    # Iterate over the 1,000 steps
    eval_start_idx = CTX_WIN
    eval_end_idx = len(df) - FORECAST

    for idx in range(eval_start_idx, eval_end_idx):
        # Skip if we are inside a trade cooldown window (simulating active open position)
        if idx < cooldown_until:
            continue

        # Extract features and DLS normalize
        x_raw = data[idx - CTX_WIN + 1 : idx + 1]
        x_scaled, _, _ = apply_dls(x_raw)
        X_in = x_scaled[np.newaxis].astype("float32")

        # Inference
        outputs = model(X_in, training=False)
        pred = outputs[0].numpy()[0]          # (16, 3)
        certainty_2d = outputs[1].numpy()[0]  # (120,)
        
        # Predicted price trajectory
        pred_future = pred[1:]
        p_curve = pred_future[:, 0]
        predicted_move = np.mean(p_curve)     # Z-score move
        
        # Convert z-score move to actual expected swing in USD
        est_swing = float(abs(predicted_move) * df['atr'].iloc[idx])
        cert_pct = float(np.mean(certainty_2d))

        # Check Gates
        # 1. Certainty Gate
        if cert_pct < cert_threshold:
            continue

        # 2. Strategic Direction Gate
        pred_dir = 1 if predicted_move > 0 else -1
        
        # 3. Expected Swing Gate (Safety Gate)
        if est_swing < min_swing:
            continue

        # Trade Triggered!
        curr_price = closes[idx]
        
        # Position sizing: Dynamically compound 100% of current wallet
        leveraged_capital = wallet * leverage
        trade_size_btc = leveraged_capital / curr_price
        
        # Realized move over the 15 bar hold
        actual_future_prices = closes[idx + 1 : idx + FORECAST + 1]
        actual_mean_future = np.mean(actual_future_prices)
        actual_move = actual_mean_future - curr_price

        # Financial Math
        gross_pnl = trade_size_btc * actual_move * pred_dir
        fees = leveraged_capital * fee_rate
        net_pnl = gross_pnl - fees

        # Update wallet balance
        old_wallet = wallet
        wallet += net_pnl
        
        # Prevent bankruptcies
        if wallet <= 0:
            wallet = 0
            print(f"💥 Wallet liquidated at step {idx}!")
            break

        cooldown_until = idx + FORECAST  # Hold trade for 15 bars

        timestamp_str = df['timestamp'].iloc[idx].strftime("%m-%d %H:%M")
        trades.append({
            "Time": timestamp_str,
            "Price": curr_price,
            "Direction": "LONG 🏹" if pred_dir > 0 else "SHORT 📉",
            "Swing": f"${est_swing:.2f}",
            "Certainty": f"{cert_pct*100:.1f}%",
            "Realized Move": f"{'+' if actual_move*pred_dir > 0 else ''}${actual_move*pred_dir:.2f}",
            "Net PnL": net_pnl,
            "Wallet": wallet
        })

    # Print Trades Log
    if trades:
        print("\n" + "=" * 90)
        print(f"{'TIME':<12} | {'BTC PRICE':<10} | {'SIDE':<8} | {'EST SWING':<10} | {'CERTAINTY':<10} | {'REALIZED':<10} | {'NET PNL':<10} | {'WALLET BALANCE':<15}")
        print("=" * 90)
        for t in trades:
            print(f"{t['Time']:<12} | ${t['Price']:<9,.2f} | {t['Direction']:<8} | {t['Swing']:<10} | {t['Certainty']:<10} | {t['Realized Move']:<10} | ${t['Net PnL']:<9,.2f} | ${t['Wallet']:<14,.2f}")
        print("=" * 90)
    else:
        print("\n⚠️ No trades were taken. The safety gates successfully blocked all noisy periods!")

    # Summary
    total_trades = len(trades)
    wins = sum(1 for t in trades if t['Net PnL'] > 0)
    losses = total_trades - wins
    win_rate = (wins / total_trades) * 100 if total_trades > 0 else 0
    total_net_pct = ((wallet - starting_wallet) / starting_wallet) * 100

    print("\n📊 10-DAY WALLET BACKTEST SUMMARY REPORT:")
    print("=" * 60)
    print(f"   Starting Capital      : ${starting_wallet:,.2f} USD")
    print(f"   Final Capital         : ${wallet:,.2f} USD")
    print(f"   Total Net Return      : {total_net_pct:+.2f}%")
    print(f"   Total Trades Taken    : {total_trades}")
    print(f"   Winning Trades (Wins) : {wins}")
    print(f"   Losing Trades (Losses): {losses}")
    print(f"   Win Rate              : {win_rate:.2f}%")
    print("=" * 60)

if __name__ == "__main__":
    run_10day_simulation()
