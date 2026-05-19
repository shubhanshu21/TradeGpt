"""
📊 SOVEREIGN KRAKEN — Live Performance Evaluator (Today's Data)
==============================================================
Loads the active Sandbox brain, fetches all real-time 15m candles from today,
runs inference at every single bar of today's session, and evaluates the exact
directional accuracy against the actual realized price changes.
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from core.hydra import build_kraken
from exchange.fetch_data import fetch_live_kat_data
from data.preprocess import build_feature_cols, compute_indicators

def evaluate_today():
    print("=" * 60)
    print("🔮 EVALUATING ACTIVE BRAIN ON TODAY'S LIVE DATA (MAY 19, 2026)")
    print("=" * 60)

    # 1. Load active sandbox model
    model_p = ROOT / "models" / "sandbox_active.keras"
    if not model_p.exists():
        print(f"❌ Active model weights not found at {model_p}")
        return

    features = build_feature_cols()
    n_feats = len(features)
    CTX_WIN = 120
    FORECAST = 15

    print("🏗️  Building neural structure...")
    model = build_kraken(n_features=n_feats, context_window=CTX_WIN)
    model.load_weights(str(model_p))
    print("✅ Model loaded successfully!")

    # 2. Fetch candles covering today
    # 24 hours * 4 candles = 96 candles + CTX_WIN + FORECAST = ~250 candles
    print("\n📡 Fetching live market stream for May 19 session...")
    df = fetch_live_kat_data(symbol="BTCUSD", n_candles=250, timeframe="15m")
    if df is None or len(df) == 0:
        print("❌ Failed to fetch candles.")
        return

    df = compute_indicators(df)
    data = df[features].values.astype("float32")
    closes = df['close'].values

    # Determine index where today starts (2026-05-19 00:00:00 UTC)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    today_start = datetime(2026, 5, 19, 0, 0, 0, tzinfo=timezone.utc)
    
    # Find indices that fall inside today
    today_indices = df[df['timestamp'] >= today_start].index.tolist()
    if not today_indices:
        print("⚠️ No candles found for today yet. Using the last 80 candles as proxy.")
        today_indices = list(range(len(df) - 80 - FORECAST, len(df) - FORECAST))

    # Filter out indices that don't have enough forward data to check realization
    eval_indices = [idx for idx in today_indices if idx >= CTX_WIN and idx < len(df) - FORECAST]

    if not eval_indices:
        print("❌ Not enough completed candles today yet to evaluate a 15-bar forecast.")
        return

    print(f"📈 Found {len(eval_indices)} complete validation steps for today's session.")
    print("🧠 Running forward neural simulation step-by-step...")

    correct_predictions = 0
    total_predictions = 0
    certainties_correct = []
    certainties_incorrect = []

    results = []

    for idx in eval_indices:
        # Context window slicing
        x_raw = data[idx - CTX_WIN + 1 : idx + 1]
        l_mean = x_raw.mean(axis=0)
        l_std = x_raw.std(axis=0) + 1e-8
        x_scaled = (x_raw - l_mean) / l_std
        X_in = x_scaled[np.newaxis].astype("float32")

        # Inference
        outputs = model(X_in, training=False)
        pred = outputs[0].numpy()[0]          # (16, 3)
        certainty_2d = outputs[1].numpy()[0]  # (120,)
        
        # Predicted price trajectory trajectory
        pred_future = pred[1:]                 # (15, 3)
        p_curve = pred_future[:, 0]
        predicted_move = np.mean(p_curve)     # Z-score mean move

        # Actual realization check
        actual_curr_p = closes[idx]
        actual_future_prices = closes[idx + 1 : idx + FORECAST + 1]
        actual_mean_future = np.mean(actual_future_prices)
        actual_move = actual_mean_future - actual_curr_p

        # Classify direction
        pred_dir = 1 if predicted_move > 0 else -1
        actual_dir = 1 if actual_move > 0 else -1

        is_correct = (pred_dir == actual_dir)
        cert_pct = float(np.mean(certainty_2d)) * 100

        if is_correct:
            correct_predictions += 1
            certainties_correct.append(cert_pct)
        else:
            certainties_incorrect.append(cert_pct)

        total_predictions += 1

        results.append({
            "Time": df['timestamp'].iloc[idx].strftime("%H:%M UTC"),
            "Price": f"${actual_curr_p:,.2f}",
            "Pred Move": f"{'+' if pred_dir > 0 else ''}{predicted_move:.2f}σ",
            "Actual Move": f"{'+' if actual_move > 0 else ''}${actual_move:.2f}",
            "Certainty": f"{cert_pct:.1f}%",
            "Result": "🟢 CORRECT" if is_correct else "🔴 INCORRECT"
        })

    # Summary calculations
    acc = (correct_predictions / total_predictions) * 100 if total_predictions > 0 else 0
    avg_cert_c = np.mean(certainties_correct) if certainties_correct else 0
    avg_cert_i = np.mean(certainties_incorrect) if certainties_incorrect else 0

    # Print Table
    print("\n" + "=" * 85)
    print(f"{'TIMESTAMP':<12} | {'BTC PRICE':<12} | {'PREDICTED SWING':<15} | {'REALIZED MOVE':<15} | {'CERTAINTY':<10} | {'OUTCOME':<12}")
    print("=" * 85)
    for r in results[-15:]:  # Print the last 15 steps for readability
        print(f"{r['Time']:<12} | {r['Price']:<12} | {r['Pred Move']:<15} | {r['Actual Move']:<15} | {r['Certainty']:<10} | {r['Result']:<12}")
    print("=" * 85)

    print("\n📊 SUMMARY PERFORMANCE METRICS FOR TODAY:")
    print("=" * 60)
    print(f"   Total Evaluated Steps : {total_predictions}")
    print(f"   Correct Predictions   : {correct_predictions}")
    print(f"   Directional Accuracy  : {acc:.2%}" if total_predictions > 0 else "   Directional Accuracy  : 0.00%")
    print(f"   Avg Certainty (Wins)  : {avg_cert_c:.2f}%")
    print(f"   Avg Certainty (Losses): {avg_cert_i:.2f}%")
    print("=" * 60)

if __name__ == "__main__":
    evaluate_today()
