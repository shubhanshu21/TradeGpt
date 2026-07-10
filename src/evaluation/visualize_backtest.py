"""
SOVEREIGN KRAKEN — Prediction vs Reality Visualizer ⚓📈
=========================================================
Generates a zoomed-in plot comparing the model's forecast trajectory against
the actual realized price, saved to backtest_honesty.png.

Usage:
    python src/evaluation/visualize_backtest.py
"""
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

from core.inference import load_trained_model, run_inference
from data.preprocess import compute_indicators, build_feature_cols, apply_dls, tokenize_returns


def visualize_performance():
    print("🔬 INITIALIZING PREDICTION VISUALIZER")

    model, vocab = load_trained_model(ROOT / "models")
    ctx = vocab["context_window"]
    forecast = vocab["forecast_steps"]
    timeframe = vocab["timeframe"]
    print(f"✅ Model loaded — timeframe={timeframe}, context_window={ctx}, forecast_steps={forecast}")

    features = build_feature_cols()
    df = pd.read_parquet(ROOT / "data" / f"BTCUSD_{timeframe}_history_master.parquet")
    df_feat = compute_indicators(df)
    data = df_feat[features].values.astype("float32")
    t_close = features.index("close")

    num_windows = 150
    valid_range = len(data) - ctx - forecast
    start = valid_range - num_windows

    print(f"🔬 Generating {num_windows} predictions...")
    y_pred_usd, y_true_usd = [], []
    for idx in range(start, valid_range):
        x_scaled, local_mean, local_std = apply_dls(data[idx: idx + ctx])
        raw_returns = np.diff(data[idx - 1: idx + ctx, t_close]) / (data[idx - 1: idx + ctx - 1, t_close] + 1e-9)
        tok_ids = tokenize_returns(raw_returns.astype("float64"), vocab["bin_edges"])
        pred, _, _, _ = run_inference(model, x_scaled, tok_ids)

        pred_scaled_ret = pred[-1, 0]  # terminal step, close channel
        y_pred_usd.append((pred_scaled_ret * local_std[t_close]) + local_mean[t_close])
        y_true_usd.append(float(df_feat['close'].iloc[idx + ctx + forecast - 1]))

    y_pred_usd = np.array(y_pred_usd)
    y_true_usd = np.array(y_true_usd)

    print("📊 Crafting visualization...")
    plt.figure(figsize=(16, 8))
    plt.style.use('dark_background')
    plt.plot(y_true_usd, color='#00FFFF', linewidth=3.0, label='ACTUAL BTC (TRUTH)')
    plt.plot(y_pred_usd, color='#FFD700', linewidth=2.0, linestyle='--', label='IRON ORACLE FORECAST')

    avg_err = np.mean(np.abs(y_true_usd - y_pred_usd))
    dir_acc = np.mean(np.sign(np.diff(y_true_usd)) == np.sign(np.diff(y_pred_usd))) * 100

    plt.title(f'IRON ORACLE: ${avg_err:.2f} Avg Error | {dir_acc:.1f}% Trend Accuracy', fontsize=16, color='white')
    plt.xlabel(f'Last {num_windows} Candles ({timeframe} Resolution)', color='#888888')
    plt.ylabel('BTC Price (USD)', color='#888888')
    plt.grid(True, linestyle=':', alpha=0.2)
    plt.legend(loc='upper right')

    plot_path = ROOT / "backtest_honesty.png"
    plt.savefig(plot_path, dpi=180)
    plt.close()

    print(f"✅ VISUAL COMPLETED: {plot_path}")
    print(f"📊 FINAL STATS: MAE ${avg_err:.2f} | DIR {dir_acc:.1f}%")


if __name__ == "__main__":
    visualize_performance()
