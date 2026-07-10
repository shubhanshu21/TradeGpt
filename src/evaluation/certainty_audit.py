"""
SOVEREIGN KRAKEN — Real-Time Certainty Timeline Audit ⚓📊
==========================================================
Shows the model's certainty/reasoning output for today's candles, one row
per candle, so you can see at a glance how often it would actually fire a
trade vs. sit out.

Usage:
    python src/evaluation/certainty_audit.py
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

from core.inference import load_trained_model, run_inference
from data.preprocess import compute_indicators, build_feature_cols, apply_dls, tokenize_returns
from config.sovereign_config import CERT_THRESHOLD


def run_certainty_distribution():
    print("\n" + "=" * 60)
    print("🚀 IRON ORACLE: MULTI-CERTAINTY TIMELINE AUDIT")
    print("=" * 60)

    model, vocab = load_trained_model(ROOT / "models")
    ctx_win = vocab["context_window"]
    timeframe = vocab["timeframe"]
    print(f"✅ Model loaded — timeframe={timeframe}, context_window={ctx_win}")

    df = pd.read_parquet(ROOT / "data" / f"BTCUSD_{timeframe}_history_master.parquet")
    df_feat = compute_indicators(df)
    features = build_feature_cols()
    data = df_feat[features].values.astype("float32")
    timestamps = df_feat.index
    t_close = features.index("close")

    today_start = pd.Timestamp(datetime.now().date(), tz="UTC")
    today_indices = np.where(timestamps >= today_start)[0]
    if len(today_indices) == 0:
        print("⚠️ No data for today yet. Checking last 50 candles.")
        today_indices = range(len(data) - 50, len(data))

    print(f"\n📊 CERTAINTY DISTRIBUTION (Today: {today_start.strftime('%Y-%m-%d')})")
    print(f"{'Time (UTC)':<12} | {'Price':<10} | {'Certainty':<10} | {'Bias':<18} | {'Status'}")
    print("-" * 75)

    bias_map = {0: "LONG 🏹", 1: "SHORT 📉", 2: "FEE_TRAP ⚠️", 3: "NOISE 😴"}

    for i in today_indices:
        if i < ctx_win + 1:
            continue

        x_scaled, local_mean, local_std = apply_dls(data[i - ctx_win: i])
        raw_returns = np.diff(data[i - ctx_win - 1: i, t_close]) / (data[i - ctx_win - 1: i - 1, t_close] + 1e-9)
        tok_ids = tokenize_returns(raw_returns.astype("float64"), vocab["bin_edges"])

        pred, cert, reasoning_probs, _ = run_inference(model, x_scaled, tok_ids)
        cert_pct = float(np.mean(cert)) * 100.0
        reas_idx = int(np.argmax(reasoning_probs))

        bias_str = bias_map.get(reas_idx, "UNKNOWN")
        status = "🟢 TRADE" if (reas_idx in [0, 1] and cert_pct >= (CERT_THRESHOLD * 100)) else "⚪ SCAN"
        if reas_idx == 2: status = "🟡 TRAP"
        if reas_idx == 3: status = "🔵 NOISE"

        ts_str = timestamps[i - 1].strftime("%H:%M")
        price_str = f"${df['close'].iloc[i - 1]:,.0f}"
        print(f"{ts_str:<12} | {price_str:<10} | {cert_pct:<10.2f}% | {bias_str:<18} | {status}")

    print("=" * 60 + "\n")


if __name__ == "__main__":
    run_certainty_distribution()
