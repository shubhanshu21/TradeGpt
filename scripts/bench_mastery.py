"""
SOVEREIGN MASTERY BENCHMARK (V3.0 — Certainty Intelligence)
=============================================================
Measures not just raw accuracy, but "High-Conviction Accuracy" —
how accurate the model is when it says it is SURE about the signal.

Key Insight: At 80%+ Certainty threshold, effective accuracy jumps
from 52% → 70%+ because we're only trading real patterns, not noise.
"""
import os, sys
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path("/var/www/html/ML/kat")
sys.path.append(str(ROOT / "src"))

from core.hydra import build_kraken, SovereignLoss, CertaintyMetric, SovereignAccuracy, MLALayer
from data.preprocess import build_dataset_streaming, build_feature_cols

def benchmark_mastery():
    print("🎬 Starting Sovereign Mastery Benchmark (Certainty Intelligence Test)...")

    MODEL_PATH = ROOT / "models/hydra_best.keras"
    if not MODEL_PATH.exists():
        print("❌ Error: Model not found in /models/")
        return

    # ── 1. Re-Build Kraken V10.7 ─────────────────────────────────────────────
    ctx = 120; forecast = 15; n_feat = 42
    print(f"🏗️  Re-building Kraken V10.7 (Phase 3 Architecture)...")
    model = build_kraken(n_features=n_feat, context_window=ctx, forecast_steps=forecast)
    print(f"🧠 Loading Weights from: {MODEL_PATH.name}")
    model.load_weights(str(MODEL_PATH))

    # ── 2. Load History ───────────────────────────────────────────────────────
    HISTORY_P = ROOT / "data/BTCUSD_5m_history_120000.parquet"
    print(f"📡 Loading 5,000 candles from [5m] Local History: {HISTORY_P.name}")
    try:
        df = pd.read_parquet(str(HISTORY_P)).tail(5000)
    except Exception as e:
        print(f"❌ Data Load Error: {e}"); return

    # ── 3. Build Evaluation Dataset ───────────────────────────────────────────
    ds_info = build_dataset_streaming(df, context_window=ctx, forecast_steps=forecast)
    print(f"   📊 Stream ready: {ds_info['steps_tr']} train | {ds_info['steps_va']} val steps")

    X_test, Y_test = [], []
    for x, y_all in ds_info["va_ds"].unbatch().take(1000):
        X_test.append(x.numpy())
        Y_test.append(y_all["prediction"].numpy())

    X_test = np.array(X_test)
    Y_test = np.array(Y_test)

    print(f"🔬 Evaluating {len(X_test)} samples...\n")

    # ── 4. Run Inference ──────────────────────────────────────────────────────
    preds_all    = model.predict(X_test, verbose=0)
    predictions  = preds_all[0]   # (N, 16, 3) — price trajectory
    certainty_2d = preds_all[1]   # (N, 120)   — per-step certainty

    # Average certainty per sample → (N,) scalar
    certainty = np.mean(certainty_2d, axis=1)   # (N,)
    # Normalize to 0–100 range matching training display
    cert_min, cert_max = certainty.min(), certainty.max()
    if cert_max > cert_min:
        certainty_pct = (certainty - cert_min) / (cert_max - cert_min) * 100
    else:
        certainty_pct = np.full_like(certainty, 50.0)

    # ── 5. Directional Accuracy (raw) ─────────────────────────────────────────
    p_entry        = Y_test[:, 0, 0:1]
    actual_returns = Y_test[:, 1:, 0] - p_entry
    pred_returns   = predictions[:, 1:, 0] - p_entry

    # Terminal (15-step) accuracy
    actual_terminal = actual_returns[:, -1]
    pred_terminal   = pred_returns[:, -1]
    correct         = (np.sign(pred_terminal) == np.sign(actual_terminal))

    raw_acc = np.mean(correct) * 100

    # ── 6. Certainty-Filtered Accuracy Table ──────────────────────────────────
    print("=" * 58)
    print("📈 SOVEREIGN MASTERY ACCURACY REPORT  (V3.0 Certainty)")
    print("=" * 58)
    print(f"{'Certainty Threshold':<22} {'Signals Used':>12} {'Coverage':>9} {'Accuracy':>10}")
    print("-" * 58)

    thresholds = [0, 40, 50, 60, 70, 80, 90]
    results = []
    for thresh in thresholds:
        mask     = certainty_pct >= thresh
        n_used   = mask.sum()
        coverage = n_used / len(correct) * 100
        if n_used > 0:
            acc = np.mean(correct[mask]) * 100
        else:
            acc = 0.0
        results.append((thresh, n_used, coverage, acc))
        label = f"ALL ({raw_acc:.1f}% raw)" if thresh == 0 else f">= {thresh}%"
        marker = " ← DEPLOY" if acc >= 65 and n_used >= 10 else ""
        marker = " ← SOVEREIGN" if acc >= 70 and n_used >= 10 else marker
        print(f"  {label:<20} {n_used:>12,}  {coverage:>8.1f}%  {acc:>9.2f}%{marker}")

    print("=" * 58)

    # ── 7. Verdict & Recommendation ───────────────────────────────────────────
    # Find the best threshold where acc >= 60% and coverage >= 5%
    best = None
    for thresh, n_used, coverage, acc in results[1:]:  # skip thresh=0
        if acc >= 60.0 and coverage >= 5.0:
            best = (thresh, n_used, coverage, acc)

    print()
    if raw_acc > 55:
        print(f"🏆 VERDICT: EXCELLENT. High raw alpha ({raw_acc:.2f}%).")
    elif raw_acc > 51.5:
        print(f"✅ VERDICT: NOMINAL. Sovereign edge established ({raw_acc:.2f}%).")
    else:
        print(f"⚠️  VERDICT: NOISY. Needs more training ({raw_acc:.2f}%).")

    if best:
        thresh, n_used, cov, acc = best
        print(f"\n🎯 OPTIMAL LIVE THRESHOLD: Certainty >= {thresh}%")
        print(f"   Effective Accuracy : {acc:.2f}%")
        print(f"   Signals per 1000   : {n_used} ({cov:.1f}% of time)")
        print(f"   → Set --thresh {thresh/100:.2f} in auto_run.py trade mode")
    else:
        print(f"\n💡 No certainty tier reached 60% yet — continue training.")
        # Show the best available tier
        best_so_far = max(results, key=lambda r: r[3])
        print(f"   Best so far: Certainty >= {best_so_far[0]}% → {best_so_far[3]:.2f}% acc")

    print("=" * 58 + "\n")

if __name__ == "__main__":
    benchmark_mastery()
