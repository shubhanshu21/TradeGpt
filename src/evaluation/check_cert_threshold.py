"""
SOVEREIGN KRAKEN — Certainty Threshold Finder ⚓📐
==================================================
Run after a checkpoint exists to find a data-driven CERT_THRESHOLD instead
of the fixed 0.85 guess — computes the real certainty distribution across
recent windows and reports what percentile each candidate threshold falls at.

Usage:
    python src/evaluation/check_cert_threshold.py
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

from core.inference import load_trained_model, run_inference
from data.preprocess import build_feature_cols, compute_indicators, apply_dls, tokenize_returns

MODELS_DIR = ROOT / "models"

try:
    model, vocab = load_trained_model(MODELS_DIR)
except FileNotFoundError as e:
    print(str(e))
    sys.exit(0)

ctx_win = vocab["context_window"]
timeframe = vocab["timeframe"]
print(f"✅ Model loaded — timeframe={timeframe}, context_window={ctx_win}")

features = build_feature_cols()
df = pd.read_parquet(ROOT / "data" / f"BTCUSD_{timeframe}_history_master.parquet")
df_feat = compute_indicators(df)
data = df_feat[features].values.astype("float32")
t_close = features.index("close")

certs = []
n_windows = min(650, len(data) - ctx_win - 1)
for i in range(len(data) - n_windows, len(data)):
    if i < ctx_win + 1:
        continue
    x_scaled, local_mean, local_std = apply_dls(data[i - ctx_win: i])
    raw_returns = np.diff(data[i - ctx_win - 1: i, t_close]) / (data[i - ctx_win - 1: i - 1, t_close] + 1e-9)
    tok_ids = tokenize_returns(raw_returns.astype("float64"), vocab["bin_edges"])
    _, cert, _, _ = run_inference(model, x_scaled, tok_ids)
    certs.append(float(np.mean(cert)))

certs = np.array(certs)
print(f"\nCertainty distribution over {len(certs)} windows:")
print(f"  Min:    {certs.min():.4f}")
print(f"  Mean:   {certs.mean():.4f}")
print(f"  Median: {np.median(certs):.4f}")
print(f"  Max:    {certs.max():.4f}")
print(f"\nPercentiles:")
for p in [50, 60, 70, 75, 80, 85, 90, 95]:
    v = np.percentile(certs, p)
    print(f"  p{p:2d}: {v:.4f}  → {len(certs[certs >= v])} windows would trade")

from config.sovereign_config import CERT_THRESHOLD
print(f"\nCurrent CERT_THRESHOLD = {CERT_THRESHOLD}")
print(f"  Would allow: {(certs >= CERT_THRESHOLD).sum()} / {len(certs)} windows "
      f"({100 * (certs >= CERT_THRESHOLD).mean():.1f}%)")
print(f"\nRecommended threshold (p80): {np.percentile(certs, 80):.3f}")
