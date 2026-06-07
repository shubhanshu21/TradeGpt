"""
Run after a checkpoint exists to find the right CERT_THRESHOLD.
Usage: python scripts/check_cert_threshold.py
"""
import sys, os
sys.path.insert(0, '/var/www/html/ML/kat/src')
sys.path.insert(0, '/var/www/html/ML/kat')
os.environ["PYTHONUNBUFFERED"] = "1"

import numpy as np
import keras
from pathlib import Path
from core.hydra import build_kraken, SovereignLoss, certainty_loss, CertaintyMetric, SovereignAccuracy
from data.preprocess import build_feature_cols, compute_indicators, apply_dls
from exchange.fetch_data import fetch_live_kat_data

ROOT = Path('/var/www/html/ML/kat')
MODELS_DIR = ROOT / 'models'

# Load latest checkpoint
checkpoints = sorted(MODELS_DIR.glob("hydra_checkpoint_E*.keras"), reverse=True)
best = MODELS_DIR / "hydra_best.keras"
model_path = checkpoints[0] if checkpoints else (best if best.exists() else None)

if model_path is None:
    print("No checkpoint found yet. Run after epoch 1 completes.")
    sys.exit(0)

print(f"Loading: {model_path.name}")
model = build_kraken()
model.load_weights(str(model_path))

# Fetch ~500 recent candles and run inference on sliding windows
features = build_feature_cols()
df = fetch_live_kat_data('BTCUSD', 650, '15m')
from data.preprocess import compute_indicators
df_feat = compute_indicators(df)
data = df_feat[features].values.astype('float32')

ctx = 120
certs = []
for i in range(len(data) - ctx):
    x_raw = data[i:i+ctx]
    x_scaled, _, _ = apply_dls(x_raw)
    out = model(x_scaled[np.newaxis], training=False)
    cert = float(np.mean(out[1].numpy()[0]))
    certs.append(cert)

certs = np.array(certs)
print(f"\nCertainty distribution over {len(certs)} windows:")
print(f"  Min:    {certs.min():.4f}")
print(f"  Mean:   {certs.mean():.4f}")
print(f"  Median: {np.median(certs):.4f}")
print(f"  Max:    {certs.max():.4f}")
print(f"\nPercentiles:")
for p in [50, 60, 70, 75, 80, 85, 90, 95]:
    v = np.percentile(certs, p)
    print(f"  p{p:2d}: {v:.4f}  → {len(certs[certs>=v])} windows would trade")

print(f"\nCurrent CERT_THRESHOLD = 0.85")
print(f"  Would allow: {(certs >= 0.85).sum()} / {len(certs)} windows ({100*(certs>=0.85).mean():.1f}%)")
print(f"\nRecommended threshold: p80 = {np.percentile(certs, 80):.3f}")
