import sys, os
from pathlib import Path
import numpy as np
import tensorflow as tf

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

from core.hydra import build_kraken
from data.preprocess import KATScaler, build_feature_cols, compute_indicators, apply_dls
from exchange.fetch_data import fetch_live_kat_data

def run_flash():
    print("⚡ FLASH BACKTEST (100 STEPS) STARTED...")
    features    = build_feature_cols()
    n_feat      = len(features)
    models_dir  = ROOT / "models"
    checkpoints = sorted(models_dir.glob("hydra_checkpoint_E*.keras"), reverse=True)
    model_p     = checkpoints[0] if checkpoints else models_dir / "hydra_best.keras"
    if not model_p.exists():
        print(f"❌ No model found in {models_dir}"); return
    print(f"🧠 Loading: {model_p.name} | Features: {n_feat}")
    model = build_kraken(n_feat, 120, 15)
    model.load_weights(str(model_p))
    
    df = fetch_live_kat_data("BTCUSD", 300, "15m")  # Use 15m to match training
    df_feat = compute_indicators(df)
    close_idx = features.index("close")
    data = df_feat[features].values.astype("float32")
    # DLS — match training pipeline
    # Actuals are compared from raw close prices (sign of price move)
    close_idx = features.index("close")
    preds       = []
    actuals     = []
    certainties = []
    
    ctx = 120
    print(f"🔬 Evaluating {len(data) - ctx - 15} windows...")
    for i in range(ctx, len(data) - 15):
        X_in = apply_dls(data[i - ctx : i])[0].reshape(1, ctx, n_feat)
        out = model(X_in, training=False)
        
        # Predicted direction at T+15 relative to anchor (T+0)
        pred_vals = out[0].numpy()[0]
        p_anchor = pred_vals[0, 0]   # z-score at anchor step
        p_15     = pred_vals[15, 0]  # z-score at step 15
        preds.append(np.sign(p_15 - p_anchor))  # direction of change
        
        # Certainty (Channel 1)
        cert_val = np.mean(out[1].numpy())
        certainties.append(cert_val)
        
        # Actual direction at T+15 from raw (unscaled) close prices
        v_now = data[i,      close_idx]
        v_15  = data[i + 15, close_idx]
        actuals.append(np.sign(v_15 - v_now))
        
        if len(preds) % 20 == 0:
            print(f"   Step {len(preds):3d}...")

    preds       = np.array(preds)
    actuals     = np.array(actuals)
    certainties = np.array(certainties)
    
    # Normalize certainty 0-100
    c_min, c_max = certainties.min(), certainties.max()
    c_pct = (certainties - c_min) / (c_max - c_min + 1e-9) * 100
    
    print(f"\n======================================")
    print(f"📊 FLASH RESULTS (Filtered by Certainty)")
    print(f"--------------------------------------")
    for th in [0, 50, 80, 90]:
        mask = c_pct >= th
        hits = (preds[mask] == actuals[mask])
        acc  = np.mean(hits) * 100 if len(hits) > 0 else 0
        print(f"🎯 Threshold {th}% | Acc: {acc:6.2f}% | Trades: {len(hits)}")
    print(f"======================================\n")

if __name__ == "__main__":
    run_flash()
