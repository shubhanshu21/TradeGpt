import sys, os
from pathlib import Path
import numpy as np
import tensorflow as tf

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

from core.hydra import build_kraken
from data.preprocess import KATScaler, build_feature_cols, compute_indicators
from exchange.fetch_data import fetch_live_kat_data

def run_flash():
    print("⚡ FLASH BACKTEST (100 STEPS) STARTED...")
    features    = build_feature_cols()
    n_feat      = len(features)
    models_dir  = ROOT / "models"
    checkpoints = sorted(models_dir.glob("hydra_checkpoint_E*.keras"), reverse=True)
    model_p     = checkpoints[0] if checkpoints else models_dir / "hydra_best.keras"
    if not model_p.exists():
        print(f"❌ No checkpoint found in {models_dir}"); return
    print(f"🧠 Loading: {model_p.name} | Features: {n_feat}")
    model = build_kraken(n_feat, 120, 15)
    model.load_weights(str(model_p))
    
    df = fetch_live_kat_data("BTCUSD", 300, "15m")  # Use 15m to match training
    df_feat = compute_indicators(df)
    close_idx = features.index("close")
    data = df_feat[features].values.astype("float32")
    # DLS — match training pipeline (no global scaler)
    def _scale(window): return (window - window.mean(0)) / (window.std(0) + 1e-8)
    scaled = data
    
    results = []
    ctx = 120
    print(f"🔬 Evaluating {len(data) - ctx - 15} windows...")
    for i in range(ctx, len(data) - 15):
        X_in = _scale(data[i - ctx : i]).reshape(1, ctx, n_feat)
        out = model(X_in, training=False)
        pred = out[0].numpy()[0] # (16, 3)
        
        # Predicted move at T+15
        p_15 = pred[15, 0]
        dir_pred = np.sign(p_15)
        
        # Actual move at T+15
        v_now = scaled[i, close_idx]
        v_15  = scaled[i+15, close_idx]
        dir_actual = np.sign(v_15 - v_now)
        
        hit = (dir_pred == dir_actual)
        results.append(hit)
        
        # Print progress every 20 steps
        if len(results) % 20 == 0:
            moving_acc = np.mean(results) * 100
            print(f"   Step {len(results):3d}: Moving Acc: {moving_acc:5.2f}% | Last Pred: {dir_pred:2.1f} Act: {dir_actual:2.1f}")
    
    acc = np.mean(results) * 100
    print(f"\n======================================")
    print(f"📊 FLASH RESULTS (T+15 TREND): {acc:.2f}% Accuracy")
    print(f"   Steps: {len(results)} | Threshold: 0.0")
    print(f"======================================\n")

if __name__ == "__main__":
    run_flash()
