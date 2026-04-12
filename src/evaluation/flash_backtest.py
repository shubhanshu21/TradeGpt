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
    model = build_kraken(27, 120, 15)
    model.load_weights(str(ROOT / "models/hydra_best.keras"))
    scaler = KATScaler.load(str(ROOT / "models/scaler_base.pkl"))
    
    df = fetch_live_kat_data("BTCUSD", 300, "1m")
    df_feat = compute_indicators(df)
    features = build_feature_cols()
    close_idx = features.index("close")
    
    data = df_feat[features].values.astype("float32")
    scaled = scaler.transform_X(data)
    
    results = []
    print(f"🔬 Evaluating {len(scaled) - 135} windows...")
    for i in range(120, len(scaled) - 15):
        X_in = scaled[i - 120 : i].reshape(1, 120, 27)
        out = model(X_in, training=False)
        pred = out[0].numpy()[0] # (16, 3)
        
        # Predicted move at T+15
        p_15 = pred[15, 0]
        dir_pred = np.sign(p_15)
        
        # Actual move at T+15
        v_now = scaled[i, close_idx]
        v_15  = scaled[i+15, close_idx]
        dir_actual = np.sign(v_15 - v_now)
        
        results.append(dir_pred == dir_actual)
    
    acc = np.mean(results) * 100
    print(f"\n======================================")
    print(f"📊 FLASH RESULTS (T+15 TREND): {acc:.2f}% Accuracy")
    print(f"   Steps: {len(results)} | Threshold: 0.0")
    print(f"======================================\n")

if __name__ == "__main__":
    run_flash()
