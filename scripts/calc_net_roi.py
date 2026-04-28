import os, sys
from pathlib import Path
ROOT = Path('/var/www/html/ML/kat')
sys.path.append(str(ROOT / 'src'))
import numpy as np
import pandas as pd
from core.hydra import build_kraken
from data.preprocess import compute_indicators, build_feature_cols
from exchange.fetch_data import fetch_live_kat_data

def calc_pnl_fees():
    ctx=120; f=15; n=42; model = build_kraken(n, ctx, f)
    model.load_weights('models/hydra_best.keras')
    df_raw = fetch_live_kat_data('BTCUSD', 5000, '15m')
    df = compute_indicators(df_raw)
    raw_prices = df['close'].values
    features = build_feature_cols()
    data = df[features].values.astype('float32')
    
    indices = range(len(df) - ctx - f - 500, len(df) - ctx - f)
    X = np.array([(data[i:i+ctx] - data[i:i+ctx].mean(0)) / (data[i:i+ctx].std(0) + 1e-8) for i in indices])
    usd_diffs = np.array([raw_prices[i + ctx + f - 1] - raw_prices[i + ctx - 1] for i in indices])
    
    preds = model.predict(X, verbose=0)
    traj = preds[0][:, -1, 0]
    certs = np.mean(preds[1], axis=1)
    c_pct = (certs - certs.min()) / (certs.max() - certs.min() + 1e-9) * 100
    
    pos_size_usd = 4000.0
    fee_rate = 0.0006 # 0.06% round trip
    
    print("-" * 40)
    print("NET ROI REPORT (POST-FEES) - $200 Account")
    print("-" * 40)
    for th in [80, 85, 90]:
        mask = c_pct >= th
        n_t = mask.sum()
        if n_t > 0:
            e_p = raw_prices[np.array(indices)[mask] + ctx - 1]
            gross = (np.sign(traj[mask]) * (usd_diffs[mask] / e_p) * pos_size_usd).sum()
            fees = n_t * (pos_size_usd * fee_rate)
            net = gross - fees
            print(f"Tier {th}% | Trades: {n_t} | Gross: ${gross:.2f} | Fees: ${fees:.2f} | NET: ${net:.2f}")
    print("-" * 40)

if __name__ == "__main__":
    calc_pnl_fees()
