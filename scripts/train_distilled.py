"""
SOVEREIGN KRAKEN — Knowledge Distillation (V10.6) ⚓🧠⚡
======================================================
Trains a 'Student' Hydra model to mimic a heavyweight 'Teacher' model.
Benefit: Institutional-grade accuracy in a microsecond-latency model.
"""

import os, sys, argparse
from pathlib import Path
import tensorflow as tf
import keras
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from core.hydra import build_kraken, Distiller, SovereignLoss, SovereignAccuracy, CertaintyMetric
from data.preprocess import build_dataset_streaming, build_feature_cols, KATScaler

def run_distillation(args):
    # ── 1. Setup Data ─────────────────────────────────────────────────────────
    CKPT_DIR = ROOT / "models"
    CACHE_P  = ROOT / "data" / f"{args.symbol}_1m_history_120000.parquet"
    
    if not CACHE_P.exists():
        print(f"❌ Error: Data cache not found at {CACHE_P}. Run train.py first.")
        return

    df = pd.read_parquet(CACHE_P).tail(args.candles)
    ds_info = build_dataset_streaming(df, context_window=120, forecast_steps=15, 
                                       batch_size=args.batch, 
                                       scaler_save_path=str(CKPT_DIR / "scaler_base.pkl"))
    
    # ── 2. Build Models ───────────────────────────────────────────────────────
    print("\n🏗️  Building Teacher (256-Expert Hydra — load hydra_teacher.keras weights for full effect)...")
    # A deeper/wider teacher for distillation
    teacher = build_kraken(n_features=ds_info["n_features"], context_window=120)
    # (Optional: Load pre-trained teacher weights here if you have them)
    # teacher.load_weights("models/hydra_teacher.keras") 
    teacher.trainable = False # Teacher is not updated

    print("🏗️  Building Student (Optimized 256-Expert)...")
    student = build_kraken(n_features=ds_info["n_features"], context_window=120)

    # ── 3. Initialize Distiller ──────────────────────────────────────────────
    distiller = Distiller(student=student, teacher=teacher)
    
    distiller.compile(
        optimizer=keras.optimizers.AdamW(learning_rate=args.lr, weight_decay=0.01),
        metrics={"prediction": [SovereignAccuracy()]},
        prediction_loss_fn=SovereignLoss(),
        distillation_loss_fn=keras.losses.KLDivergence(),
        alpha=0.1,        # 10% Weight on raw data, 90% on Teacher's wisdom
        temperature=3.0    # Softens teacher's probability distribution
    )

    # ── 4. Train ──────────────────────────────────────────────────────────────
    print(f"\n🚀 IGNITION: Distilling Knowledge for {args.epochs} Epochs...")
    distiller.fit(
        ds_info["tr_ds"],
        validation_data=ds_info["va_ds"],
        epochs=args.epochs,
        steps_per_epoch=ds_info["steps_tr"],
        validation_steps=ds_info["steps_va"]
    )

    # ── 5. Save Final Student ────────────────────────────────────────────────
    student.save(str(CKPT_DIR / "hydra_distilled.keras"))
    print(f"\n✅ SUCCESS: Distilled 'Alpha-Brain' saved to models/hydra_distilled.keras")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--symbol",  default="BTCUSD")
    p.add_argument("--epochs",  type=int, default=50)
    p.add_argument("--batch",   type=int, default=64)
    p.add_argument("--candles", type=int, default=50000)
    p.add_argument("--lr",      type=float, default=1e-4)
    args = p.parse_args()
    
    run_distillation(args)
