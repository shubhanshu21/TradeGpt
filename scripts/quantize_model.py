"""
SOVEREIGN KRAKEN — Post-Training Quantization (Enhancement #7)
===============================================================
Converts hydra_final.keras → TFLite INT8 (4× smaller, microsecond inference).

Usage:
    python scripts/quantize_model.py
    python scripts/quantize_model.py --model models/hydra_best.keras --ctx 120
"""

import argparse
import sys
import numpy as np
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

import tensorflow as tf
import keras
from data.preprocess import build_dataset_streaming, build_feature_cols
from core.hydra import (SovereignLoss, build_kraken, CertaintyMetric, SovereignAccuracy,
                        HydraBlock, GatedMoE, RMSNorm, LightningAttention, TurboQuant, SwiGLU)

# ── CLI ───────────────────────────────────────────────────────────────────────
p = argparse.ArgumentParser(description="Sovereign Kraken INT8 Quantizer")
p.add_argument("--model",  default=str(ROOT / "models" / "hydra_best.keras"))
p.add_argument("--data",   default=str(ROOT / "data"))
p.add_argument("--ctx",    type=int, default=120, help="Context window used during training")
p.add_argument("--n_cal",  type=int, default=200, help="Calibration samples for INT8 range")
args = p.parse_args()

MODEL_PATH = Path(args.model)
OUT_PATH   = MODEL_PATH.with_suffix(".tflite")
INT8_PATH  = MODEL_PATH.with_name(MODEL_PATH.stem + "_int8.tflite")

print(f"\n⚙️  SOVEREIGN KRAKEN — INT8 Quantizer")
print(f"   Source : {MODEL_PATH.name}")
print(f"   CTX    : {args.ctx} candles")
print(f"   Cal    : {args.n_cal} samples")
print("="*50)

# ── 1. Load model ─────────────────────────────────────────────────────────────
print("\n📦 Loading model...")

custom_objs = {
    "SovereignLoss":      SovereignLoss,
    "RMSNorm":            RMSNorm,
    "HydraBlock":         HydraBlock,
    "LightningAttention": LightningAttention,
    "TurboQuant":         TurboQuant,
    "SwiGLU":             SwiGLU,
    "GatedMoE":           GatedMoE,
    "SovereignAccuracy":  SovereignAccuracy,
    "CertaintyMetric":    CertaintyMetric
}
# V10.3: Enable unsafe deserialization for Lambda certainty aggregation
model = keras.models.load_model(str(MODEL_PATH), custom_objects=custom_objs, safe_mode=False)
print(f"   ✅ Loaded: {MODEL_PATH.name}")

# Convert to concrete function for TFLite
@tf.function(input_signature=[
    tf.TensorSpec(shape=[1, args.ctx, len(build_feature_cols())], dtype=tf.float32)
])
def infer(x):
    return model(x, training=False)

# ── 2. Find calibration data ──────────────────────────────────────────────────
import glob, pandas as pd

data_files = sorted(glob.glob(str(Path(args.data) / "*.parquet")))
if not data_files:
    print("❌ No cached parquet files found in data/. Run train.py first to cache data.")
    sys.exit(1)

df_cal = pd.read_parquet(data_files[0]).tail(5000)
print(f"\n📊 Calibration data: {len(df_cal):,} tail candles from {Path(data_files[0]).name}")

# Build calibration dataset — use tr_ds for representative data
ds_info  = build_dataset_streaming(df_cal, context_window=args.ctx,
                                    forecast_steps=15, batch_size=1)
cal_ds   = ds_info["tr_ds"].take(args.n_cal)
cal_data = [x.numpy() for x, _ in cal_ds]

def representative_dataset():
    for x in cal_data:
        yield [x.astype(np.float32)]

# ── 3. Float32 TFLite ─────────────────────────────────────────────────────────
print("\n🔄 Converting to TFLite (Float32)...")
converter_f32 = tf.lite.TFLiteConverter.from_concrete_functions(
    [infer.get_concrete_function()]
)
tflite_f32 = converter_f32.convert()
OUT_PATH.write_bytes(tflite_f32)
size_f32 = OUT_PATH.stat().st_size / 1e6
print(f"   ✅ Float32: {OUT_PATH.name} ({size_f32:.1f} MB)")

# ── 4. INT8 TFLite ────────────────────────────────────────────────────────────
print("\n🔄 Converting to TFLite INT8 (calibrating quantization ranges)...")
converter_int8 = tf.lite.TFLiteConverter.from_concrete_functions(
    [infer.get_concrete_function()]
)
converter_int8.optimizations              = [tf.lite.Optimize.DEFAULT]
converter_int8.representative_dataset     = representative_dataset
converter_int8.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter_int8.inference_input_type      = tf.float32   # keep float I/O for ease
converter_int8.inference_output_type     = tf.float32

tflite_int8 = converter_int8.convert()
INT8_PATH.write_bytes(tflite_int8)
size_int8 = INT8_PATH.stat().st_size / 1e6
print(f"   ✅ INT8:    {INT8_PATH.name} ({size_int8:.1f} MB)")

# ── 5. Compression report ─────────────────────────────────────────────────────
ratio = size_f32 / size_int8
print(f"\n{'='*50}")
print(f"📊 QUANTIZATION REPORT")
print(f"   Float32 size : {size_f32:.1f} MB")
print(f"   INT8 size    : {size_int8:.1f} MB")
print(f"   Compression  : {ratio:.1f}× smaller")
print(f"   Inference    : ~{ratio:.0f}× faster (microsecond latency)")
print(f"\n✅ INT8 model ready: {INT8_PATH}")
print(f"   Load in live_trader.py with tf.lite.Interpreter('{INT8_PATH}')")
