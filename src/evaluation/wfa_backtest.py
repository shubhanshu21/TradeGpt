"""
SOVEREIGN KRAKEN — Walk-Forward Fine-Tuning Simulator ⚓🔬
=========================================================
Simulates live production by iteratively predicting on out-of-sample data chunks,
and then fine-tuning the model on those chunks as if a trading day had concluded.

Usage:
    python src/evaluation/wfa_backtest.py
"""

import sys, shutil
import numpy as np
import pandas as pd
import tensorflow as tf
import keras
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

from core.inference import load_trained_model, run_inference
from core.hydra import SovereignLoss, CertaintyMetric, SovereignAccuracy, certainty_loss
from data.preprocess import build_feature_cols, compute_indicators, apply_dls, tokenize_returns, build_dataset_streaming
from config.sovereign_config import FEE_RATE

# ── CONFIG ────────────────────────────────────────────────────────────────────
TEST_CANDLES = 2880       # Approx test window (candle-count, not calendar days)
CHUNK_SIZE   = 96         # Candles per fine-tune chunk
FT_EPOCHS    = 3          # Epochs per fine-tune step
LR           = 1e-6       # Fine-tuning learning rate
THRESHOLD    = 0.15       # Trade trigger Z-score
FEE_PCT      = FEE_RATE
FREEZE_BELOW = 6          # Freeze bottom blocks during fine-tuning

# ── 1. Load base model + real trained shape ───────────────────────────────────
MODELS_DIR = ROOT / "models"
model, vocab = load_trained_model(MODELS_DIR)
CTX_WIN   = vocab["context_window"]
FORECAST  = vocab["forecast_steps"]
TIMEFRAME = vocab["timeframe"]
n_feats   = vocab["n_features"]
print(f"✅ Model loaded — timeframe={TIMEFRAME}, context_window={CTX_WIN}, forecast_steps={FORECAST}")

WFA_MODEL_PATH = MODELS_DIR / "wfa_sandbox.keras"
print(f"📦 Saving sandbox clone -> {WFA_MODEL_PATH.name}")
model.save(str(WFA_MODEL_PATH))

# Freeze bottom blocks for fine-tuning stability
frozen = 0
for layer in model.layers:
    parts = layer.name.split("_")
    if parts[0] == "hydra" and len(parts) == 2 and parts[1].isdigit():
        idx = int(parts[1])
        if idx < FREEZE_BELOW:
            layer.trainable = False
            frozen += 1
print(f"🔒 Frozen {frozen} core blocks for fine-tuning stability.")

# ── 2. Load real cached data (not a fresh live fetch — deterministic/repeatable) ──
features = build_feature_cols()
df = pd.read_parquet(ROOT / "data" / f"BTCUSD_{TIMEFRAME}_history_master.parquet")
df_feat = compute_indicators(df)
data = df_feat[features].values.astype("float32")
t_close = features.index("close")

n_chunks = TEST_CANDLES // CHUNK_SIZE
start_idx = len(data) - TEST_CANDLES - FORECAST

results = []
print(f"\n🚀 Initiating Walk-Forward Loop: {n_chunks} chunks of {CHUNK_SIZE} candles...")

for c in range(n_chunks):
    chunk_start = start_idx + c * CHUNK_SIZE
    chunk_end   = chunk_start + CHUNK_SIZE

    print(f"\n--- [ CHUNK {c+1}/{n_chunks} ] ---")

    # ── A. Trade (Inference on Unseen Chunk) ──
    chunk_results = []
    for i in range(chunk_start, chunk_end):
        x_scaled, local_mean, local_std = apply_dls(data[i - CTX_WIN: i])
        raw_returns = np.diff(data[i - CTX_WIN - 1: i, t_close]) / (data[i - CTX_WIN - 1: i - 1, t_close] + 1e-9)
        tok_ids = tokenize_returns(raw_returns.astype("float64"), vocab["bin_edges"])
        pred, cert, _, _ = run_inference(model, x_scaled, tok_ids)

        p_anchor = pred[0, 0]
        mean_move = float(np.mean(pred[1:, 0] - p_anchor))
        mean_cert = float(np.mean(cert))

        if mean_cert < 0.85: signal = "HOLD"
        elif mean_move > THRESHOLD: signal = "LONG"
        elif mean_move < -THRESHOLD: signal = "SHORT"
        else: signal = "HOLD"

        actual_now = float(data[i - 1, t_close])
        actual_mean_future = float(np.mean(data[i: i + FORECAST, t_close]))
        actual_dir = np.sign(actual_mean_future - actual_now)

        # Naive persistence baseline: "next move repeats the last completed
        # move." The whole point of the model is to beat this for free — if
        # it can't, the extra complexity isn't buying anything.
        naive_dir = float(np.sign(data[i - 1, t_close] - data[i - 2, t_close])) if i >= 2 else 0.0

        # Local realized volatility (same std DLS already computed for scaling
        # this window) — used to check whether any edge holds across both
        # calm and turbulent regimes, or only shows up in one of them.
        vol_at_i = float(local_std[t_close])

        chunk_results.append({
            "chunk": c, "signal": signal, "actual_dir": actual_dir,
            "naive_dir": naive_dir, "vol": vol_at_i,
        })

    df_c = pd.DataFrame(chunk_results)
    trades_c = df_c[df_c["signal"] != "HOLD"]
    if len(trades_c) > 0:
        trades_c = trades_c.copy()
        trades_c["correct"] = trades_c.apply(lambda r: (1.0 if r["signal"] == "LONG" else -1.0) == r["actual_dir"], axis=1)
        c_win = trades_c["correct"].mean() * 100
        c_pnl = trades_c.apply(lambda r: (1 - FEE_PCT) if r["correct"] else (-1 - FEE_PCT), axis=1).sum()
    else:
        c_win = 0.0
        c_pnl = 0.0

    print(f"📊 Trades: {len(trades_c):<3} | Win: {c_win:>5.1f}% | Net P&L: {c_pnl:+.2f}")
    results.extend(chunk_results)

    # ── B. Fine-Tune (Learn from the Chunk) ──
    slice_start = chunk_start - CTX_WIN - 150
    slice_end   = chunk_end + FORECAST
    df_chunk = df.iloc[max(0, slice_start):slice_end]

    ds_info = build_dataset_streaming(df_chunk, context_window=CTX_WIN, forecast_steps=FORECAST, batch_size=8)

    label_counts = np.maximum(ds_info["label_counts"], 1)
    ft_total = label_counts.sum()
    ft_class_weights = {i: ft_total / (4 * label_counts[i]) for i in range(4)}
    ft_weights_tensor = tf.constant([ft_class_weights[i] for i in range(4)], dtype=tf.float32)

    def ft_weighted_reasoning_loss(y_true, y_pred):
        y_true_int = tf.reshape(tf.cast(y_true, tf.int32), [-1])
        y_true_oh  = tf.one_hot(y_true_int, depth=4)
        unweighted = tf.keras.losses.categorical_crossentropy(y_true_oh, y_pred, label_smoothing=0.1)
        return unweighted * tf.gather(ft_weights_tensor, y_true_int)

    model.compile(
        optimizer=keras.optimizers.AdamW(LR, weight_decay=0.01, clipnorm=0.5),
        loss={
            "prediction": SovereignLoss(direction_weight=10.0),
            "certainty":  certainty_loss,
            "reasoning":  ft_weighted_reasoning_loss,
            "next_token": keras.losses.SparseCategoricalCrossentropy(),
        },
        loss_weights={"prediction": 6.0, "certainty": 1.0, "reasoning": 1.0, "next_token": 2.0},
        metrics={"prediction": [SovereignAccuracy()]}
    )

    if ds_info["steps_tr"] > 0:
        model.fit(ds_info["tr_ds"], epochs=FT_EPOCHS, steps_per_epoch=ds_info["steps_tr"], verbose=0)
        model.save(str(WFA_MODEL_PATH))

def wilson_ci(wins: int, n: int, z: float = 1.959964):
    """95% Wilson score interval — same formula as train.py's EdgeTracker,
    used here so a backtest win rate and a training-epoch accuracy are
    judged by the same statistical bar."""
    if n == 0:
        return 0.5, 0.0, 1.0
    p = wins / n
    denom  = 1.0 + (z * z) / n
    center = (p + (z * z) / (2 * n)) / denom
    margin = (z * ((p * (1 - p) / n + (z * z) / (4 * n ** 2)) ** 0.5)) / denom
    return p, center - margin, center + margin


# ── 3. Final Report ────────────────────────────────────────────────────────────
df_r = pd.DataFrame(results)
trades = df_r[df_r["signal"] != "HOLD"].copy()
n_trades = len(trades)

print("\n" + "=" * 60)
print("📊 WALK-FORWARD SIMULATION RESULTS")
print("=" * 60)

if n_trades > 0:
    trades["pred_dir"] = trades["signal"].map({"LONG": 1.0, "SHORT": -1.0})
    trades["correct"]  = (trades["pred_dir"] == trades["actual_dir"])
    trades["pnl"]      = trades["correct"].map({True: 1 - FEE_PCT, False: -1 - FEE_PCT})
    trades["naive_correct"] = (trades["naive_dir"] == trades["actual_dir"])

    win_rate = trades["correct"].mean() * 100
    tot_pnl  = trades["pnl"].sum()
    n_wins   = int(trades["correct"].sum())

    p, ci_low, ci_high = wilson_ci(n_wins, n_trades)
    significant = ci_low > 0.50
    verdict = "✅ STATISTICALLY SIGNIFICANT EDGE" if significant else "— not significant yet (could be noise)"

    # Naive baseline: same persistence rule, judged unconditionally over
    # every candle in the test window, and again restricted to only the
    # candles the model actually chose to trade (the fair comparison — if
    # the model can't beat the naive rule on the trades it hand-picked,
    # the certainty gate isn't adding anything).
    all_valid = df_r[df_r["naive_dir"] != 0.0]
    naive_unconditional = (all_valid["naive_dir"] == all_valid["actual_dir"]).mean() * 100
    naive_on_same_trades = trades["naive_correct"].mean() * 100

    print(f"  Total Chunks   : {n_chunks}")
    print(f"  Total Trades   : {n_trades}")
    print(f"  Win Rate       : {win_rate:.1f}%  (95% CI [{ci_low*100:.1f}%, {ci_high*100:.1f}%])  {verdict}")
    print(f"  Naive Baseline : {naive_unconditional:.1f}% (all candles) | "
          f"{naive_on_same_trades:.1f}% (same trades model picked)")
    print(f"  Total P&L      : {tot_pnl:+.2f} units")
    print(f"  Return/Trade   : {tot_pnl/n_trades*100:+.1f}%")

    # Volatility-regime breakdown — median split on the local realized vol
    # each trade was made under, to check the edge isn't concentrated in
    # just one market regime (the literature's #1 reason backtested "edge"
    # doesn't survive live trading).
    vol_median = trades["vol"].median()
    print(f"\n  Regime breakdown (median local vol = {vol_median:.4f}):")
    for label, mask in [("Low-vol ", trades["vol"] <= vol_median),
                         ("High-vol", trades["vol"] > vol_median)]:
        sub = trades[mask]
        if len(sub) > 0:
            sub_win = sub["correct"].mean() * 100
            _, sub_lo, sub_hi = wilson_ci(int(sub["correct"].sum()), len(sub))
            print(f"    {label} : {len(sub):>4} trades | win {sub_win:5.1f}% | "
                  f"95% CI [{sub_lo*100:.1f}%, {sub_hi*100:.1f}%]")
        else:
            print(f"    {label} : 0 trades")
else:
    print("  No trades executed during the simulation period.")
print("=" * 60 + "\n")
