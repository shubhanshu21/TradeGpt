"""
SHARED INFERENCE HELPER
========================
Every evaluation/backtest script in this repo used to build its own copy of
"load a checkpoint, build a window, run inference" — all of them written
against the old single-input model signature. When the model gained a second
input (token_input, for GPT-style next-token prediction), every one of those
copies broke silently or with a shape-mismatch crash.

This module is the one place that knows the model's real input/output shape.
Fix a bug here once, instead of in every script that calls the model.
"""
import pickle
import numpy as np

from core.hydra import build_kraken, init_kraken_hardware
from data.preprocess import build_feature_cols, compute_indicators, apply_dls, tokenize_returns


def load_trained_model(models_dir, checkpoint_name=None):
    """
    Load a checkpoint + its saved return-token vocabulary/shape metadata.
    Returns (model, vocab_dict). vocab_dict includes context_window,
    forecast_steps, n_features, timeframe, bin_edges, bin_centers, vocab_size —
    everything needed to build a compatible model and tokenize new windows
    the exact same way training did.
    """
    models_dir = str(models_dir)
    import os
    vocab_path = os.path.join(models_dir, "return_vocab.pkl")
    if not os.path.exists(vocab_path):
        raise FileNotFoundError(
            f"No return_vocab.pkl at {vocab_path} — train the model first "
            "(this file is written by train.py alongside the checkpoint)."
        )
    with open(vocab_path, "rb") as f:
        vocab = pickle.load(f)

    if checkpoint_name is None:
        best = os.path.join(models_dir, "hydra_best.keras")
        if not os.path.exists(best):
            raise FileNotFoundError(f"No checkpoint at {best} — train the model first.")
        checkpoint_name = "hydra_best.keras"
    ckpt_path = os.path.join(models_dir, checkpoint_name)

    init_kraken_hardware()
    n_feat = vocab.get("n_features", len(build_feature_cols()))
    model = build_kraken(
        n_features=n_feat,
        context_window=vocab["context_window"],
        forecast_steps=vocab["forecast_steps"],
        vocab_size=vocab["vocab_size"],
    )
    model.load_weights(ckpt_path)
    return model, vocab


def prepare_window(df, end_idx, vocab):
    """
    Build the (market_input, token_input) pair the model actually expects,
    for a real historical window ending at end_idx. Mirrors exactly what
    train.py's precompute loop and the prediction_viewer dashboard do.
    Returns (x_scaled, token_ids, local_mean, local_std, t_close_idx).
    """
    ctx_win = vocab["context_window"]
    df_feat = compute_indicators(df.iloc[max(0, end_idx - ctx_win - 150):end_idx].copy())
    features = build_feature_cols()
    data = df_feat[features].values.astype("float32")
    x_raw = data[-ctx_win:]
    x_scaled, local_mean, local_std = apply_dls(x_raw)

    t_close = features.index("close")
    close_prices = data[:, t_close]
    raw_returns = np.diff(close_prices) / (close_prices[:-1] + 1e-9)
    raw_returns = np.concatenate([[0.0], raw_returns]).astype("float64")
    token_ids = tokenize_returns(raw_returns[-ctx_win:], vocab["bin_edges"])

    return x_scaled, token_ids, local_mean, local_std, t_close


def run_inference(model, x_scaled, token_ids):
    """
    Single dual-input forward pass. Returns the raw four-tuple
    (prediction, certainty, reasoning, next_token) as numpy arrays for the
    single window passed in — the direct replacement for the old
    `model(X_in, training=False)` single-input call used everywhere.
    """
    X_in = x_scaled[np.newaxis].astype("float32")
    Tok_in = token_ids[np.newaxis].astype("int32")
    outputs = model([X_in, Tok_in], training=False)
    return tuple(o.numpy()[0] for o in outputs)
