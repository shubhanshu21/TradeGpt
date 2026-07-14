"""Regression tests for strategies/swing/ml_strategy.py's checkpoint loading."""
import pickle
import shutil
from pathlib import Path

import numpy as np
import pytest

from core.hydra import build_kraken, init_kraken_hardware
from strategies.swing.ml_strategy import MLSwingStrategy

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
TEST_SYMBOL = "_UNITTEST_VOCAB"

init_kraken_hardware()


@pytest.fixture
def tiny_checkpoint():
    """A real saved checkpoint with vocab_size DELIBERATELY different from
    build_kraken's default of 32 - reproduces the real-data scenario where
    fit_return_vocab's quantile-edge collapsing yields fewer tokens than
    requested (observed 31 on real HDFCBANK data during this session)."""
    ckpt_dir = MODELS_DIR / TEST_SYMBOL
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    n_features, ctx, forecast, vocab_size = 6, 4, 2, 17

    model = build_kraken(n_features=n_features, context_window=ctx, forecast_steps=forecast, vocab_size=vocab_size)
    model.save(ckpt_dir / "hydra_best.keras")
    with open(ckpt_dir / "model_shape.pkl", "wb") as f:
        pickle.dump({
            "context_window": ctx, "forecast_steps": forecast, "n_features": n_features,
            "symbol": TEST_SYMBOL,
            "bin_edges": np.linspace(-0.05, 0.05, vocab_size - 1),
            "bin_centers": np.zeros(vocab_size, dtype="float32"),
        }, f)

    yield ckpt_dir
    shutil.rmtree(ckpt_dir, ignore_errors=True)


def test_load_checkpoint_with_non_default_vocab_size(tiny_checkpoint):
    # Regression test for a real bug: MLSwingStrategy._load() used to call
    # build_kraken() without passing vocab_size, so it always defaulted to
    # 32 regardless of what the checkpoint was actually trained with -
    # load_weights then failed with a shape mismatch on next_candle's Dense
    # layer for any symbol whose real vocab size wasn't exactly 32 (common,
    # not rare - quantile bin edges collapse on real data often).
    strat = MLSwingStrategy()
    strat._load(TEST_SYMBOL)  # must not raise
    assert TEST_SYMBOL in strat._models
    assert strat._shapes[TEST_SYMBOL]["n_features"] == 6
