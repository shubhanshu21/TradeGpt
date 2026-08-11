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
    """A real saved checkpoint built at build_kraken's FIXED default
    vocab_size (32) - matching what train.py actually does for every real
    checkpoint (see train.py's build_kraken() call sites, which never pass
    vocab_size) - but with a COLLAPSED bin_centers/bin_edges vocabulary
    (fewer than 32 entries), reproducing the real-data scenario where
    fit_return_vocab's quantile-edge collapsing yields fewer distinct
    tokens than requested (observed 31 on real HDFCBANK data during this
    session). The next_candle Dense layer width and the bin_centers length
    are two independent things - this is what regresses against."""
    ckpt_dir = MODELS_DIR / TEST_SYMBOL
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    n_features, ctx, forecast, collapsed_vocab = 6, 4, 2, 17

    model = build_kraken(n_features=n_features, context_window=ctx, forecast_steps=forecast)
    model.save(ckpt_dir / "hydra_best.keras")
    with open(ckpt_dir / "model_shape.pkl", "wb") as f:
        pickle.dump({
            "context_window": ctx, "forecast_steps": forecast, "n_features": n_features,
            "symbol": TEST_SYMBOL,
            "bin_edges": np.linspace(-0.05, 0.05, collapsed_vocab - 1),
            "bin_centers": np.zeros(collapsed_vocab, dtype="float32"),
        }, f)

    yield ckpt_dir
    shutil.rmtree(ckpt_dir, ignore_errors=True)


def test_load_checkpoint_with_collapsed_bin_vocab(tiny_checkpoint):
    # Regression test for a real bug: MLSwingStrategy._load() used to call
    # build_kraken(vocab_size=len(bin_centers)), which broke load_weights
    # the moment a symbol's collapsed vocab (common - quantile bin edges
    # collapse on real data often) didn't match the fixed 32-wide
    # next_candle layer every real checkpoint (pretrain and every
    # fine-tune) actually has. _load() must always build at the fixed
    # default and only use bin_centers for decoding, never for sizing.
    strat = MLSwingStrategy()
    strat._load(TEST_SYMBOL)  # must not raise despite collapsed bin_centers
    assert TEST_SYMBOL in strat._models
    assert strat._shapes[TEST_SYMBOL]["n_features"] == 6
    assert len(strat._shapes[TEST_SYMBOL]["bin_centers"]) == 17
