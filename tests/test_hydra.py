"""
Unit tests for the Hydra model in core/hydra.py.

Kept deliberately tiny (small context_window/n_features/vocab_size, single
forward pass, no training loop) since these tests share CPU with whatever
production train.py run may currently be active on this host — the goal is
to catch shape/signature regressions cheaply, not to benchmark the model.
"""
import numpy as np
import pytest

from core.hydra import build_kraken, init_kraken_hardware, _sample_from_probs

# Fixed dims for a cheap-to-build test model — independent of production config.
N_FEATURES  = 6
CONTEXT_WIN = 4
FORECAST    = 2
VOCAB_SIZE  = 8
BATCH       = 2

init_kraken_hardware()


@pytest.fixture(scope="module")
def tiny_model():
    return build_kraken(
        n_features=N_FEATURES,
        context_window=CONTEXT_WIN,
        forecast_steps=FORECAST,
        vocab_size=VOCAB_SIZE,
    )


def test_build_kraken_dual_input_output_shapes(tiny_model):
    rng = np.random.default_rng(0)
    market_input = rng.normal(size=(BATCH, CONTEXT_WIN, N_FEATURES)).astype("float32")
    token_input  = rng.integers(0, VOCAB_SIZE, size=(BATCH, CONTEXT_WIN)).astype("int32")

    preds, certainty, reasoning, next_token = tiny_model([market_input, token_input], training=False)

    assert preds.shape == (BATCH, FORECAST + 1, 3)
    assert certainty.shape == (BATCH, CONTEXT_WIN)
    assert reasoning.shape == (BATCH, 4)
    assert next_token.shape == (BATCH, CONTEXT_WIN, VOCAB_SIZE)

    # certainty is sigmoid-calibrated, reasoning/next_token are softmax — all bounded [0,1].
    assert np.all(certainty.numpy() >= 0.0) and np.all(certainty.numpy() <= 1.0)
    np.testing.assert_allclose(reasoning.numpy().sum(axis=-1), 1.0, atol=1e-5)
    np.testing.assert_allclose(next_token.numpy().sum(axis=-1), 1.0, atol=1e-5)


def test_sample_from_probs_top_k_1_is_deterministic():
    probs = np.array([0.1, 0.5, 0.05, 0.35])
    argmax = int(np.argmax(probs))

    for seed in range(5):
        rng = np.random.default_rng(seed)
        token = _sample_from_probs(probs, temperature=1.0, top_k=1, rng=rng)
        assert token == argmax


def test_sample_from_probs_returns_valid_index():
    rng = np.random.default_rng(42)
    probs = np.array([0.25, 0.25, 0.25, 0.25])
    for _ in range(20):
        token = _sample_from_probs(probs, temperature=0.8, top_p=0.9, rng=rng)
        assert 0 <= token < len(probs)
