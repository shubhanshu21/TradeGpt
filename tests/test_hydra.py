"""
Unit tests for the Hydra model in core/hydra.py.

Kept deliberately tiny (small context_window/n_features, single forward
pass, no training loop) since these tests share CPU with whatever
production train.py run may currently be active on this host — the goal is
to catch shape/signature regressions cheaply, not to benchmark the model.
"""
import numpy as np
import pytest

from core.hydra import build_kraken, init_kraken_hardware

# Fixed dims for a cheap-to-build test model — independent of production config.
N_FEATURES  = 6
CONTEXT_WIN = 4
FORECAST    = 2
BATCH       = 2

init_kraken_hardware()


@pytest.fixture(scope="module")
def tiny_model():
    return build_kraken(
        n_features=N_FEATURES,
        context_window=CONTEXT_WIN,
        forecast_steps=FORECAST,
    )


def test_build_kraken_single_input_output_shapes(tiny_model):
    rng = np.random.default_rng(0)
    market_input = rng.normal(size=(BATCH, CONTEXT_WIN, N_FEATURES)).astype("float32")

    preds, certainty, reasoning = tiny_model(market_input, training=False)

    assert preds.shape == (BATCH, FORECAST + 1, 3)
    assert certainty.shape == (BATCH, CONTEXT_WIN)
    assert reasoning.shape == (BATCH, 4)

    # certainty is sigmoid-calibrated, reasoning is softmax — both bounded [0,1].
    assert np.all(certainty.numpy() >= 0.0) and np.all(certainty.numpy() <= 1.0)
    np.testing.assert_allclose(reasoning.numpy().sum(axis=-1), 1.0, atol=1e-5)


def test_build_kraken_has_no_next_token_head(tiny_model):
    # Equity edition deliberately dropped the GPT-style next-token head from
    # the model - exactly 3 outputs (prediction, certainty, reasoning), not 4.
    assert len(tiny_model.outputs) == 3


def test_hydra_block_attention_dropout_matches_block_dropout(tiny_model):
    # Regression test for a real bug: HydraBlock used to build its MLALayer
    # without forwarding dropout_rate, so the attention-probability dropout
    # silently stayed at MLALayer's own default instead of the block's
    # configured value.
    for layer in tiny_model.layers:
        if hasattr(layer, "attn") and hasattr(layer, "dropout_rate"):
            assert layer.attn.dropout_rate == layer.dropout_rate
