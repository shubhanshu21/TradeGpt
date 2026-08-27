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
VOCAB_SIZE  = 8

init_kraken_hardware()


@pytest.fixture(scope="module")
def tiny_model():
    return build_kraken(
        n_features=N_FEATURES,
        context_window=CONTEXT_WIN,
        forecast_steps=FORECAST,
        vocab_size=VOCAB_SIZE,
    )


def test_build_kraken_single_input_output_shapes(tiny_model):
    rng = np.random.default_rng(0)
    market_input = rng.normal(size=(BATCH, CONTEXT_WIN, N_FEATURES)).astype("float32")

    preds, certainty, reasoning, next_candle = tiny_model(market_input, training=False)

    # 6 channels: close, volatility, volume, open, high, low - a real full-candle forecast.
    assert preds.shape == (BATCH, FORECAST + 1, 6)
    # certainty = reasoning's own max-softmax confidence (see build_kraken) -
    # one value per window, not per-timestep (that was the old MoE-routing-
    # consensus design, replaced after it was found to collapse to a near-
    # constant output on real data).
    assert certainty.shape == (BATCH,)
    assert reasoning.shape == (BATCH, 4)
    assert next_candle.shape == (BATCH, VOCAB_SIZE)

    # certainty = max of a softmax distribution over 4 classes, so it's
    # bounded to [0.25, 1.0], not the full [0,1] a sigmoid would allow.
    assert np.all(certainty.numpy() >= 0.25) and np.all(certainty.numpy() <= 1.0)
    np.testing.assert_allclose(reasoning.numpy().sum(axis=-1), 1.0, atol=1e-5)
    np.testing.assert_allclose(next_candle.numpy().sum(axis=-1), 1.0, atol=1e-5)


def test_build_kraken_has_single_step_next_candle_head_not_generation(tiny_model):
    # Single INPUT still (no dual market+token input like the old crypto
    # version's autoregressive-generation design needed) - real next-token
    # classification OUTPUT, but single-step (predict once), not chained
    # generation. Exactly 4 outputs: prediction, certainty, reasoning, next_candle.
    assert len(tiny_model.inputs) == 1
    assert len(tiny_model.outputs) == 4


def test_hydra_block_attention_dropout_matches_block_dropout(tiny_model):
    # Regression test for a real bug: HydraBlock used to build its MLALayer
    # without forwarding dropout_rate, so the attention-probability dropout
    # silently stayed at MLALayer's own default instead of the block's
    # configured value.
    for layer in tiny_model.layers:
        if hasattr(layer, "attn") and hasattr(layer, "dropout_rate"):
            assert layer.attn.dropout_rate == layer.dropout_rate


def test_moe_expert_dropout_is_higher_than_block_dropout(tiny_model):
    # MoE's routed/sparse expert path should get a HIGHER dropout than the
    # block's own dense-layer rate (standard practice - sparse capacity is
    # more overfitting-prone since each expert only sees a fraction of
    # tokens), not silently share the same rate or default to 0.
    for layer in tiny_model.layers:
        if hasattr(layer, "moe") and hasattr(layer, "dropout_rate"):
            assert layer.moe.expert_dropout_rate > layer.dropout_rate
