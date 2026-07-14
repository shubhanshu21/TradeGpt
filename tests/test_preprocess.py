"""
Unit tests for the pure data-transformation logic in data/preprocess.py —
the pieces every training run and every inference call depends on, kept
separate from anything that needs a live checkpoint or GPU.
"""
import numpy as np
import pytest

from data.preprocess import (
    apply_dls, _hits_target_before_stop, build_feature_cols,
    fit_return_vocab, tokenize_returns,
)


def test_build_feature_cols_is_stable_and_unique():
    cols = build_feature_cols()
    assert len(cols) > 0
    assert len(cols) == len(set(cols)), "duplicate feature column names"
    assert all(isinstance(c, str) for c in cols)


def test_apply_dls_normalizes_and_clips():
    rng = np.random.default_rng(0)
    x_raw = rng.normal(loc=100.0, scale=5.0, size=(30, 4)).astype("float32")

    x_scaled, local_mean, local_std = apply_dls(x_raw)

    assert x_scaled.shape == x_raw.shape
    assert x_scaled.dtype == np.float32
    assert np.all(x_scaled >= -5.0) and np.all(x_scaled <= 5.0)
    np.testing.assert_allclose(local_mean, x_raw.mean(axis=0))


def test_apply_dls_std_floor_prevents_division_explosion():
    # A perfectly flat column (zero variance) must not blow up to inf/nan —
    # this is exactly the "volatility flatlines" case the std floor guards against.
    x_raw = np.full((10, 2), 42.0, dtype="float32")
    x_scaled, local_mean, local_std = apply_dls(x_raw)

    assert np.all(local_std >= 1e-3)
    assert np.all(np.isfinite(x_scaled))
    np.testing.assert_allclose(x_scaled, 0.0)


@pytest.mark.parametrize("direction,high_path,low_path,expect_tp", [
    (1, [101, 102, 103], [100.5, 101, 102], True),    # long: high reaches TP before low reaches SL
    (1, [100.5, 100.2, 100.1], [99, 98, 97], False),  # long: low hits SL first
    (-1, [100.5, 100.2, 100.1], [99, 98, 97], True),  # short: low reaches TP before high reaches SL
    (-1, [101, 102, 103], [100.5, 101, 102], False),  # short: high hits SL first
])
def test_hits_target_before_stop_directional(direction, high_path, low_path, expect_tp):
    entry = 100.0
    tp_price = entry * (1 + 0.02) if direction == 1 else entry * (1 - 0.02)
    sl_price = entry * (1 - 0.02) if direction == 1 else entry * (1 + 0.02)
    result = _hits_target_before_stop(
        np.array(high_path, dtype="float32"), np.array(low_path, dtype="float32"),
        tp_price, sl_price, direction,
    )
    assert result is expect_tp


def test_hits_target_before_stop_neither_hit_returns_false():
    high_path = np.array([100.5, 100.8, 100.3], dtype="float32")
    low_path = np.array([99.8, 99.9, 99.7], dtype="float32")
    result = _hits_target_before_stop(high_path, low_path, tp_price=110.0, sl_price=90.0, direction=1)
    assert result is False


def test_return_vocab_roundtrip():
    # GPT-style next-candle vocabulary: quantile-bin real returns, then
    # confirm every value tokenizes into a valid, in-range token ID.
    rng = np.random.default_rng(1)
    returns = rng.normal(0, 0.01, size=5000).astype("float32")

    bin_edges, bin_centers, vocab_size = fit_return_vocab(returns, vocab_size=32)

    assert vocab_size <= 32  # can be fewer if quantile edges collapse, never more
    assert len(bin_centers) == vocab_size
    assert np.all(np.diff(bin_edges) >= 0), "bin edges must be monotonically increasing"

    tokens = tokenize_returns(returns, bin_edges)
    assert tokens.dtype == np.int32
    assert tokens.min() >= 0
    assert tokens.max() < vocab_size

    # Quantile-based binning should spread tokens roughly evenly, not dump
    # everything into one bucket - that's the whole point over equal-width bins.
    counts = np.bincount(tokens, minlength=vocab_size)
    assert counts.max() < len(returns) * 0.5


def test_return_vocab_bin_centers_reflect_sign_of_their_bucket():
    # A token with a positive bin_center should correspond to actually-positive
    # returns, and vice versa - this is what ml_strategy.py relies on to decode
    # a predicted token back into an implied up/down direction.
    rng = np.random.default_rng(2)
    returns = np.concatenate([
        rng.uniform(-0.05, -0.01, size=1000),  # clearly negative days
        rng.uniform(0.01, 0.05, size=1000),    # clearly positive days
    ]).astype("float32")

    bin_edges, bin_centers, vocab_size = fit_return_vocab(returns, vocab_size=16)
    tokens = tokenize_returns(returns, bin_edges)

    for tok in np.unique(tokens):
        actual_returns_in_bucket = returns[tokens == tok]
        if len(actual_returns_in_bucket) == 0:
            continue
        # bin_center's sign should match the actual data in that bucket
        assert np.sign(bin_centers[tok]) == np.sign(np.median(actual_returns_in_bucket)) or bin_centers[tok] == 0
