"""
Unit tests for the pure data-transformation logic in data/preprocess.py —
the pieces every training run and every inference call depends on, kept
separate from anything that needs a live checkpoint or GPU.
"""
import numpy as np
import pytest

from data.preprocess import (
    apply_dls, fit_return_vocab, tokenize_returns,
    _hits_tp_before_sl, build_feature_cols,
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


def test_return_vocab_roundtrip():
    rng = np.random.default_rng(1)
    returns = rng.normal(0, 0.001, size=5000)
    vocab_size = 16

    bin_edges, bin_centers = fit_return_vocab(returns, vocab_size=vocab_size)

    assert len(bin_edges) == vocab_size - 1
    assert np.all(np.diff(bin_edges) >= 0), "bin edges must be monotonically increasing"
    assert len(bin_centers) == vocab_size

    tokens = tokenize_returns(returns, bin_edges)
    assert tokens.dtype == np.int32
    assert tokens.min() >= 0
    assert tokens.max() < vocab_size

    # Quantile-based binning should spread tokens roughly evenly, not dump
    # everything into one bucket.
    counts = np.bincount(tokens, minlength=vocab_size)
    assert counts.max() < len(returns) * 0.5


@pytest.mark.parametrize("direction,path,expect_tp", [
    (1, [101, 102, 103], True),    # long: price rises straight to TP
    (1, [99, 98, 97], False),      # long: price falls straight to SL
    (-1, [99, 98, 97], True),      # short: price falls straight to TP
    (-1, [101, 102, 103], False),  # short: price rises straight to SL
])
def test_hits_tp_before_sl_directional(direction, path, expect_tp):
    entry = 100.0
    result = _hits_tp_before_sl(
        entry, np.array(path, dtype="float32"),
        tp_level=2.0, sl_level=2.0, direction=direction,
    )
    assert result is expect_tp


def test_hits_tp_before_sl_neither_hit_returns_false():
    entry = 100.0
    path = np.array([100.5, 100.8, 100.3], dtype="float32")
    result = _hits_tp_before_sl(entry, path, tp_level=5.0, sl_level=5.0, direction=1)
    assert result is False
