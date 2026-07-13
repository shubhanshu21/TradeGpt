"""
Unit tests for the pure data-transformation logic in data/preprocess.py —
the pieces every training run and every inference call depends on, kept
separate from anything that needs a live checkpoint or GPU.
"""
import numpy as np
import pytest

from data.preprocess import (
    apply_dls, _hits_target_before_stop, build_feature_cols,
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
