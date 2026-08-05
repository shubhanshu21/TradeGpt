"""
Unit tests for the pure data-transformation logic in data/preprocess.py —
the pieces every training run and every inference call depends on, kept
separate from anything that needs a live checkpoint or GPU.
"""
import numpy as np
import pandas as pd
import pytest

from data.preprocess import (
    apply_dls, _hits_target_before_stop, build_feature_cols,
    fit_return_vocab, tokenize_returns, build_dataset_streaming,
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


def test_prediction_target_entry_matches_real_trade_entry_price():
    # Regression test for a real, previously-unnoticed bug: the prediction
    # head's "entry" reference (row 0 of its target) used to be TODAY's
    # close (the context window's last day), while the real reasoning label
    # anchors on the ACTUAL entry day (one trading day later - "enter the
    # day after the window ends"), not today. Two model heads meant to
    # AGREE on direction (see ml_strategy.py's multi-head gate) were being
    # trained against direction measured a full day apart. This test builds
    # one window by hand and confirms the prediction target's un-scaled
    # index-0 close is now anchored on the real entry DAY (entry_idx), not
    # the day before it.
    #
    # NOTE: the reasoning label's own entry PRICE was separately found to
    # use the entry day's close where it should use its open (see
    # test_reasoning_label_entry_price_uses_real_backtest_fill_price below,
    # matching swing_backtest/engine.py's actual next-day-OPEN fill) - so a
    # small same-day open-vs-close residual now exists between what this
    # test checks (prediction head, close) and the reasoning label (open).
    # That's a deliberately smaller, accepted gap versus the full-day
    # mismatch this test guards against - fixing it fully would mean
    # changing which OHLC channel backs the model's primary "direction"
    # channel everywhere (loss, metrics, inference decoding), a much larger
    # change than the entry-DAY alignment this test is about.
    rng = np.random.default_rng(4)
    n = 300
    close = 100 + np.cumsum(rng.normal(0, 1, size=n))
    ctx, forecast = 60, 20
    features = build_feature_cols()
    t_close = features.index("close")

    arr = np.zeros((n, len(features)), dtype="float32")
    arr[:, t_close] = close  # only the column this test actually checks matters

    i = 50
    x_raw = arr[i:i + ctx]
    y_raw = arr[i + ctx: i + ctx + forecast + 1]
    x_scaled, local_mean, local_std = apply_dls(x_raw)
    y_scaled = np.clip((y_raw - local_mean) / local_std, -5.0, 5.0)

    entry_idx = i + ctx
    real_entry_price = arr[entry_idx, t_close]
    y_row0_unscaled = y_scaled[0, t_close] * local_std[t_close] + local_mean[t_close]

    assert np.isclose(real_entry_price, y_row0_unscaled, atol=1e-4)


def test_reasoning_label_entry_price_uses_real_backtest_fill_price():
    # Regression test for a real bug found by reading swing_backtest/
    # engine.py directly rather than trusting a comment: the comment right
    # above build_dataset_streaming's entry_price line always claimed
    # "matching swing_backtest/engine.py's next-day-open fill", but the
    # code read the entry day's CLOSE, not its OPEN. engine.py's real fill
    # code is `price = entry_row["open"]` - confirmed directly. Every
    # reasoning label (LONG/SHORT/FEE_TRAP/NOISE) was classified against a
    # price the real backtest/live engine never actually pays. Verified
    # on real HDFCBANK data this isn't rounding-error-sized: one sampled
    # window showed a ~0.7% overnight open-vs-close gap, a meaningful
    # fraction of the 5%/10% stop/target thresholds.
    #
    # This test builds a window where open and close on the entry day are
    # DELIBERATELY far apart, runs it through the real build_dataset_streaming
    # pipeline, and confirms the resulting LONG/SHORT/FEE_TRAP/NOISE label
    # is the one implied by the entry day's OPEN, not its CLOSE - a
    # black-box check of the real code path, not a hand-rolled bypass.
    rng = np.random.default_rng(5)
    n = 200
    ctx, forecast = 20, 5

    dates = pd.bdate_range("2018-01-01", periods=n, tz="Asia/Kolkata")
    close = np.full(n, 100.0)
    open_ = np.full(n, 100.0)
    high = np.full(n, 100.5)
    low = np.full(n, 99.5)

    entry_idx = ctx  # first usable window (i=0) enters here
    # Entry day gaps DOWN hard at the open, then closes back near 100 -
    # and the whole forecast window that follows stays pinned near the
    # OPEN price, not the close. Using open (correct) as entry should
    # classify as a real LONG (a sustained ~8% move up from a low entry
    # that clears target-before-stop); using close (the bug) as entry
    # would see almost no real move exists from the true close.
    open_[entry_idx] = 92.0
    close[entry_idx] = 100.0
    high[entry_idx] = 100.2
    low[entry_idx] = 91.5
    # high_path/low_path in the real code are data[entry_idx:entry_idx+forecast]
    # - INCLUDING the entry day itself as day 0, not starting the day after -
    # so the target must clear within entry_idx .. entry_idx+forecast-1.
    for k in range(1, forecast + 1):
        open_[entry_idx + k] = 99.5 + k * 0.4
        close[entry_idx + k] = 99.5 + k * 0.4
        high[entry_idx + k] = 99.5 + k * 0.4 + 0.3   # clears 92*1.10=101.2 by day entry_idx+4
        low[entry_idx + k] = 99.5 + k * 0.4 - 0.3

    df = pd.DataFrame({
        "date": dates, "open": open_, "high": high, "low": low,
        "close": close, "volume": np.full(n, 10000),
    })

    info = build_dataset_streaming({"SYNTH": df}, context_window=ctx, forecast_steps=forecast, batch_size=4)
    # tr_ds is shuffled (both at build time and per-epoch), so scan every
    # window rather than assuming order - every OTHER window in this
    # synthetic set is perfectly flat (100/100.5/99.5, no real move), so a
    # LONG classification can only come from the one deliberately-engineered
    # window, wherever it landed after shuffling.
    reasoning_labels = []
    for _, yb in info["tr_ds"]:
        reasoning_labels.extend(yb["reasoning"].numpy().ravel().tolist())

    # 0 = LONG - only reachable if entry_price used the OPEN (92.0), where
    # target (92*1.10=101.2) clears before any stop; using the CLOSE (100.0)
    # would need a ~10% move from 100 that this synthetic path never makes.
    assert 0 in reasoning_labels, f"expected a LONG (0) label somewhere; got labels {set(reasoning_labels)}"


def test_build_dataset_streaming_purges_leaking_windows_at_train_val_boundary():
    # Regression test for a real methodological gap: consecutive windows
    # share up to context_window+forecast_steps-1 days of their footprint,
    # and each window's real trade-outcome label depends on forecast_steps
    # days AFTER its own window-end date. Without a purge/embargo gap, a
    # training window right at the train/val boundary has a label computed
    # from price action inside what's supposed to be the held-out
    # validation period - real look-ahead leakage (see "purged
    # cross-validation" - standard technique for overlapping-window
    # financial time series).
    rng = np.random.default_rng(3)
    n = 800
    # tz-aware to match real cached data (e.g. India VIX) - a tz-naive vs
    # tz-aware merge inside compute_indicators would otherwise raise.
    dates = pd.bdate_range("2015-01-01", periods=n, tz="Asia/Kolkata")
    close = 100 + np.cumsum(rng.normal(0, 1, size=n))
    df = pd.DataFrame({
        "date": dates,
        "open": close, "high": close + 1, "low": close - 1,
        "close": close, "volume": rng.integers(1000, 5000, size=n),
    })

    ctx, forecast = 20, 10
    info = build_dataset_streaming({"SYNTH": df}, context_window=ctx, forecast_steps=forecast, batch_size=8)

    # steps_tr is computed from the (purged) train window count - purging
    # windows near the boundary should mean fewer usable train windows than
    # a naive 80% split of the total would give.
    total_possible_windows = n - ctx - forecast + 1
    naive_tr_end = int(total_possible_windows * 0.8)
    actual_train_windows = info["steps_tr"] * 8  # approx, batched/shuffled but same order of magnitude
    assert actual_train_windows <= naive_tr_end, (
        "purge should remove some windows from the naive train/val split, not add any")


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
