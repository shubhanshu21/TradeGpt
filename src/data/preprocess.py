"""
SOVEREIGN KRAKEN — Equity Data Engine
=====================================================================
NSE cash equity swing trading: daily candles, real Zerodha delivery costs,
one model trained per symbol.

  - No order-book/microstructure features (funding rate, taker flow, OBI) —
    none of that exists for NSE cash equity; the feature set is built from
    real technical indicators + trend-structure features for daily swing data.
  - Every feature is a RATIO or bounded indicator, not a raw price level —
    a stock's own absolute price level can vary a lot over years of history
    (and, if ever pooled across symbols again, symbols also differ wildly —
    Rs 400 ITC vs Rs 2,400 RELIANCE), so scale-invariant features matter here.
  - Trade labels use the EXACT same day-by-day high/low stop/target check
    swing_backtest/engine.py uses for real trades, not an approximated
    proxy — so the model's training target matches what the real
    backtest/live engine actually trades on.
  - GPT-style next-candle token head: a genuine next-token classification
    head (see fit_return_vocab/tokenize_returns below), predicting a
    discretized bucket for tomorrow's return - trained the same way GPT
    trains (next-token cross-entropy), but as a SINGLE-STEP prediction, not
    chained autoregressive generation. Multi-step generation (sample a
    token, feed it back in, repeat for N days) was deliberately left out -
    errors compound badly over a generated numeric path with no grammar-like
    structure to self-correct, a well-documented failure mode worse for
    numeric time series than for text.
"""

import numpy as np
import pandas as pd
import tensorflow as tf
import yaml
from pathlib import Path

try:
    from config.sovereign_config import (CONTEXT_WINDOW, FORECAST_STEPS, HARD_STOP_LOSS_PCT,
                                          HARD_TARGET_PCT, FEE_RATE, UNIVERSE_SYMBOLS)
except ImportError:
    from src.config.sovereign_config import (CONTEXT_WINDOW, FORECAST_STEPS, HARD_STOP_LOSS_PCT,
                                              HARD_TARGET_PCT, FEE_RATE, UNIVERSE_SYMBOLS)

# Real cached daily candles — own copy under kat/cache/historical/, fetched
# by src/data/fetch_historical.py (broker + interval come from
# config/settings.yaml, same single source of truth used everywhere else —
# no hardcoded date range here, since that drifts the moment settings.yaml's
# start_date/end_date changes, as happened once already).
CACHE_DIR = Path(__file__).resolve().parent.parent.parent / "cache" / "historical"
_SETTINGS_PATH = Path(__file__).resolve().parent.parent.parent / "config" / "settings.yaml"
with open(_SETTINGS_PATH) as _f:
    _settings = yaml.safe_load(_f)
_BROKER_NAME = _settings["broker"]["name"]
_INTERVAL = _settings["swing"].get("interval", "day")


def load_universe_from_cache(symbols=None) -> dict:
    """Load {symbol: DataFrame} from the local cache (real broker daily
    candles). Matches by broker+symbol+interval prefix (not an exact
    start/end date range) so it stays correct regardless of what date range
    was used to fetch — picks the most recently fetched file if more than
    one exists for the same symbol."""
    symbols = symbols or UNIVERSE_SYMBOLS
    data = {}
    for symbol in symbols:
        matches = sorted(CACHE_DIR.glob(f"{_BROKER_NAME}_{symbol}_{_INTERVAL}_*.csv"))
        if not matches:
            print(f"   ⚠️ No cached data for {symbol} "
                  f"(pattern {_BROKER_NAME}_{symbol}_{_INTERVAL}_*.csv) — skipping")
            continue
        df = pd.read_csv(matches[-1], parse_dates=["date"])
        df = df.sort_values("date").reset_index(drop=True)
        data[symbol] = df
    return data


def load_benchmark_index() -> pd.DataFrame:
    """Nifty 50 index's own daily candles (cached the same way as any
    symbol - see fetch_historical.py) - used for the rel_strength_20d
    feature. A stock's OWN return says nothing about whether it's actually
    outperforming or lagging the broader market, which is a well-documented
    swing-trading signal no per-stock-only feature can capture. Returns None
    if the index hasn't been fetched/cached yet, so callers can fall back
    gracefully instead of crashing."""
    data = load_universe_from_cache(["NIFTY"])
    return data.get("NIFTY")


def load_vix_index() -> pd.DataFrame:
    """India VIX's own daily candles (cached the same way as any symbol -
    see fetch_historical.py) - used for the vix_zscore_60d feature. VIX is
    NSE's implied-volatility index (options-derived "fear gauge"): it's a
    market-wide regime signal no per-stock price/volume feature can
    capture on its own - the same 2% move means something very different
    to a swing trade when VIX is at 11 (calm market) vs 35 (panic).
    Returns None if not cached yet, so callers can fall back gracefully."""
    data = load_universe_from_cache(["INDIA VIX"])
    return data.get("INDIA VIX")


def build_feature_cols():
    """
    Equity daily-swing feature set. Every entry is a ratio, bounded
    indicator, or cyclical encoding — no raw price level — so the same
    feature vector is meaningfully comparable across stocks pooled together
    in one training set.
    """
    return [
        'day_of_week', 'month_sin', 'month_cos',
        'open', 'high', 'low', 'close', 'volume',
        'ret_1d', 'ret_5d', 'ret_20d',
        'sma20_dist', 'sma50_dist', 'sma200_dist', 'ema20_dist',
        'rsi', 'macd', 'macd_signal', 'macd_hist',
        'adx',
        'atr_pct', 'bb_width', 'bb_position',
        'volatility_20d',
        'volume_ratio',
        'hl_position_20', 'hl_position_252',
        'supertrend_dir',
        'donchian_position',
        'rel_strength_20d',
        'obv_dist',
        'vwap20_dist',
        'vix_zscore_60d',
    ] + CANDLESTICK_PATTERNS


# ALL 62 of pandas-ta-classic's candlestick patterns - none of the model's
# other 30 features have any notion of individual-candle SHAPE at all
# (they're trend/momentum/volatility summarized over rolling windows), so
# this is genuinely new information no existing feature captures. Each
# pattern is TA-Lib's standard signed -100/0/+100 bearish/none/bullish
# convention IN ONE COLUMN - both the long (bullish) and short (bearish)
# reading of a pattern share the same feature, scaled to [-1, 1] to match
# this project's other bounded features before DLS scaling is ever applied.
# Verified directly against real HDFCBANK data: all 62 compute cleanly, no
# NaN/non-numeric output (10 of the 62 never fire on that one stock's
# history - genuinely rare patterns, not a computation failure; they may
# still fire for other pooled symbols).
import pandas_ta_classic.candles as _candles_module
CANDLESTICK_PATTERN_NAMES = list(_candles_module.CDL_PATTERN_NAMES)
CANDLESTICK_PATTERNS = [f'cdl_{n}' for n in CANDLESTICK_PATTERN_NAMES]


def compute_indicators(df: pd.DataFrame, benchmark_df: pd.DataFrame = None,
                        vix_df: pd.DataFrame = None) -> pd.DataFrame:
    """
    Compute equity technical indicators. Expects columns: date, open, high,
    low, close, volume (one symbol's full daily history, sorted ascending).
    open/high/low/close/volume are kept as raw feature columns too — Dynamic
    Local Scaling (apply_dls) z-scores every window against its OWN local
    mean/std, so a raw price column is still cross-symbol comparable once
    scaled, and the labeling logic needs real close/high/low for its
    day-by-day stop/target walk.

    benchmark_df (optional): Nifty 50 index daily candles (see
    load_benchmark_index()), used for rel_strength_20d. If omitted, that
    feature is filled with 0 rather than crashing - keeps this function
    usable standalone/in tests without requiring the index to be cached.

    vix_df (optional): India VIX daily candles (see load_vix_index()), used
    for vix_zscore_60d. If omitted, that feature is filled with 0.
    """
    df = df.copy()
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)
    else:
        df['date'] = pd.date_range('2020-01-01', periods=len(df), freq='D')

    for col in ['open', 'high', 'low', 'close', 'volume']:
        if col not in df.columns:
            df[col] = df.get('close', 0.0)

    close, high, low, volume = df['close'], df['high'], df['low'], df['volume']

    # ── Returns ────────────────────────────────────────────────────────────
    df['ret_1d']  = close.pct_change(1).fillna(0)
    df['ret_5d']  = close.pct_change(5).fillna(0)
    df['ret_20d'] = close.pct_change(20).fillna(0)

    # ── Relative strength vs Nifty 50 (is this stock actually beating the
    # market, or just moving with it?) ───────────────────────────────────
    if benchmark_df is not None:
        bench = benchmark_df[['date', 'close']].copy()
        bench['date'] = pd.to_datetime(bench['date'])
        bench = bench.rename(columns={'close': 'bench_close'})
        df = df.merge(bench, on='date', how='left')
        # ffill only - bfill would pull a FUTURE benchmark value backward
        # into any row still NaN (only possible at a symbol's earliest
        # dates, before the cached benchmark series starts), a real if minor
        # leak. Leave those rows NaN here; the fillna(0) below zeroes them,
        # same as the no-benchmark-data fallback path already does.
        df['bench_close'] = df['bench_close'].ffill()
        bench_ret_20d = df['bench_close'].pct_change(20).fillna(0)
        df['rel_strength_20d'] = df['ret_20d'] - bench_ret_20d
        df = df.drop(columns=['bench_close'])
    else:
        df['rel_strength_20d'] = 0.0

    # ── Trend / moving-average distance (ratio, scale-invariant) ────────────
    sma20  = close.rolling(20,  min_periods=1).mean()
    sma50  = close.rolling(50,  min_periods=1).mean()
    sma200 = close.rolling(200, min_periods=1).mean()
    ema20  = close.ewm(span=20, adjust=False).mean()
    df['sma20_dist']  = (close - sma20)  / (sma20  + 1e-9)
    df['sma50_dist']  = (close - sma50)  / (sma50  + 1e-9)
    df['sma200_dist'] = (close - sma200) / (sma200 + 1e-9)
    df['ema20_dist']  = (close - ema20)  / (ema20  + 1e-9)

    # ── RSI ───────────────────────────────────────────────────────────────
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df['rsi'] = (100 - 100 / (1 + rs)).fillna(50)

    # ── MACD (normalized by price — scale-invariant across symbols) ─────────
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    macd_signal = macd_line.ewm(span=9, adjust=False).mean()
    df['macd']        = macd_line / (close + 1e-9)
    df['macd_signal'] = macd_signal / (close + 1e-9)
    df['macd_hist']   = (macd_line - macd_signal) / (close + 1e-9)

    # ── ADX (trend strength) ─────────────────────────────────────────────
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    prev_c = close.shift(1).fillna(close)
    tr = pd.concat([high - low, (high - prev_c).abs(), (low - prev_c).abs()], axis=1).max(axis=1)
    atr14 = tr.ewm(alpha=1/14, adjust=False).mean()
    atr_smooth = atr14.replace(0, np.nan)
    plus_di = 100 * pd.Series(plus_dm, index=df.index).ewm(alpha=1/14, adjust=False).mean() / atr_smooth
    minus_di = 100 * pd.Series(minus_dm, index=df.index).ewm(alpha=1/14, adjust=False).mean() / atr_smooth
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    df['adx'] = dx.ewm(alpha=1/14, adjust=False).mean().fillna(0)

    # ── ATR% / Bollinger ─────────────────────────────────────────────────
    df['atr_pct'] = atr14 / (close + 1e-9)
    std20 = close.rolling(20, min_periods=1).std().fillna(1e-9)
    bb_upper = sma20 + 2 * std20
    bb_lower = sma20 - 2 * std20
    df['bb_width'] = (bb_upper - bb_lower) / (sma20 + 1e-9)
    df['bb_position'] = (close - bb_lower) / (bb_upper - bb_lower + 1e-9)

    # ── Realized volatility ──────────────────────────────────────────────
    df['volatility_20d'] = close.pct_change().rolling(20, min_periods=1).std().fillna(0)

    # ── Volume ────────────────────────────────────────────────────────────
    avg_vol20 = volume.rolling(20, min_periods=1).mean()
    df['volume_ratio'] = (volume / (avg_vol20 + 1e-9)).clip(0, 10)

    # ── On-Balance Volume (is accumulation/distribution volume confirming
    # or diverging from price?) — raw OBV is an unbounded running total (it
    # only ever accumulates over a stock's whole history), so instead of the
    # raw level we use its distance from its OWN 20-day moving average, the
    # same ratio pattern as sma20_dist — bounded, scale-invariant, and
    # answers "is OBV trending above/below its own recent normal?" rather
    # than an ever-growing number with no natural zero point. ─────────────
    obv_step = np.sign(close.diff().fillna(0)) * volume
    obv = obv_step.cumsum()
    obv_sma20 = obv.rolling(20, min_periods=1).mean()
    obv_scale = volume.rolling(20, min_periods=1).mean() * 20 + 1e-9  # normalizer, not obv_sma20 itself
    df['obv_dist'] = ((obv - obv_sma20) / obv_scale).clip(-5, 5)

    # ── Rolling volume-weighted average price distance. NOTE: this is NOT
    # intraday VWAP (that needs intraday tick/bar data we don't have here,
    # daily-only OHLCV) — it's a 20-day rolling volume-weighted average of
    # the daily typical price, an honest daily-bar analog. Distance from it
    # is scale-invariant like the other *_dist features. ──────────────────
    typical_price = (high + low + close) / 3.0
    vwap20 = (typical_price * volume).rolling(20, min_periods=1).sum() / (
        volume.rolling(20, min_periods=1).sum() + 1e-9)
    df['vwap20_dist'] = (close - vwap20) / (vwap20 + 1e-9)

    # ── India VIX z-score (market-wide "fear gauge" regime signal, not
    # derived from this stock's own price/volume at all) — z-scored against
    # its OWN trailing 60-day mean/std rather than used as a raw level,
    # since VIX's "normal" range drifts over multi-year history; this tells
    # the model whether fear is unusually high/low right NOW relative to
    # recent normal. ───────────────────────────────────────────────────────
    if vix_df is not None:
        vix = vix_df[['date', 'close']].copy()
        vix['date'] = pd.to_datetime(vix['date'])
        vix = vix.rename(columns={'close': 'vix_close'})
        df = df.merge(vix, on='date', how='left')
        df['vix_close'] = df['vix_close'].ffill()  # no bfill - see bench_close above
        vix_mean60 = df['vix_close'].rolling(60, min_periods=1).mean()
        vix_std60 = df['vix_close'].rolling(60, min_periods=1).std().fillna(1e-9).replace(0, 1e-9)
        df['vix_zscore_60d'] = ((df['vix_close'] - vix_mean60) / vix_std60).clip(-5, 5)
        df = df.drop(columns=['vix_close'])
    else:
        df['vix_zscore_60d'] = 0.0

    # ── Price position in range (20d and 52-week) ────────────────────────
    high20, low20 = high.rolling(20, min_periods=1).max(), low.rolling(20, min_periods=1).min()
    df['hl_position_20'] = (close - low20) / (high20 - low20 + 1e-9)
    high252, low252 = high.rolling(252, min_periods=1).max(), low.rolling(252, min_periods=1).min()
    df['hl_position_252'] = (close - low252) / (high252 - low252 + 1e-9)

    # ── Supertrend direction (+1/-1) ──────────────────────────────────────
    period, multiplier = 10, 3.0
    hl2 = (high + low) / 2
    atr_st = tr.ewm(alpha=1/period, adjust=False).mean()
    upper_band = (hl2 + multiplier * atr_st).values
    lower_band = (hl2 - multiplier * atr_st).values
    close_v = close.values
    direction = np.ones(len(df))
    final_upper, final_lower = upper_band.copy(), lower_band.copy()
    for i in range(1, len(df)):
        if close_v[i - 1] <= final_upper[i - 1]:
            final_upper[i] = min(upper_band[i], final_upper[i - 1])
        if close_v[i - 1] >= final_lower[i - 1]:
            final_lower[i] = max(lower_band[i], final_lower[i - 1])
        if direction[i - 1] == 1:
            direction[i] = -1 if close_v[i] < final_lower[i] else 1
        else:
            direction[i] = 1 if close_v[i] > final_upper[i] else -1
    df['supertrend_dir'] = direction

    # ── Donchian position (10-day - short-term breakout, deliberately a
    # different window than hl_position_20's 20-day range; using the same
    # 20-day window here would make this a literal duplicate feature) ────
    donch_upper, donch_lower = high.rolling(10, min_periods=1).max(), low.rolling(10, min_periods=1).min()
    df['donchian_position'] = (close - donch_lower) / (donch_upper - donch_lower + 1e-9)

    # ── Candlestick patterns (pandas-ta-classic) — pure candle SHAPE, no
    # other feature here captures this at all (everything else is
    # trend/momentum/volatility summarized over a rolling window, not the
    # shape of an individual candle). TA-Lib's standard -100/0/+100
    # bearish/none/bullish scale, normalized to [-1, 1]. ────────────────
    try:
        from pandas_ta_classic.candles.cdl_pattern import cdl_pattern
        cdl_df = cdl_pattern(df['open'], high, low, close, name=CANDLESTICK_PATTERN_NAMES)
        for col, feat_name in zip(cdl_df.columns, CANDLESTICK_PATTERNS):
            df[feat_name] = (cdl_df[col] / 100.0).fillna(0.0)
    except Exception as exc:
        print(f"   ⚠️ Candlestick pattern detection failed ({exc}) - those {len(CANDLESTICK_PATTERNS)} "
              f"features will be 0 for this symbol")
        for feat_name in CANDLESTICK_PATTERNS:
            df[feat_name] = 0.0

    # ── Time features ─────────────────────────────────────────────────────
    df['day_of_week'] = df['date'].dt.dayofweek / 6.0
    month = df['date'].dt.month
    df['month_sin'] = np.sin(2 * np.pi * month / 12.0)
    df['month_cos'] = np.cos(2 * np.pi * month / 12.0)

    for col in build_feature_cols():
        if col not in df.columns:
            df[col] = 0.0

    df = df.replace([np.inf, -np.inf], np.nan)
    feat_df = df.fillna(0)
    feat_df[build_feature_cols()] = feat_df[build_feature_cols()].astype("float32").clip(-1e9, 1e9)
    feat_df["date"] = df["date"]
    return feat_df


def apply_dls(x_raw: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Dynamic Local Scaling — z-score a window against its OWN local mean/std.
    This is what makes pooling raw-price columns (open/high/low/close)
    across stocks with wildly different price levels safe: every window is
    normalized relative to itself, not to some global/cross-symbol scale.
    Std floor (1e-3) and Z-score clipping [-5,5] prevent blowups when
    volatility flatlines.
    """
    local_mean = x_raw.mean(axis=0)
    local_std = x_raw.std(axis=0)
    local_std = np.maximum(local_std, 1e-3)
    x_scaled = np.clip((x_raw - local_mean) / local_std, -5.0, 5.0).astype("float32")
    return x_scaled, local_mean, local_std


def fit_return_vocab(returns: np.ndarray, vocab_size: int = 32):
    """Builds the "vocabulary" for the GPT-style next-candle token head -
    quantile-bins a distribution of daily returns into vocab_size discrete
    tokens. Quantile-based (not equal-width) bins keep token frequency
    roughly balanced across the vocabulary - equal-width bins would waste
    most of the vocabulary on rare extreme moves and cram most ordinary
    trading days into just 2-3 buckets, which defeats the point of having
    a 32-token vocabulary at all.

    Returns (bin_edges, bin_centers, actual_vocab_size) - actual_vocab_size
    can be slightly less than requested if the data has enough ties that
    some quantile edges collapse (e.g. a stock with many exact-zero-return
    days), which happens on real data more often than on paper.
    """
    quantile_pts = np.linspace(0, 100, vocab_size + 1)[1:-1]
    bin_edges = np.unique(np.percentile(returns, quantile_pts))
    token_ids = np.digitize(returns, bin_edges)
    actual_vocab_size = len(bin_edges) + 1
    bin_centers = np.array([
        np.median(returns[token_ids == i]) if np.any(token_ids == i) else 0.0
        for i in range(actual_vocab_size)
    ], dtype="float32")
    return bin_edges, bin_centers, actual_vocab_size


def tokenize_returns(returns: np.ndarray, bin_edges: np.ndarray) -> np.ndarray:
    """Converts raw returns into token IDs using an already-fit vocabulary
    (bin_edges from fit_return_vocab) - the equivalent of a GPT tokenizer
    converting text into token IDs, just for discretized price moves."""
    return np.digitize(returns, bin_edges).astype("int32")


def _hits_target_before_stop(high_path, low_path, tp_price, sl_price, direction) -> bool:
    """
    Walk a day-by-day high/low path and check whether price reaches the
    target before the stop, using the same stop-price/target-price checks
    and same-day precedence as swing_backtest/engine.py's real trade
    simulation, so the model's training label matches what the real
    backtest/live engine actually trades on, not an approximated proxy.
    direction: +1 long, -1 short.

    Checks stop before target on a same-day overlap, matching
    swing_backtest/engine.py's conservative convention (real intraday
    touch order is unknowable from daily OHLC alone, and the two checks
    must agree or training labels a losing real trade as a win).
    """
    for h, l in zip(high_path, low_path):
        if direction == 1:
            hit_stop = l <= sl_price
            hit_target = h >= tp_price
        else:
            hit_stop = h >= sl_price
            hit_target = l <= tp_price
        if hit_stop:
            return False
        if hit_target:
            return True
    return False  # neither hit within the holding window


def build_dataset_streaming(data_by_symbol: dict, context_window=CONTEXT_WINDOW,
                             forecast_steps=FORECAST_STEPS, batch_size=32, stride=5,
                             burn_in_days=252):
    """
    Builds a training set from {symbol: DataFrame} - typically just one
    symbol per call now (see train.py's per-symbol loop), but still handles
    multiple if ever pooled again.

    Each symbol's own history produces its own windows (own DLS scaling per
    window — a stock's absolute price level never leaks across symbols since
    every window is normalized against only its own local mean/std). All
    symbols' windows are then pooled and split CHRONOLOGICALLY BY DATE (not
    by per-symbol row count) — every training window's window-end-date is
    strictly before every validation window's window-end-date, regardless of
    which symbol it came from, so there's no cross-symbol date leakage.

    `stride` controls the step between consecutive window start indices.
    stride=1 (the old default) makes consecutive windows share
    context_window-1 of their context_window days (98%+ overlap at the
    default 60-day window) - the model then sees the same underlying price
    history repeated ~context_window times per real epoch, just shifted by
    a day each time, which let it start memorizing pooled pretraining data
    by epoch ~7 despite a 200K+ window dataset (real overfitting observed:
    train/val accuracy gap widening past 0.24 with validation losing its
    significant edge, epoch_tracker__pretrained_base.csv epochs 7-8).
    Defaulting to stride=5 cuts window count ~5x but removes most of that
    redundancy, so a real epoch is closer to an actual pass over unique
    information instead of ~60 near-duplicate passes over the same data.

    `burn_in_days` drops each symbol's earliest rows before windowing -
    every rolling feature in compute_indicators() uses min_periods=1, so
    without this, early rows get degenerate long-window statistics (e.g. a
    "200-day SMA" from 5 real days) that are real but non-representative,
    landing only on the train side (chronologically earliest) and giving
    the model an easy, non-generalizing pattern to memorize. Default 252
    covers the longest lookback used (sma200/high252/low252). Set to 0 for
    small synthetic test data where warm-up realism isn't the point.

    Returns dict with: tr_ds, va_ds (tf.data.Dataset), steps_tr, steps_va,
    n_features, label_counts, vocab_size, bin_edges, bin_centers (the
    latter three describe the GPT-style next-candle token vocabulary, fit
    on this call's own train-portion data - persist them in model_shape.pkl
    so inference tokenizes/decodes with the exact same vocabulary training used).
    """
    features = build_feature_cols()
    t_close = features.index('close')
    t_vol = features.index('volatility_20d')
    t_volu = features.index('volume')
    t_high = features.index('high')
    t_low = features.index('low')
    t_open = features.index('open')
    t_ret1d = features.index('ret_1d')

    all_X, all_Y_pred, all_Y_cert, all_Y_reason, all_end_dates, all_next_ret_raw = [], [], [], [], [], []

    benchmark_df = load_benchmark_index()
    if benchmark_df is None:
        print("   ⚠️ No cached Nifty 50 index data - rel_strength_20d will be 0 for all symbols "
              "(run fetch_historical.py with NIFTY in the universe, or ignore if that's expected)")

    vix_df = load_vix_index()
    if vix_df is None:
        print("   ⚠️ No cached India VIX data - vix_zscore_60d will be 0 for all symbols "
              "(run fetch_historical.py, or ignore if that's expected)")

    print(f"   ⚓ Equity Data Engine: {len(data_by_symbol)} symbols → CTX={context_window} FORECAST={forecast_steps}")
    # Longest lookback used by any feature in compute_indicators() (sma200,
    # high252/low252) - every rolling call there uses min_periods=1, so
    # rows before this many real days of history exist get a degenerate
    # long-window statistic (e.g. a "200-day SMA" computed from 5 real
    # days) instead of a genuine one. Those rows land only on the train
    # side (chronologically earliest), giving the model an easy, fake,
    # non-generalizing pattern to memorize - a real contributor to the
    # train/val gap, found after two regularization fixes (weight_decay,
    # SwiGLU dropout) failed to move it. Burn-in is dropped before
    # windowing rather than patched per-indicator, so every feature gets a
    # real lookback consistently.
    for symbol, raw_df in data_by_symbol.items():
        df_feat = compute_indicators(raw_df, benchmark_df=benchmark_df, vix_df=vix_df)
        if burn_in_days > 0:
            if len(df_feat) <= burn_in_days:
                print(f"   ⚠️ {symbol}: not enough history ({len(df_feat)} rows) to clear the "
                      f"{burn_in_days}-day feature burn-in period — skipping")
                continue
            df_feat = df_feat.iloc[burn_in_days:].reset_index(drop=True)
        data = df_feat[features].values.astype("float32")
        dates = df_feat["date"].values
        n = len(data)
        # One fewer than before (no "+1") - y_raw now starts at entry_idx
        # instead of context_window-1, so it reaches one day further into
        # the future; the last window must still have that extra day
        # available, or the final iteration's y_raw would silently come up
        # short and break the np.stack below with a shape mismatch.
        n_win = n - context_window - forecast_steps
        if n_win <= 0:
            print(f"   ⚠️ {symbol}: not enough history ({n} rows) for CTX={context_window}+FORECAST={forecast_steps} — skipping")
            continue

        for i in range(0, n_win, stride):
            x_raw = data[i: i + context_window]
            # y_raw starts at the REAL entry day (i+context_window), not
            # "today" (i+context_window-1) - a real, previously-unnoticed
            # bug: SovereignLoss/SovereignAccuracy anchor their "correct
            # direction" comparison on row 0 of y_raw, and the real
            # trade-outcome/reasoning label below anchors on entry_idx's
            # close. Those used to be ONE DAY APART (today vs. the actual
            # entry day) - the prediction head and the reasoning head were
            # being trained to agree on direction while measuring it from
            # two different reference prices, undermining the whole
            # "multiple heads must agree" design ml_strategy.py depends on.
            # Anchoring both on the same entry_idx close fixes that.
            y_raw = data[i + context_window: i + context_window + forecast_steps + 1]

            x_scaled, local_mean, local_std = apply_dls(x_raw)
            y_scaled = np.clip((y_raw - local_mean) / local_std, -5.0, 5.0).astype("float32")

            all_X.append(x_scaled)
            # Channels: [close, volatility, volume, open, high, low] - close
            # stays channel 0 (SovereignLoss's direction term and every
            # inference site key on that position specifically).
            yc, yv, yu = y_scaled[:, t_close], y_scaled[:, t_vol], y_scaled[:, t_volu]
            yo, yh, yl = y_scaled[:, t_open], y_scaled[:, t_high], y_scaled[:, t_low]
            all_Y_pred.append(np.stack([yc, yv, yu, yo, yh, yl], axis=-1))
            all_end_dates.append(dates[i + context_window - 1])

            # Real trade outcome: enter long at the day AFTER the window ends
            # (matching swing_backtest/engine.py's next-day-open fill), walk
            # real high/low for HARD_TARGET_PCT/HARD_STOP_LOSS_PCT.
            #
            # Real bug found and fixed: this used to read t_close here, while
            # the comment directly above (and swing_backtest/engine.py's
            # actual code - verified by reading it: `price = entry_row["open"]`)
            # both say the real fill is the entry day's OPEN. Every reasoning
            # label (LONG/SHORT/FEE_TRAP/NOISE) was being classified against
            # a price the real backtest/live engine never actually pays -
            # overnight gaps between a day's close and the next day's open
            # are common and often meaningful for NSE equities, so this
            # wasn't a rounding-error-sized mismatch.
            entry_idx = i + context_window
            entry_price = float(data[entry_idx, t_open]) if entry_idx < n else float(data[-1, t_open])
            # +1: swing_backtest/engine.py still checks stop/target on the
            # day holding_days == max_holding BEFORE force-exiting on that
            # same day (elif chain: hit_stop/hit_target checked first, only
            # falling through to hit_max_holding if neither hits) - so a
            # real trade gets forecast_steps+1 days of stop/target
            # opportunity (holding_days 0..forecast_steps inclusive), not
            # forecast_steps. Without the +1 here, a trade that only clears
            # its target on that last day is a real WIN in the backtest/live
            # engine but got labeled FEE_TRAP/NOISE in training - never
            # seeing the day the real engine still gives it a chance on.
            high_path = data[entry_idx: entry_idx + forecast_steps + 1, t_high]
            low_path = data[entry_idx: entry_idx + forecast_steps + 1, t_low]

            # GPT-style next-candle target: the REAL next day's return
            # (ret_1d at entry_idx = tomorrow's close vs today's close),
            # tokenized after all windows are collected (needs the full
            # train-portion distribution first to fit the vocabulary).
            all_next_ret_raw.append(float(data[entry_idx, t_ret1d]) if entry_idx < n else 0.0)

            tp_long, sl_long = entry_price * (1 + HARD_TARGET_PCT), entry_price * (1 - HARD_STOP_LOSS_PCT)
            tp_short, sl_short = entry_price * (1 - HARD_TARGET_PCT), entry_price * (1 + HARD_STOP_LOSS_PCT)

            long_viable = _hits_target_before_stop(high_path, low_path, tp_long, sl_long, +1)
            short_viable = _hits_target_before_stop(high_path, low_path, tp_short, sl_short, -1)

            move_pct = (high_path.max() - entry_price) / (entry_price + 1e-9) if len(high_path) else 0.0
            move_pct_down = (entry_price - low_path.min()) / (entry_price + 1e-9) if len(low_path) else 0.0

            if long_viable and not short_viable:
                r = 0   # SOVEREIGN_LONG
            elif short_viable and not long_viable:
                r = 1   # SOVEREIGN_SHORT
            elif long_viable and short_viable:
                r = 2   # FEE_TRAP (choppy/ambiguous — both directions cleared)
            elif max(move_pct, move_pct_down) > FEE_RATE:
                r = 2   # FEE_TRAP (real move, but never cleanly cleared target-before-stop)
            else:
                r = 3   # NOISE

            all_Y_reason.append(r)
            all_Y_cert.append(1.0 if r in (0, 1) else 0.0)

        print(f"      {symbol}: {len(range(0, n_win, stride)):,} windows (stride={stride})")

    if not all_X:
        raise ValueError("No usable windows built — check data_by_symbol has enough history per symbol.")

    X_all = np.array(all_X, dtype="float32")
    Y_pred = np.array(all_Y_pred, dtype="float32")
    Y_cert = np.array(all_Y_cert, dtype="float32").reshape(-1, 1)
    Y_reason = np.array(all_Y_reason, dtype="int32").reshape(-1, 1)
    next_ret_raw = np.array(all_next_ret_raw, dtype="float32")
    end_dates = pd.to_datetime(all_end_dates)

    # Chronological split BY DATE across the pooled multi-symbol set — a
    # window from any symbol ending after the cutoff never lands in train.
    order = np.argsort(end_dates.values)
    X_all, Y_pred, Y_cert, Y_reason = X_all[order], Y_pred[order], Y_cert[order], Y_reason[order]
    next_ret_raw = next_ret_raw[order]
    end_dates_sorted = end_dates.values[order]

    n_win_total = len(X_all)
    tr_end_raw = int(n_win_total * 0.8)
    va_end = int(n_win_total * 0.9)

    # Purge gap between train and validation (standard technique for
    # overlapping-window time series - see "purged cross-validation").
    # Consecutive windows share up to context_window+forecast_steps-1 days
    # of their footprint (e.g. 79/80 days here), and each window's real
    # trade-outcome LABEL depends on forecast_steps days AFTER its own
    # window-end date. Without a gap, a training window sitting right at
    # the train/val boundary has a label that was computed from price
    # action inside what's supposed to be the held-out validation period -
    # real look-ahead leakage, not just window-overlap noise. Purge any
    # training window whose own label horizon (end_date + forecast_steps
    # trading days, approximated with a generous calendar-day buffer for
    # weekends/holidays) reaches into the validation period.
    val_start_date = end_dates_sorted[tr_end_raw]
    embargo = pd.Timedelta(days=int(forecast_steps * 1.6) + 5)
    purge_cutoff = val_start_date - embargo
    tr_end = int(np.searchsorted(end_dates_sorted[:tr_end_raw], purge_cutoff, side="right"))
    n_purged = tr_end_raw - tr_end
    if n_purged > 0:
        print(f"   ✂️  Purged {n_purged:,} training windows within the embargo gap "
              f"(their forecast label reached into the validation period)")

    label_counts = np.bincount(Y_reason[:tr_end].ravel(), minlength=4).astype(float)
    print(f"   ✅ Pooled {n_win_total:,} windows across {len(data_by_symbol)} symbols. "
          f"Label dist (train): LONG={int(label_counts[0]):,} SHORT={int(label_counts[1]):,} "
          f"FEE_TRAP={int(label_counts[2]):,} NOISE={int(label_counts[3]):,}")

    # GPT-style next-candle vocabulary - fit ONLY on the train portion (same
    # no-leakage discipline as everything else here), then used to tokenize
    # train AND val so both speak the same vocabulary.
    bin_edges, bin_centers, vocab_size = fit_return_vocab(next_ret_raw[:tr_end], vocab_size=32)
    Y_next_token = tokenize_returns(next_ret_raw, bin_edges).reshape(-1, 1)
    print(f"   🔤 Next-candle vocabulary: {vocab_size} tokens (quantile-binned from train-portion returns)")

    np.random.seed(42)
    shuffled_idx = np.random.permutation(tr_end)
    X_train_shuffled = X_all[:tr_end][shuffled_idx]
    Y_pred_shuffled = Y_pred[:tr_end][shuffled_idx]
    Y_cert_shuffled = Y_cert[:tr_end][shuffled_idx]
    Y_reason_shuffled = Y_reason[:tr_end][shuffled_idx]
    Y_next_token_shuffled = Y_next_token[:tr_end][shuffled_idx]

    # Single-input model - pass the raw tensor (not a {"market_input": ...}
    # dict). Keras 3 unwraps a single-key dict to a bare tensor internally
    # anyway, which otherwise triggers a harmless but noisy "structure of
    # inputs doesn't match" warning every step.
    #
    # .shuffle() here (not just the one-time np.random.permutation above) -
    # a real gap found during an overfitting audit: without a per-epoch
    # reshuffle, train.py's model.fit(shuffle=False, ...) meant every epoch
    # replayed training batches in the EXACT SAME order, every time. Static
    # batch order/composition across epochs is a known contributor to
    # training instability (the optimizer can end up correlated with a
    # fixed batch sequence instead of seeing genuinely fresh randomization
    # each epoch, which is part of what makes stochastic gradient descent
    # behave well). reshuffle_each_iteration=True (the default) gives a
    # fresh order every epoch; buffer_size = the full train set since it's
    # already fully in memory as numpy arrays, not streamed from disk.
    tr_ds = (
        tf.data.Dataset.from_tensor_slices((
            X_train_shuffled,
            {"prediction": Y_pred_shuffled, "certainty": Y_cert_shuffled, "reasoning": Y_reason_shuffled,
             "next_candle": Y_next_token_shuffled}))
        .shuffle(buffer_size=len(X_train_shuffled), seed=42, reshuffle_each_iteration=True)
        .batch(batch_size).prefetch(tf.data.AUTOTUNE)
    )
    va_ds = (
        tf.data.Dataset.from_tensor_slices((
            X_all[tr_end:va_end],
            {"prediction": Y_pred[tr_end:va_end], "certainty": Y_cert[tr_end:va_end], "reasoning": Y_reason[tr_end:va_end],
             "next_candle": Y_next_token[tr_end:va_end]}))
        .batch(batch_size).prefetch(tf.data.AUTOTUNE)
    )

    steps_tr = max(1, tr_end // batch_size)
    steps_va = max(1, (va_end - tr_end) // batch_size)
    print(f"   📊 Stream ready: {tr_end:,} train | {va_end - tr_end:,} val windows. Steps/epoch: {steps_tr}")

    return {
        "tr_ds": tr_ds, "va_ds": va_ds, "steps_tr": steps_tr, "steps_va": steps_va,
        "n_features": len(features), "label_counts": label_counts,
        "vocab_size": vocab_size, "bin_edges": bin_edges, "bin_centers": bin_centers,
    }
