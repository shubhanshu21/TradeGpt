"""ML-based swing strategy - loads the neural network trained by train.py
and implements the same SwingStrategy interface as the 3 rule-based
strategies, so it plugs into the same backtest engine, paper trading, and
live trading.

Signal logic requires the certainty threshold, the reasoning class, the
predicted price trajectory's direction, AND the GPT-style next-candle
head's implied direction to all agree - three independent heads reading
the same 60-day window, not just one head's opinion.
"""
import numpy as np
import pandas as pd

from strategies.swing.base import SwingStrategy
from config.sovereign_config import CERT_THRESHOLD


class MLSwingStrategy(SwingStrategy):
    """Each symbol has its OWN trained model (models/<SYMBOL>/hydra_best.keras),
    not one model shared/pooled across symbols - see train.py, which now
    trains one checkpoint per symbol. Models are lazy-loaded and cached per
    symbol as they're first needed."""
    name = "ml_swing"

    def __init__(self, checkpoint_name: str = "hydra_best.keras", cert_threshold: float = CERT_THRESHOLD):
        self.cert_threshold = cert_threshold
        self._checkpoint_name = checkpoint_name
        self._models = {}   # symbol -> loaded keras model
        self._shapes = {}   # symbol -> model_shape.pkl contents
        self._benchmark = "unloaded"  # lazy-loaded once, shared across all symbols (see generate_signals)
        self._vix = "unloaded"  # lazy-loaded once, shared across all symbols (see generate_signals)

    def _load(self, symbol: str):
        """Lazy load per symbol - so importing this module doesn't require
        every symbol to have a trained checkpoint yet."""
        if symbol in self._models:
            return
        import pickle
        from pathlib import Path
        from core.hydra import build_kraken, init_kraken_hardware

        models_dir = Path(__file__).resolve().parent.parent.parent.parent / "models" / symbol
        shape_path = models_dir / "model_shape.pkl"
        ckpt_path = models_dir / self._checkpoint_name
        if not shape_path.exists() or not ckpt_path.exists():
            raise FileNotFoundError(
                f"No trained checkpoint found for {symbol} ({ckpt_path}). "
                f"Train it first: python3 train.py --symbols {symbol}")

        with open(shape_path, "rb") as f:
            shape = pickle.load(f)

        init_kraken_hardware()
        # vocab_size isn't stored directly in model_shape.pkl, but
        # len(bin_centers) IS the actual trained vocab size (fit_return_vocab
        # can return fewer tokens than requested when quantile edges
        # collapse - real, common, not rare: observed 31 instead of the
        # requested 32 on real data during this session's own testing).
        # Passing the wrong vocab_size here silently breaks load_weights
        # with a shape mismatch the moment a symbol's real vocab != 32.
        vocab_size = len(shape["bin_centers"]) if shape.get("bin_centers") is not None else 32
        model = build_kraken(
            n_features=shape["n_features"],
            context_window=shape["context_window"],
            forecast_steps=shape["forecast_steps"],
            vocab_size=vocab_size,
        )
        model.load_weights(str(ckpt_path))
        self._models[symbol] = model
        self._shapes[symbol] = shape

    def generate_signals(self, df: pd.DataFrame, symbol: str = None, batch_size: int = 32) -> pd.DataFrame:
        from data.preprocess import compute_indicators, build_feature_cols, apply_dls, load_benchmark_index, load_vix_index

        if symbol is None:
            raise ValueError("MLSwingStrategy.generate_signals requires symbol= (one model per symbol)")
        self._load(symbol)
        model = self._models[symbol]
        ctx = self._shapes[symbol]["context_window"]

        if self._benchmark == "unloaded":
            self._benchmark = load_benchmark_index()
        if self._vix == "unloaded":
            self._vix = load_vix_index()

        df = df.copy()
        df["signal"] = 0
        if len(df) <= ctx + 5:
            return df

        df_feat = compute_indicators(df, benchmark_df=self._benchmark, vix_df=self._vix)
        features = build_feature_cols()
        data = df_feat[features].values.astype("float32")

        n = len(data)
        # Window ending at row i-1 (inclusive) - only information available up
        # to and including "today" (row i-1), matching the base class's
        # no-lookahead contract (a signal at row i-1 fills at row i's open).
        idxs = list(range(ctx, n))
        if not idxs:
            return df

        # Precompute every window's DLS-scaled tensor up front, then run the
        # model in batches instead of one day at a time - thousands of
        # single-example calls (each paying full Python/graph-dispatch
        # overhead) made a full-history backtest impractically slow.
        windows = np.stack([apply_dls(data[i - ctx: i])[0] for i in idxs]).astype("float32")

        all_pred, all_cert, all_reason, all_next_tok = [], [], [], []
        for start in range(0, len(windows), batch_size):
            batch = windows[start: start + batch_size]
            pred, certainty, reasoning, next_candle = model(batch, training=False)
            all_pred.append(pred.numpy())
            all_cert.append(certainty.numpy())
            all_reason.append(reasoning.numpy())
            all_next_tok.append(next_candle.numpy())
        pred = np.concatenate(all_pred, axis=0)
        certainty = np.concatenate(all_cert, axis=0)
        reasoning = np.concatenate(all_reason, axis=0)
        next_candle = np.concatenate(all_next_tok, axis=0)

        cert = certainty.mean(axis=1)                       # (n_windows,)
        reasoning_cls = np.argmax(reasoning, axis=1)         # 0=LONG 1=SHORT 2=FEE_TRAP 3=NOISE
        mean_move = pred[:, 1:, 0].mean(axis=1) - pred[:, 0, 0]
        price_dir = np.where(mean_move > 0, 1, np.where(mean_move < 0, -1, 0))
        reasoning_dir = np.where(reasoning_cls == 0, 1, -1)

        # GPT-style next-candle head: decode the predicted token back into an
        # implied direction (bin_centers[token] > 0 = up, < 0 = down) using
        # THIS symbol's own vocabulary from training, and require it to also
        # agree - a third independent confirmation on top of the trajectory
        # and reasoning heads, not a replacement for either.
        bin_centers = self._shapes[symbol].get("bin_centers")
        if bin_centers is not None and len(bin_centers) > 0:
            next_tok_id = np.argmax(next_candle, axis=1)
            next_tok_id = np.clip(next_tok_id, 0, len(bin_centers) - 1)
            next_candle_return = bin_centers[next_tok_id]
            next_candle_dir = np.where(next_candle_return > 0, 1, np.where(next_candle_return < 0, -1, 0))
        else:
            next_candle_dir = reasoning_dir  # no vocabulary saved (older checkpoint) - don't gate on it

        fire = (
            (cert >= self.cert_threshold)
            & np.isin(reasoning_cls, (0, 1))
            & (price_dir == reasoning_dir)
            & (next_candle_dir == reasoning_dir)
        )

        signal_arr = np.zeros(n, dtype="int64")
        target_rows = np.array(idxs) - 1   # signal at row i-1, matching the no-lookahead contract above
        signal_arr[target_rows[fire]] = reasoning_dir[fire]
        df["signal"] = signal_arr

        return df

    def explain_latest(self, df: pd.DataFrame, symbol: str) -> dict:
        """Plain-language snapshot of every real model output for the MOST
        RECENT available day only - built for the dashboard's "Today's
        Market View" card, not for trading. generate_signals() answers "did
        a trade fire on each historical day"; this answers "what does the
        model actually think about this stock right now, in words a
        non-technical user can read", decoding the SAME four model heads
        (prediction/certainty/reasoning/next_candle) generate_signals uses.
        """
        from data.preprocess import (compute_indicators, build_feature_cols, apply_dls,
                                      load_benchmark_index, load_vix_index,
                                      CANDLESTICK_PATTERNS, CANDLESTICK_PATTERN_NAMES)

        try:
            self._load(symbol)
        except FileNotFoundError:
            return {"available": False, "symbol": symbol,
                    "reason": "This stock's model hasn't finished training yet - check back later."}

        model = self._models[symbol]
        ctx = self._shapes[symbol]["context_window"]

        if self._benchmark == "unloaded":
            self._benchmark = load_benchmark_index()
        if self._vix == "unloaded":
            self._vix = load_vix_index()

        if len(df) < ctx:
            return {"available": False, "symbol": symbol,
                    "reason": f"Needs at least {ctx} days of price history, only have {len(df)}."}

        df_feat = compute_indicators(df, benchmark_df=self._benchmark, vix_df=self._vix)
        features = build_feature_cols()
        data = df_feat[features].values.astype("float32")

        x_raw = data[-ctx:]
        x_scaled, local_mean, local_std = apply_dls(x_raw)
        batch = x_scaled[np.newaxis, ...].astype("float32")

        pred, certainty, reasoning, next_candle = model(batch, training=False)
        pred = pred.numpy()[0]                          # (forecast_steps+1, 6)
        certainty = float(certainty.numpy()[0].mean())
        reasoning_probs = reasoning.numpy()[0]
        reasoning_cls = int(np.argmax(reasoning_probs))
        next_candle = next_candle.numpy()[0]

        # Un-scale the predicted close trajectory back into a real % move,
        # using THIS window's own DLS mean/std (same z-score convention
        # build_dataset_streaming trains on) - channel 0 is close.
        t_close = features.index('close')
        close_mean, close_std = float(local_mean[t_close]), float(local_std[t_close])
        pred_close_raw = pred[:, 0] * close_std + close_mean
        expected_move_pct = float((pred_close_raw[1:].mean() - pred_close_raw[0]) / (pred_close_raw[0] + 1e-9) * 100)

        reasoning_labels = {
            0: ("BUY setup", "The model sees a real long (buy) opportunity here."),
            1: ("SELL setup", "The model sees a real short (sell) opportunity here."),
            2: ("Choppy / unclear", "Price could move either way without a clean edge - not a reliable setup right now."),
            3: ("No signal", "Nothing unusual detected - an ordinary, quiet day for this stock."),
        }
        signal_label, signal_explanation = reasoning_labels.get(reasoning_cls, ("Unknown", ""))

        bin_centers = self._shapes[symbol].get("bin_centers")
        next_day_pct = None
        if bin_centers is not None and len(bin_centers) > 0:
            next_tok_id = int(np.clip(np.argmax(next_candle), 0, len(bin_centers) - 1))
            next_day_pct = round(float(bin_centers[next_tok_id]) * 100, 2)

        # Candlestick patterns detected on the most recent real candle only.
        last_row = df_feat.iloc[-1]
        patterns_detected = []
        for col_name, raw_name in zip(CANDLESTICK_PATTERNS, CANDLESTICK_PATTERN_NAMES):
            v = float(last_row.get(col_name, 0.0))
            if v > 0:
                patterns_detected.append({"name": raw_name.replace('_', ' ').title(), "bias": "Bullish"})
            elif v < 0:
                patterns_detected.append({"name": raw_name.replace('_', ' ').title(), "bias": "Bearish"})

        vix_z = float(last_row.get('vix_zscore_60d', 0.0))
        if vix_z > 2:
            market_mood = "High market-wide fear (India VIX well above its recent normal)"
        elif vix_z > 1:
            market_mood = "Elevated market-wide fear"
        elif vix_z < -1:
            market_mood = "Calmer than usual market-wide"
        else:
            market_mood = "Normal market mood"

        rel_strength_pct = round(float(last_row.get('rel_strength_20d', 0.0)) * 100, 2)

        return {
            "available": True,
            "symbol": symbol,
            "as_of_date": str(pd.Timestamp(df_feat['date'].iloc[-1]).date()),
            "last_close": round(float(df_feat['close'].iloc[-1]), 2),
            "signal_label": signal_label,
            "signal_explanation": signal_explanation,
            "certainty_pct": round(certainty * 100, 1),
            "meets_trade_threshold": bool(certainty >= self.cert_threshold and reasoning_cls in (0, 1)),
            "expected_move_over_holding_period_pct": round(expected_move_pct, 2),
            "next_day_expected_move_pct": next_day_pct,
            "candlestick_patterns": patterns_detected[:6],
            "market_mood": market_mood,
            "vix_zscore": round(vix_z, 2),
            "relative_strength_20d_pct": rel_strength_pct,
        }
