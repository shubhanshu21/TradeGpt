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
        model = build_kraken(
            n_features=shape["n_features"],
            context_window=shape["context_window"],
            forecast_steps=shape["forecast_steps"],
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
