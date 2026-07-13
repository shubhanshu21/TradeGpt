"""ML-based swing strategy - loads the neural network trained by train.py
and implements the same SwingStrategy interface as the 3 rule-based
strategies, so it plugs into the same backtest engine, paper trading, and
live trading.

Signal logic mirrors the certainty threshold + reasoning class + direction
agreement gating philosophy used throughout this project's live trading
path, so a signal fired here means the same thing it would mean anywhere
else in the system.
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

    def generate_signals(self, df: pd.DataFrame, symbol: str = None) -> pd.DataFrame:
        from data.preprocess import compute_indicators, build_feature_cols, apply_dls

        if symbol is None:
            raise ValueError("MLSwingStrategy.generate_signals requires symbol= (one model per symbol)")
        self._load(symbol)
        model = self._models[symbol]
        ctx = self._shapes[symbol]["context_window"]

        df = df.copy()
        df["signal"] = 0
        if len(df) <= ctx + 5:
            return df

        df_feat = compute_indicators(df)
        features = build_feature_cols()
        data = df_feat[features].values.astype("float32")

        n = len(data)
        for i in range(ctx, n):
            # Window ending at row i-1 (inclusive) - only information
            # available up to and including "today" (row i-1), matching the
            # base class's no-lookahead contract (a signal at row i-1 fills
            # at row i's open).
            x_raw = data[i - ctx: i]
            x_scaled, _, _ = apply_dls(x_raw)
            X_in = x_scaled[np.newaxis].astype("float32")

            pred, certainty, reasoning = model(X_in, training=False)
            cert = float(np.mean(certainty.numpy()[0]))
            reasoning_cls = int(np.argmax(reasoning.numpy()[0]))  # 0=LONG 1=SHORT 2=FEE_TRAP 3=NOISE

            if cert < self.cert_threshold or reasoning_cls not in (0, 1):
                continue

            p = pred.numpy()[0]
            p_anchor = p[0, 0]
            mean_move = float(np.mean(p[1:, 0] - p_anchor))
            price_dir = 1 if mean_move > 0 else (-1 if mean_move < 0 else 0)
            reasoning_dir = 1 if reasoning_cls == 0 else -1
            if price_dir != reasoning_dir:
                continue

            df.loc[df.index[i - 1], "signal"] = reasoning_dir

        return df
