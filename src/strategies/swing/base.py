"""Swing strategy contract - deliberately different from strategies/base.py
(the intraday one). Intraday strategies see one day's candles at a time and
must finish flat by end of day; swing strategies see a symbol's ENTIRE
daily price history at once and can hold a position for many days, because
indicators like a 200-day moving average or a 52-week high need that much
lookback, and the whole point of swing trading is multi-day holds.

The swing_backtest engine (not the strategy) owns position sizing,
stop-loss/target, max holding period, and cost accounting - same
separation-of-concerns as the intraday side, so all 15 strategies are
judged on a level playing field.
"""
from abc import ABC, abstractmethod

import pandas as pd


class SwingStrategy(ABC):
    name: str = "base"

    @abstractmethod
    def generate_signals(self, df: pd.DataFrame, symbol: str = None) -> pd.DataFrame:
        """df has columns: date, open, high, low, close, volume - one
        symbol's full daily history, sorted ascending, index 0..n-1.
        symbol identifies which one (needed by strategies that keep a
        separate model per symbol, e.g. MLSwingStrategy).

        Must return df with an added integer 'signal' column:
            1  -> enter long at the NEXT day's open
           -1  -> enter short at the NEXT day's open (rarely used for
                  delivery/CNC in practice, but kept symmetric for comparison)
            0  -> no signal

        A signal at row i must only use information available up to and
        including row i (no lookahead) - the engine fills at row i+1's open.
        """
        raise NotImplementedError
