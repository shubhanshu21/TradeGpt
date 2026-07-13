"""Swing paper trading: unlike intraday paper trading (which polls every
few seconds during market hours), swing positions are held for days, so
this only needs to check once per trading day - whether to exit an open
position (stop/target/max-holding, checked against the latest completed
daily candle) or enter a new one (signal detected on the latest completed
daily candle, filled at the current live price).

Meant to run as a long-lived background process: it sleeps until the next
check time and repeats, logging every virtual trade to CSV. No real
orders are ever placed - same "virtual money only" guarantee as
paper_trading/engine.py.
"""
import threading
from datetime import timedelta

import pandas as pd

from exchange.time_utils import now_ist
from swing_paper_trading.portfolio import SwingPortfolio

CHECK_TIME = (9, 20)  # IST, shortly after market open - use today's live price as the fill


class SwingPaperTradingEngine:
    def __init__(self, cfg: dict, strategies: list, broker, symbols: list,
                 stop_event: threading.Event = None):
        self.cfg = cfg
        self.strategies = {s.name: s for s in strategies}
        self.broker = broker
        self.symbols = symbols
        self.portfolio = SwingPortfolio(cfg, cfg["paper_trading"].get(
            "swing_log_file", "logs/swing_paper_trades.csv"))
        self._bars_held = {}       # symbol -> count of daily checks since entry
        self._position_strategy = {}  # symbol -> which strategy opened it
        self._acted_signal_date = {}  # (strategy, symbol) -> last date acted on
        self._warned_no_model = set()  # symbols with no trained checkpoint yet - warn once, not every cycle
        self.stop_event = stop_event or threading.Event()
        self.last_error = None
        self.last_check_date = None

    def run_forever(self):
        print(f"Swing paper trading started - strategies={list(self.strategies)}, "
              f"symbols={self.symbols}")
        while not self.stop_event.is_set():
            now = now_ist()
            is_weekday = now.weekday() < 5
            already_checked_today = self.last_check_date == now.date()
            past_check_time = (now.hour, now.minute) >= CHECK_TIME

            if is_weekday and past_check_time and not already_checked_today:
                try:
                    self.check_once()
                except Exception as exc:
                    self.last_error = str(exc)
                    print(f"ERROR during daily check: {exc}")
                finally:
                    # Mark today as checked either way - a holiday shouldn't
                    # be retried every 30 minutes for the rest of the day.
                    self.last_check_date = now.date()

            self.stop_event.wait(1800)  # re-check every 30 min whether it's time

    def _market_open_today(self) -> bool:
        """Weekday alone isn't enough - NSE also closes for holidays
        (Diwali, Republic Day, etc.) that don't follow a fixed calendar
        rule. Rather than maintaining a holiday list that goes stale every
        year, ask the broker directly: fetch today's MINUTE-level data (not
        daily - today's daily candle isn't finalized this early in the
        session) for a reference symbol. If the market opened, a few
        minute candles already exist a few minutes after 9:15 - if it's a
        holiday, there will be none.
        """
        reference_symbol = self.symbols[0]
        now = now_ist().replace(tzinfo=None)
        day_start = now.replace(hour=9, minute=15, second=0, microsecond=0)
        try:
            df = self.broker.historical_candles(reference_symbol, "minute", day_start, now)
        except Exception:
            return True  # can't verify - proceed rather than silently never checking
        return not df.empty

    def check_once(self):
        today = now_ist().date()
        if not self._market_open_today():
            print(f"[{today}] Market did not trade today (holiday) - skipping check.")
            return
        print(f"[{today}] Running daily swing check...")
        self._check_exits(today)
        self._check_entries(today)

    def _history(self, symbol) -> pd.DataFrame:
        end = now_ist().replace(tzinfo=None)
        start = end - timedelta(days=800)  # enough lookback for 200-day MAs etc
        return self.broker.historical_candles(symbol, self.cfg["swing"].get("interval", "day"), start, end)

    def _check_exits(self, today):
        for symbol in list(self.portfolio.positions.keys()):
            df = self._history(symbol)
            if df.empty:
                continue
            last = df.iloc[-1]
            self._bars_held[symbol] = self._bars_held.get(symbol, 0) + 1

            reason = self.portfolio.check_exit(
                symbol, last["high"], last["low"], self._bars_held[symbol])
            if reason:
                exit_price = last["close"]
                if reason == "stop_loss":
                    exit_price = self.portfolio.positions[symbol].stop_price
                elif reason == "target":
                    exit_price = self.portfolio.positions[symbol].target_price
                net = self.portfolio.close_position(
                    symbol, exit_price, today, reason, self._bars_held[symbol])
                self._bars_held.pop(symbol, None)
                self._position_strategy.pop(symbol, None)
                print(f"  [EXIT] {symbol} {reason} @ {exit_price:.2f} net_pnl={net:.2f}")

    def _check_entries(self, today):
        if not self.portfolio.can_open_new():
            return
        for symbol in self.symbols:
            if symbol in self.portfolio.positions:
                continue
            if not self.portfolio.can_open_new():
                break

            df = self._history(symbol)
            if len(df) < 210:  # need enough for the slowest indicator (200-day MA)
                continue

            for strategy_name, strategy in self.strategies.items():
                try:
                    signals = strategy.generate_signals(df, symbol=symbol)
                except FileNotFoundError:
                    if symbol not in self._warned_no_model:
                        print(f"  [SKIP] {symbol}: no trained checkpoint yet for strategy {strategy_name}")
                        self._warned_no_model.add(symbol)
                    continue
                last = signals.iloc[-1]
                if last["signal"] == 0:
                    continue
                key = (strategy_name, symbol)
                if self._acted_signal_date.get(key) == last["date"]:
                    continue

                prices = self.broker.ltp([symbol])
                price = prices.get(symbol)
                if price is None:
                    continue

                direction = int(last["signal"])
                pos = self.portfolio.open_position(symbol, direction, price, today)
                self._acted_signal_date[key] = last["date"]
                if pos:
                    self._bars_held[symbol] = 0
                    self._position_strategy[symbol] = strategy_name
                    side = "LONG" if direction == 1 else "SHORT"
                    print(f"  [ENTRY] {symbol} {side} via {strategy_name} "
                          f"qty={pos.quantity} @ {price:.2f}")
                break  # first strategy to signal wins the slot for this symbol

    def get_status(self) -> dict:
        return {
            "strategies": list(self.strategies.keys()),
            "symbols": self.symbols,
            "running": not self.stop_event.is_set(),
            "last_check_date": str(self.last_check_date) if self.last_check_date else None,
            "positions": [
                {
                    "symbol": s, "direction": p.direction, "quantity": p.quantity,
                    "entry_price": p.entry_price, "entry_date": str(p.entry_date),
                    "stop_price": p.stop_price, "target_price": p.target_price,
                    "strategy": self._position_strategy.get(s),
                    "bars_held": self._bars_held.get(s, 0),
                }
                for s, p in self.portfolio.positions.items()
            ],
            "total_realized_pnl": self.portfolio.realized_pnl_total,
            "last_error": self.last_error,
        }
