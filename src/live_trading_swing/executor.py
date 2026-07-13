"""Swing live trading executor - places REAL delivery (CNC) orders via
whichever broker is configured. Structurally mirrors
swing_paper_trading/engine.py (same signal/entry/exit logic) so a strategy
that behaved a certain way in paper trading behaves the same way here -
the only things that change are broker.place_order() actually reaching the
exchange, and a broker-side stop-loss/target order being attempted too.

SAFETY GATES (all must pass before a single order is placed):
  1. config/settings.yaml -> live_trading.enabled must be true.
  2. You must type the exact confirmation phrase at startup.
  3. If live_trading.confirm_each_order is true, every individual order
     needs a y/n confirmation at the terminal.

CLI-only by design, same reasoning as before: a "start live trading" web
button is exactly the kind of one-click, hard-to-reverse, real-money
action that should not exist in a UI. There is no API/dashboard control
for this - see api/main.py's /api/live/status.

STOP-LOSS SAFETY NET: after each entry, this tries to place a broker-side
GTT/OCO order (common/brokers/base.py: place_gtt_oco) so the stop-loss and
target live on the broker's servers, not just in this process's memory.
Only Zerodha's adapter implements this so far (and it is UNVERIFIED against
a live account - see zerodha_broker.py's docstring). Upstox/Dhan fall back
to this process polling prices once per day, same limitation the old
intraday executor had - if this process is down when a stop should have
fired, the position has no protection until it's checked again. For a
position held days to weeks, that's a real, not theoretical, risk.
"""
import csv
import sys
import threading
from datetime import timedelta
from pathlib import Path

import pandas as pd

from exchange.time_utils import now_ist
from live_trading_swing.risk_manager import RiskManager

CHECK_TIME = (9, 20)  # IST, shortly after market open
CONFIRMATION_PHRASE = "I UNDERSTAND THE RISK"


class SwingLiveExecutor:
    def __init__(self, cfg: dict, strategies: list, broker, symbols: list,
                 stop_event: threading.Event = None):
        self.cfg = cfg
        self.strategies = {s.name: s for s in strategies}
        self.broker = broker
        self.symbols = symbols
        self.risk_manager = RiskManager(cfg)
        self.positions = {}  # symbol -> dict(direction, quantity, entry_price, entry_date, stop_price, target_price, strategy, gtt_id)
        self._bars_held = {}
        self._acted_signal_date = {}
        self._warned_no_model = set()  # symbols with no trained checkpoint yet - warn once, not every cycle
        self.stop_event = stop_event or threading.Event()
        self.last_check_date = None

        self.log_path = Path(cfg["live_trading"]["log_file"])
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.log_path.exists():
            with open(self.log_path, "w", newline="") as f:
                csv.writer(f).writerow([
                    "entry_date", "symbol", "strategy", "direction", "quantity",
                    "entry_price", "exit_date", "exit_price", "holding_days",
                    "gross_pnl", "exit_reason", "entry_order_id", "exit_order_id",
                ])

    def _gate_or_exit(self):
        if not self.cfg["live_trading"]["enabled"]:
            print("live_trading.enabled is false in config/settings.yaml. "
                  "This is a deliberate safety default - flip it to true only "
                  "once swing paper trading has proven itself over real time.")
            sys.exit(1)

        print("=" * 70)
        print("SWING LIVE TRADING - REAL ORDERS, REAL MONEY")
        print(f"Broker: {self.broker.name}  Strategies: {list(self.strategies)}")
        print(f"Symbols: {self.symbols}")
        print(f"Capital per trade: Rs {self.cfg['swing']['capital_per_trade']:,}")
        print(f"Max concurrent positions: {self.cfg['swing']['max_concurrent_positions']}")
        print(f"Cumulative loss kill switch: {self.cfg['swing']['max_total_loss_pct'] * 100:.1f}% of "
              f"Rs {self.cfg['swing']['initial_capital']:,}")
        print("=" * 70)
        typed = input(f"Type exactly '{CONFIRMATION_PHRASE}' to proceed: ").strip()
        if typed != CONFIRMATION_PHRASE:
            print("Confirmation phrase did not match. Aborting - no orders placed.")
            sys.exit(1)

    def _confirm_order(self, description: str) -> bool:
        if not self.cfg["live_trading"].get("confirm_each_order", True):
            return True
        resp = input(f"Place order? {description} [y/N]: ").strip().lower()
        return resp == "y"

    def run(self):
        self._gate_or_exit()
        print(f"Swing live trading started with strategies={list(self.strategies)}")

        while not self.stop_event.is_set():
            now = now_ist()
            is_weekday = now.weekday() < 5
            already_checked_today = self.last_check_date == now.date()
            past_check_time = (now.hour, now.minute) >= CHECK_TIME

            if is_weekday and past_check_time and not already_checked_today:
                try:
                    self.check_once()
                except Exception as exc:
                    print(f"ERROR during daily check: {exc}")
                finally:
                    self.last_check_date = now.date()

            self.stop_event.wait(1800)

    def _market_open_today(self) -> bool:
        """See swing_paper_trading/engine.py's version of this method for
        why a minute-data probe, not a hardcoded holiday calendar.
        """
        reference_symbol = self.symbols[0]
        now = now_ist().replace(tzinfo=None)
        day_start = now.replace(hour=9, minute=15, second=0, microsecond=0)
        try:
            df = self.broker.historical_candles(reference_symbol, "minute", day_start, now)
        except Exception:
            return True
        return not df.empty

    def check_once(self):
        today = now_ist().date()
        if not self._market_open_today():
            print(f"[{today}] Market did not trade today (holiday) - skipping check.")
            return
        print(f"[{today}] Running daily live swing check...")
        self._check_exits(today)
        self._check_entries(today)

    def _history(self, symbol) -> pd.DataFrame:
        end = now_ist().replace(tzinfo=None)
        start = end - timedelta(days=800)
        return self.broker.historical_candles(symbol, self.cfg["swing"].get("interval", "day"), start, end)

    def _still_holding(self, symbol: str) -> bool:
        """Best-effort check that we haven't already been exited by a
        broker-side GTT before this process tries to place its own exit -
        avoids firing a redundant order. Broker position dict shapes
        differ, so this errs toward "yes, still holding" if it can't tell.
        """
        try:
            positions = self.broker.positions()
        except Exception:
            return True
        for p in positions:
            sym = p.get("tradingsymbol") or p.get("symbol") or p.get("trading_symbol")
            qty = p.get("quantity") or p.get("net_quantity") or p.get("netQty") or 0
            if sym == symbol and qty:
                return True
        return not positions  # if we can't parse the shape at all, don't assume closed

    def _check_exits(self, today):
        for symbol in list(self.positions.keys()):
            df = self._history(symbol)
            if df.empty:
                continue
            last = df.iloc[-1]
            self._bars_held[symbol] = self._bars_held.get(symbol, 0) + 1
            pos = self.positions[symbol]

            reason = None
            if pos["direction"] == 1:
                if last["low"] <= pos["stop_price"]:
                    reason = "stop_loss"
                elif last["high"] >= pos["target_price"]:
                    reason = "target"
            else:
                if last["high"] >= pos["stop_price"]:
                    reason = "stop_loss"
                elif last["low"] <= pos["target_price"]:
                    reason = "target"
            if self._bars_held[symbol] >= self.cfg["swing"]["max_holding_days"]:
                reason = reason or "max_holding_period"

            if reason:
                self._exit_position(symbol, reason, today)

    def _exit_position(self, symbol, reason, today):
        pos = self.positions.pop(symbol)
        self._bars_held.pop(symbol, None)

        if not self._still_holding(symbol):
            print(f"  [SKIP EXIT] {symbol} already closed (likely by broker-side GTT) - not re-ordering.")
            return

        exit_direction = -pos["direction"]
        side = "SELL" if pos["direction"] == 1 else "BUY"
        if not self._confirm_order(f"{side} {symbol} qty={pos['quantity']} ({reason})"):
            print(f"  Skipped exit for {symbol} (not confirmed) - position left open, will retry next check.")
            self.positions[symbol] = pos  # put it back, we didn't actually exit
            self._bars_held[symbol] = self._bars_held.get(symbol, 0)
            return

        order_id = self.broker.place_order(
            symbol, exit_direction, pos["quantity"], order_type="MARKET",
            product=self.cfg["live_trading"]["product"])

        prices = self.broker.ltp([symbol])
        exit_price = prices.get(symbol, pos["entry_price"])
        gross = (exit_price - pos["entry_price"]) * pos["quantity"] * pos["direction"]
        self.risk_manager.record_realized(gross)

        with open(self.log_path, "a", newline="") as f:
            csv.writer(f).writerow([
                pos["entry_date"], symbol, pos["strategy"], pos["direction"], pos["quantity"],
                pos["entry_price"], today, exit_price, self._bars_held.get(symbol, 0),
                round(gross, 2), reason, pos["entry_order_id"], order_id,
            ])
        print(f"  [EXIT] {symbol} {reason} @ ~{exit_price:.2f} order_id={order_id} "
              f"gross_pnl~{gross:.2f} (excl. exact costs)")

    def _check_entries(self, today):
        if not self.risk_manager.can_open_new(len(self.positions)):
            return
        for symbol in self.symbols:
            if symbol in self.positions:
                continue
            if not self.risk_manager.can_open_new(len(self.positions)):
                break

            df = self._history(symbol)
            if len(df) < 210:
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

                direction = int(last["signal"])
                side = "BUY" if direction == 1 else "SELL"
                if not self._confirm_order(f"{side} {symbol} via {strategy_name} (signal-based entry)"):
                    print(f"  Skipped entry for {symbol} (not confirmed).")
                    self._acted_signal_date[key] = last["date"]
                    continue

                self._enter_position(symbol, direction, strategy_name)
                self._acted_signal_date[key] = last["date"]
                break

    def _enter_position(self, symbol, direction, strategy_name):
        prices = self.broker.ltp([symbol])
        price = prices.get(symbol)
        if price is None:
            return
        capital_per_trade = self.cfg["swing"]["capital_per_trade"]
        quantity = int(capital_per_trade // price)
        if quantity <= 0:
            return

        order_id = self.broker.place_order(
            symbol, direction, quantity, order_type="MARKET",
            product=self.cfg["live_trading"]["product"])

        stop_pct = self.cfg["swing"]["hard_stop_loss_pct"]
        target_pct = self.cfg["swing"]["hard_target_pct"]
        if direction == 1:
            stop_price = price * (1 - stop_pct)
            target_price = price * (1 + target_pct)
        else:
            stop_price = price * (1 + stop_pct)
            target_price = price * (1 - target_pct)

        today = now_ist().date()
        self.positions[symbol] = {
            "direction": direction, "quantity": quantity, "entry_price": price,
            "entry_date": today, "stop_price": stop_price, "target_price": target_price,
            "strategy": strategy_name, "entry_order_id": order_id,
        }
        self._bars_held[symbol] = 0

        side = "LONG" if direction == 1 else "SHORT"
        print(f"[ENTRY] {symbol} {side} via {strategy_name} qty={quantity} "
              f"@ ~{price:.2f} order_id={order_id}")

        if self.cfg["live_trading"].get("use_broker_stop_loss", True):
            try:
                gtt_id = self.broker.place_gtt_oco(
                    symbol, direction, quantity, price, stop_price, target_price)
                self.positions[symbol]["gtt_id"] = gtt_id
                print(f"  Broker-side GTT/OCO placed (id={gtt_id}) - stop={stop_price:.2f} target={target_price:.2f}")
            except NotImplementedError as exc:
                print(f"  WARNING: no broker-side stop-loss for {self.broker.name} ({exc}). "
                      f"Relying solely on this process's daily polling for {symbol} - "
                      f"if this process is down when the stop should fire, it won't.")
            except Exception as exc:
                print(f"  WARNING: broker-side GTT placement failed ({exc}). "
                      f"Falling back to this process's daily polling only.")

    def get_status(self) -> dict:
        return {
            "strategies": list(self.strategies.keys()),
            "symbols": self.symbols,
            "running": not self.stop_event.is_set(),
            "last_check_date": str(self.last_check_date) if self.last_check_date else None,
            "positions": [
                {"symbol": s, **{k: v for k, v in p.items() if k not in ("entry_date",)},
                 "entry_date": str(p["entry_date"]), "bars_held": self._bars_held.get(s, 0)}
                for s, p in self.positions.items()
            ],
            "kill_switch_tripped": self.risk_manager.kill_switch_tripped,
            "total_realized_pnl": self.risk_manager.realized_pnl_total,
        }
