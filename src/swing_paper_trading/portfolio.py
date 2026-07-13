"""Virtual multi-day position tracking for swing paper trading. Mirrors
paper_trading/portfolio.py's shape (so the CSV log format is familiar) but
uses delivery costs and day-granularity holding instead of intraday
stop-loss/target polling.
"""
import csv
from pathlib import Path

from backtest.delivery_costs import delivery_round_trip_cost


class SwingPosition:
    def __init__(self, symbol, direction, quantity, entry_price, entry_date,
                 stop_price, target_price):
        self.symbol = symbol
        self.direction = direction
        self.quantity = quantity
        self.entry_price = entry_price
        self.entry_date = entry_date
        self.stop_price = stop_price
        self.target_price = target_price


class SwingPortfolio:
    def __init__(self, cfg: dict, log_file: str):
        self.cfg = cfg
        self.positions: dict[str, SwingPosition] = {}
        self.realized_pnl_total = 0.0
        self.log_path = Path(log_file)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.log_path.exists():
            with open(self.log_path, "w", newline="") as f:
                csv.writer(f).writerow([
                    "entry_date", "symbol", "direction", "quantity", "entry_price",
                    "exit_date", "exit_price", "holding_days", "gross_pnl",
                    "costs", "net_pnl", "exit_reason",
                ])

    def can_open_new(self) -> bool:
        return len(self.positions) < self.cfg["swing"]["max_concurrent_positions"]

    def open_position(self, symbol, direction, entry_price, entry_date):
        capital_per_trade = self.cfg["swing"]["capital_per_trade"]
        stop_pct = self.cfg["swing"]["hard_stop_loss_pct"]
        target_pct = self.cfg["swing"]["hard_target_pct"]

        quantity = int(capital_per_trade // entry_price)
        if quantity <= 0:
            return None

        if direction == 1:
            stop_price = entry_price * (1 - stop_pct)
            target_price = entry_price * (1 + target_pct)
        else:
            stop_price = entry_price * (1 + stop_pct)
            target_price = entry_price * (1 - target_pct)

        position = SwingPosition(symbol, direction, quantity, entry_price,
                                  entry_date, stop_price, target_price)
        self.positions[symbol] = position
        return position

    def check_exit(self, symbol, day_high, day_low, holding_days) -> str | None:
        pos = self.positions.get(symbol)
        if pos is None:
            return None
        max_holding = self.cfg["swing"]["max_holding_days"]

        if pos.direction == 1:
            if day_low <= pos.stop_price:
                return "stop_loss"
            if day_high >= pos.target_price:
                return "target"
        else:
            if day_high >= pos.stop_price:
                return "stop_loss"
            if day_low <= pos.target_price:
                return "target"
        if holding_days >= max_holding:
            return "max_holding_period"
        return None

    def close_position(self, symbol, exit_price, exit_date, exit_reason, holding_days):
        pos = self.positions.pop(symbol)
        gross = (exit_price - pos.entry_price) * pos.quantity * pos.direction
        cost = delivery_round_trip_cost(pos.entry_price, exit_price, pos.quantity, pos.direction, self.cfg)
        net = gross - cost
        self.realized_pnl_total += net

        with open(self.log_path, "a", newline="") as f:
            csv.writer(f).writerow([
                pos.entry_date, symbol, pos.direction, pos.quantity, pos.entry_price,
                exit_date, exit_price, holding_days, round(gross, 2),
                round(cost, 2), round(net, 2), exit_reason,
            ])
        return net
