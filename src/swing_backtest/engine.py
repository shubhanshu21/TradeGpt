"""Multi-day swing backtest engine - the swing counterpart to
backtest/engine.py, with the key differences that actually matter for
positional trading rather than intraday:
  - A position can stay open for many days (up to swing.max_holding_days),
    not forced flat by end of day.
  - Stop-loss/target are checked once per day against that day's high/low,
    not on every intraday candle.
  - Costs use backtest/delivery_costs.py (Zerodha CNC/delivery charges),
    not the intraday cost model - they are genuinely different fee
    structures, not just smaller numbers.
  - Same lookahead-avoidance rule as intraday: a signal on day i fills at
    day i+1's open, never day i's own close.
"""
import pandas as pd

from backtest.delivery_costs import delivery_round_trip_cost


class SwingBacktestEngine:
    def __init__(self, cfg: dict, strategy):
        self.cfg = cfg
        self.strategy = strategy

    def run(self, data: dict) -> pd.DataFrame:
        all_trades = []
        for symbol, df in data.items():
            all_trades.extend(self._simulate_symbol(symbol, df))
        accepted = self._apply_portfolio_constraints(all_trades)
        columns = ["symbol", "day", "entry_time", "entry_price", "exit_time",
                   "exit_price", "direction", "quantity", "gross_pnl", "costs",
                   "net_pnl", "exit_reason", "holding_days"]
        return pd.DataFrame(accepted, columns=columns)

    def _simulate_symbol(self, symbol: str, df: pd.DataFrame) -> list:
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)

        try:
            signals = self.strategy.generate_signals(df, symbol=symbol)
        except FileNotFoundError as exc:
            print(f"  [SKIP] {exc}")
            return []
        n = len(signals)

        stop_pct = self.cfg["swing"]["hard_stop_loss_pct"]
        target_pct = self.cfg["swing"]["hard_target_pct"]
        max_holding = self.cfg["swing"]["max_holding_days"]
        capital_per_trade = self.cfg["swing"]["capital_per_trade"]

        trades = []
        in_position = False
        direction = entry_idx = entry_price = quantity = None
        stop_price = target_price = None

        i = 0
        while i < n:
            row = signals.iloc[i]

            if not in_position:
                if row["signal"] != 0 and i + 1 < n:
                    entry_row = signals.iloc[i + 1]
                    price = entry_row["open"]
                    if pd.isna(price) or price <= 0:
                        i += 1
                        continue
                    qty = int(capital_per_trade // price)
                    if qty <= 0:
                        i += 1
                        continue

                    direction = int(row["signal"])
                    entry_price = price
                    quantity = qty
                    entry_idx = i + 1
                    if direction == 1:
                        stop_price = entry_price * (1 - stop_pct)
                        target_price = entry_price * (1 + target_pct)
                    else:
                        stop_price = entry_price * (1 + stop_pct)
                        target_price = entry_price * (1 - target_pct)
                    in_position = True
                    i = entry_idx
                    continue
                i += 1
                continue

            # in_position: check this day for stop/target/max-holding
            day_row = signals.iloc[i]
            if direction == 1:
                hit_stop = day_row["low"] <= stop_price
                hit_target = day_row["high"] >= target_price
            else:
                hit_stop = day_row["high"] >= stop_price
                hit_target = day_row["low"] <= target_price

            holding_days = i - entry_idx
            hit_max_holding = holding_days >= max_holding

            exit_price = exit_reason = None
            if hit_stop:
                exit_price, exit_reason = stop_price, "stop_loss"
            elif hit_target:
                exit_price, exit_reason = target_price, "target"
            elif hit_max_holding:
                exit_price, exit_reason = day_row["close"], "max_holding_period"

            if exit_price is not None:
                trades.append(self._make_trade(
                    symbol, signals.iloc[entry_idx]["date"], entry_price,
                    day_row["date"], exit_price, direction, quantity,
                    exit_reason, holding_days))
                in_position = False
            i += 1

        if in_position:
            last = signals.iloc[-1]
            holding_days = (n - 1) - entry_idx
            trades.append(self._make_trade(
                symbol, signals.iloc[entry_idx]["date"], entry_price,
                last["date"], last["close"], direction, quantity,
                "end_of_data", holding_days))

        return trades

    def _make_trade(self, symbol, entry_time, entry_price, exit_time,
                     exit_price, direction, quantity, exit_reason, holding_days) -> dict:
        gross = (exit_price - entry_price) * quantity * direction
        cost = delivery_round_trip_cost(entry_price, exit_price, quantity, direction, self.cfg)
        return {
            "symbol": symbol, "day": pd.Timestamp(entry_time).date(),
            "entry_time": entry_time, "entry_price": entry_price,
            "exit_time": exit_time, "exit_price": exit_price,
            "direction": direction, "quantity": quantity,
            "gross_pnl": gross, "costs": cost, "net_pnl": gross - cost,
            "exit_reason": exit_reason, "holding_days": holding_days,
        }

    def _apply_portfolio_constraints(self, trades: list) -> list:
        if not trades:
            return trades
        trades = sorted(trades, key=lambda t: t["entry_time"])

        max_positions = self.cfg["swing"]["max_concurrent_positions"]
        open_exit_times = []
        accepted = []

        for t in trades:
            open_exit_times = [et for et in open_exit_times if et > t["entry_time"]]
            if len(open_exit_times) >= max_positions:
                continue
            accepted.append(t)
            open_exit_times.append(t["exit_time"])

        return accepted
