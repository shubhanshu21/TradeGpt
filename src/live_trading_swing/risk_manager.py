"""Portfolio-level guardrails for swing live trading, enforced independently
of whatever a strategy signals. A strategy saying "enter" is necessary but
never sufficient - this is the layer that can veto it.

Swing doesn't have a clean "daily loss" concept the way intraday did
(a position can still be open and unrealized for weeks), so the kill
switch here is cumulative realized loss since this process started,
against swing.max_total_loss_pct.
"""
from exchange.time_utils import now_ist


class RiskManager:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.realized_pnl_total = 0.0
        self.kill_switch_tripped = False

    def can_open_new(self, open_position_count: int) -> bool:
        if self.kill_switch_tripped:
            return False
        max_positions = self.cfg["swing"]["max_concurrent_positions"]
        max_total_loss = (self.cfg["swing"]["max_total_loss_pct"]
                          * self.cfg["swing"]["initial_capital"])

        if open_position_count >= max_positions:
            return False
        if self.realized_pnl_total <= -max_total_loss:
            self.kill_switch_tripped = True
            print(f"[{now_ist()}] RISK MANAGER: cumulative loss limit breached - "
                  "kill switch tripped, no new entries.")
            return False
        return True

    def record_realized(self, net_pnl: float):
        self.realized_pnl_total += net_pnl
