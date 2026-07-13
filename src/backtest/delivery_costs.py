"""Zerodha equity DELIVERY (CNC) cost model - genuinely different from
intraday (backtest/costs.py), not just a smaller version of it:
  - Brokerage is Rs 0 (Zerodha charges nothing for delivery trades).
  - STT is 0.1% on BOTH the buy and sell leg (intraday is 0.025% sell-only).
  - Stamp duty is 0.015% on buy (vs 0.003% intraday).
  - A flat per-sell DP (depository participant) charge applies - roughly
    Rs 13.5 + 18% GST, charged by CDSL/NSDL regardless of quantity or value.
    Intraday trades never touch the depository so never pay this; swing/
    delivery trades pay it on every single exit, and it's a meaningful drag
    on smaller trade sizes since it doesn't scale down with position size.
Numbers reflect Zerodha's published charges as of this project's writing -
re-check against zerodha.com/charges before trusting absolute results.
"""
from dataclasses import dataclass


@dataclass
class DeliveryLegCost:
    brokerage: float
    stt: float
    exchange_txn: float
    sebi: float
    stamp_duty: float
    dp_charge: float
    gst: float

    @property
    def total(self) -> float:
        return (self.brokerage + self.stt + self.exchange_txn + self.sebi
                + self.stamp_duty + self.dp_charge + self.gst)


def delivery_leg_cost(turnover: float, side: str, cfg: dict) -> DeliveryLegCost:
    c = cfg["swing"]["costs"]

    brokerage = 0.0  # Zerodha: free equity delivery
    stt = turnover * c["stt_pct"]  # both sides for delivery
    exchange_txn = turnover * c["exchange_txn_pct"]
    sebi = turnover * c["sebi_pct"]
    stamp_duty = turnover * c["stamp_duty_buy_pct"] if side == "buy" else 0.0
    # dp_charge_flat (config/settings.yaml) is already GST-inclusive (~Rs 13.5 + 18% GST
    # = 15.93) - it must NOT be included in the gst base below, or GST gets charged on
    # top of an amount that already has GST baked into it.
    dp_charge = c["dp_charge_flat"] if side == "sell" else 0.0
    gst = (brokerage + exchange_txn + sebi) * c["gst_pct"]

    return DeliveryLegCost(brokerage, stt, exchange_txn, sebi, stamp_duty, dp_charge, gst)


def delivery_round_trip_cost(entry_price: float, exit_price: float, quantity: int,
                              direction: int, cfg: dict) -> float:
    """direction: 1 for long (buy then sell), -1 for short (sell then buy) -
    note intraday-style shorting isn't available on CNC/delivery in practice
    (you'd need equity F&O or specific short-delivery products), but the
    math is kept symmetric here in case a strategy is tested short-only for
    comparison purposes.
    """
    entry_side = "buy" if direction == 1 else "sell"
    exit_side = "sell" if direction == 1 else "buy"

    entry_turnover = entry_price * quantity
    exit_turnover = exit_price * quantity

    entry_costs = delivery_leg_cost(entry_turnover, entry_side, cfg)
    exit_costs = delivery_leg_cost(exit_turnover, exit_side, cfg)

    return entry_costs.total + exit_costs.total
